"""Architecture-facing change signature (research POC).

This module retrofits the reference architecture's *precondition*
layering onto rope's change signature.  A
`ChangeSignatureTransformation` coordinates an ordered sequence of
`SignatureStep` objects -- one per legacy argument changer -- and
derives its applicability from theirs instead of re-implementing it.
A `ChangeSignatureRefactoring` decorates the composite with
behavior-preserving preconditions, some held at the composite level
(properties of the shared occurrence scope) and some contributed by
step refactorings (`RemoveParameterRefactoring`,
`AddParameterRefactoring`).

**A step is not a transformation.**  It constructs no `ChangeSet` and
cannot execute; it pairs a legacy `_ArgumentChanger` with the
applicability conditions that changer never had.  The composite folds
every changer into one occurrence pass (`_SignatureAnalysis`), which
is what keeps its output byte-identical to the legacy path: the same
finder configuration, the same `_FunctionChangers` semantics and the
same `_ChangeCallsInModule` rewriting run exactly once.

The composite form of the reference architecture -- independently
executable children, each seeing the previous child's edits -- does
not transfer, and the reason is a property of the host engine rather
than of this port.  Pharo composes through `RBNamespace`, a
change-scoped overlay of the program model that lets child *i+1* read
child *i*'s pending edits without touching the image.  Rope's change
model is write-only: a `ChangeSet` is applied by `project.do()` and is
never a view anything can read back.  Composition of independently
executable transformations therefore presupposes an abstraction rope
does not have.  What transfers instead is the layering: ordered
parameter-level units, each carrying reified conditions checked
against the signature the prior units produce.

Late configuration is what makes that checkable.  A step cannot be
validated up front -- an index in range, a name not duplicated, are
only meaningful against the signature produced by the *prior* steps --
so the composite binds each step during its own
``prepare_for_execution``, the late-instantiation hook the reference
architecture introduced for composition.  The fold that produces those
signatures lives in one place, `ChangeSignatureTransformation.definition_infos`.

A step whose changer cannot be applied propagates its input signature
unchanged, so later index conditions may mis-report alongside the true
failure; the driver aggregates all failures, so the first reported
condition is always a real one.
"""

import copy

from rope.base import exceptions, pyobjects, taskhandle
from rope.base.change import ChangeContents, ChangeSet
from rope.refactor import arch, functionutils, occurrences
from rope.refactor import change_signature as legacy
from rope.refactor.change_signature import (
    ArgumentAdder,
    ArgumentDefaultInliner,
    ArgumentNormalizer,
    ArgumentRemover,
    ArgumentReorderer,
)


class SignatureViolation:
    """A violator: a parameter slot that fails against the current signature."""

    def __init__(self, subject, definition_info):
        self.subject = subject
        self.definition_info = definition_info


class StepCondition(arch.Condition):
    """A condition over one step, checked against the step's input signature."""

    def __init__(self, step):
        super().__init__()
        self.step = step


class NoDuplicateParameterCondition(StepCondition):
    """The added parameter name is not already in the signature.

    Checked against the signature produced by the prior steps; the
    error string reproduces the legacy `ArgumentAdder` message.
    """

    name = "no-duplicate-parameter"
    level = arch.APPLICABILITY

    def _find_violators(self):
        info = self.step.input_definition_info
        name = self.step.changer.name
        for pair in info.args_with_defaults:
            if pair[0] == name:
                return [SignatureViolation(name, info)]
        return []

    def error_string(self):
        return "Adding duplicate parameter: <%s>." % self.step.changer.name


class ParameterExistsCondition(StepCondition):
    """The removed index denotes an existing parameter slot.

    Replicates exactly the slots `ArgumentRemover` edits: a named
    parameter, the ``*args`` slot at ``len(args)``, or the ``**kwargs``
    slot just after it.  The legacy path silently no-ops outside these
    slots; here the missing slot is a reified applicability failure.
    """

    name = "parameter-exists"
    level = arch.APPLICABILITY

    def _find_violators(self):
        info = self.step.input_definition_info
        index = self.step.changer.index
        named_count = len(info.args_with_defaults)
        if 0 <= index < named_count:
            return []
        if index == named_count and info.args_arg is not None:
            return []
        if (
            index == named_count
            and info.args_arg is None
            and info.keywords_arg is not None
        ) or (
            index == named_count + 1
            and info.args_arg is not None
            and info.keywords_arg is not None
        ):
            return []
        return [SignatureViolation(index, info)]

    def error_string(self):
        info = self.step.input_definition_info
        return (
            f"No parameter at index <{self.step.changer.index}> to remove;"
            f" the signature at this step is {info.to_string()}."
        )


class ReorderIndicesValidCondition(StepCondition):
    """Every reorder index denotes an existing named parameter.

    The legacy path accepts prefix reorders (fewer indices than
    parameters), so only the indices themselves are validated; an
    invalid index was a raw IndexError at rewrite time.
    """

    name = "reorder-indices-valid"
    level = arch.APPLICABILITY

    def _find_violators(self):
        info = self.step.input_definition_info
        named_count = len(info.args_with_defaults)
        new_order = self.step.changer.new_order
        violators = [
            SignatureViolation(index, info)
            for index in new_order
            if not 0 <= index < named_count
        ]
        if len(new_order) > named_count:
            violators.append(SignatureViolation(new_order, info))
        return violators

    def error_string(self):
        info = self.step.input_definition_info
        return (
            f"Invalid parameter ordering <{self.step.changer.new_order}>;"
            f" the signature at this step is {info.to_string()}."
        )


class ParameterIndexInRangeCondition(StepCondition):
    """The inlined index denotes an existing named parameter.

    The legacy path crashed with an IndexError during call rewriting;
    the missing precondition is regained as a reified condition.
    """

    name = "parameter-index-in-range"
    level = arch.APPLICABILITY

    def _find_violators(self):
        info = self.step.input_definition_info
        index = self.step.changer.index
        if 0 <= index < len(info.args_with_defaults):
            return []
        return [SignatureViolation(index, info)]

    def error_string(self):
        info = self.step.input_definition_info
        return (
            f"No parameter at index <{self.step.changer.index}> to inline;"
            f" the signature at this step is {info.to_string()}."
        )


class SignatureStep:
    """One parameter-level unit of a signature change.

    A step is not a transformation: it has no `private_transform` and
    constructs no `ChangeSet`.  It pairs a legacy `_ArgumentChanger`
    -- wrapped unmodified, so the composite's rewrite is the exact
    legacy rewrite -- with the applicability conditions that changer
    never had.

    ``bind()`` is the late-configuration point: the composite supplies
    the signature produced by the prior steps, which is the only
    signature this step's conditions are meaningful against.
    """

    changer_class = None

    def __init__(self, *args, changer=None):
        self.changer = changer if changer is not None else self.changer_class(*args)
        self.composite = None
        self.input_definition_info = None

    def bind(self, composite, input_definition_info):
        self.composite = composite
        self.input_definition_info = input_definition_info

    def applicability_preconditions(self):
        return []

    def check_preconditions(self):
        if self.input_definition_info is None:
            raise exceptions.RefactoringError(
                "Signature step used before being bound to a composite."
            )
        arch.check_applicability_preconditions(self)


class NormalizeParametersStep(SignatureStep):
    changer_class = ArgumentNormalizer


class AddParameterStep(SignatureStep):
    """Add a parameter at `index`.

    No index condition: the legacy ``list.insert`` clamps out-of-range
    indices, so every index is applicable.
    """

    changer_class = ArgumentAdder

    def applicability_preconditions(self):
        return [
            arch.ValidNameCondition(self.changer.name),
            NoDuplicateParameterCondition(self),
        ]


class RemoveParameterStep(SignatureStep):
    changer_class = ArgumentRemover

    def applicability_preconditions(self):
        return [ParameterExistsCondition(self)]


class ReorderParametersStep(SignatureStep):
    changer_class = ArgumentReorderer

    def applicability_preconditions(self):
        return [ReorderIndicesValidCondition(self)]


class InlineParameterDefaultStep(SignatureStep):
    changer_class = ArgumentDefaultInliner

    def applicability_preconditions(self):
        return [ParameterIndexInRangeCondition(self)]


class GenericChangerStep(SignatureStep):
    """Compatibility fallback wrapping a custom `_ArgumentChanger`.

    Contributes no conditions; an edit-time `RefactoringError` from
    the custom changer still surfaces, as on the legacy path.
    """


class ChangeSignatureTransformation(arch.Transformation):
    """Behavior-agnostic signature change over ordered parameter steps.

    Applicability is derived from the steps' preconditions by
    aggregation, never re-implemented here; the target being a
    function is a hard `prepare_for_execution` failure, exactly as
    method-ness is for the rename transformation.
    """

    def __init__(
        self,
        project,
        resource,
        offset,
        steps,
        in_hierarchy=False,
        resources=None,
        task_handle=taskhandle.DEFAULT_TASK_HANDLE,
    ):
        self.project = project
        self.resource = resource
        self.offset = offset
        self.steps = steps
        self.in_hierarchy = in_hierarchy
        self._given_resources = resources
        self.resources = None
        self.task_handle = task_handle
        self.name = None
        self.primary = None
        self.pyname = None
        self.others = None
        self.pyfunction = None
        self.definition_info = None
        self.changes = None
        self.analysis = None
        self._definition_info_fold = None
        # protocol attributes for the shared arch conditions
        self.old_name = None
        self.docs = False
        self._prepared = False

    def prepare_for_execution(self):
        """Resolve the target and bind each step to its input signature.

        Binding happens here -- not at construction -- because step
        *i*'s preconditions are only meaningful against the signature
        produced by steps *0..i-1*, which does not exist before the
        composite resolves the target and projects the earlier steps.
        """
        if self._prepared:
            return
        (self.name, self.primary, self.pyname, self.others) = (
            legacy._resolve_signature_target(self.project, self.resource, self.offset)
        )
        if (
            self.pyname is None
            or self.pyname.get_object() is None
            or not isinstance(self.pyname.get_object(), pyobjects.PyFunction)
        ):
            raise exceptions.RefactoringError(
                "Change method signature should be performed on functions"
            )
        self.old_name = self.name
        self.pyfunction = self.pyname.get_object()
        self.definition_info = functionutils.DefinitionInfo.read(self.pyfunction)
        if self._given_resources is None:
            self.resources = self.project.get_python_files()
        else:
            self.resources = self._given_resources
        for step, info in zip(self.unwrapped_steps(), self.definition_infos()):
            step.bind(self, info)
        self.analysis = _SignatureAnalysis(self)
        self._prepared = True

    def unwrapped_steps(self):
        """The steps, refactoring-decorated ones unwrapped."""
        return [getattr(step, "transformation", step) for step in self.steps]

    def definition_infos(self):
        """The signature each step sees: one fold, computed once.

        Entry *i* is the signature produced by steps ``0..i-1``.  This
        is the only projection in the module; every step condition
        reads its input from here, so a condition can never disagree
        with the signature the composite believes in.

        A changer that cannot be applied to its input -- the legacy
        duplicate-add validation, the raw `IndexError` of an
        out-of-range reorder or inline -- contributes its input
        unchanged, and its own reified condition reports the failure
        at check time.
        """
        if self._definition_info_fold is None:
            infos = [self.definition_info]
            for step in self.unwrapped_steps():
                projected = copy.deepcopy(infos[-1])
                try:
                    step.changer.change_definition_info(projected)
                except (exceptions.RefactoringError, IndexError):
                    projected = infos[-1]
                infos.append(projected)
            self._definition_info_fold = infos
        return self._definition_info_fold

    def is_method(self):
        return isinstance(self.pyfunction.parent, pyobjects.PyClass)

    def applicability_preconditions(self):
        return [
            condition
            for step in self.unwrapped_steps()
            for condition in step.applicability_preconditions()
        ]

    def private_transform(self):
        """Construct the `ChangeSet` from the shared signature analysis."""
        self.analysis.ensure_ran()
        changes = ChangeSet("Changing signature of <%s>" % self.name)
        for file_, new_content in self.analysis.new_contents:
            changes.add_change(ChangeContents(file_, new_content))
        self.changes = changes
        return changes


class NoArgumentValueLostCondition(StepCondition):
    """No call site supplies a value for the removed parameter.

    The legacy rewrite silently drops such a value.  The check replays
    the argument mapping of each recorded call site through the steps
    *before* this one and flags the sites where the removed parameter
    still receives an explicit value.  Only named-parameter removal is
    checked; the ``*args``/``**kwargs`` slots are out of scope.
    """

    name = "no-argument-value-lost"
    level = arch.BEHAVIOR_PRESERVING

    def _find_violators(self):
        step = self.step
        composite = step.composite
        analysis = composite.analysis
        analysis.ensure_ran()
        info = step.input_definition_info
        index = step.changer.index
        if not 0 <= index < len(info.args_with_defaults):
            return []
        removed_name = info.args_with_defaults[index][0]
        position = composite.unwrapped_steps().index(step)
        preceding = composite.unwrapped_steps()[:position]
        infos = composite.definition_infos()
        violators = []
        for record in analysis.call_records:
            call_info = functionutils.CallInfo.read(
                record.primary, record.pyname, composite.definition_info, record.code
            )
            mapping = functionutils.ArgumentMapping(
                composite.definition_info, call_info
            )
            for definition_info, preceding_step in zip(infos, preceding):
                preceding_step.changer.change_argument_mapping(
                    definition_info, mapping
                )
            if removed_name in mapping.param_dict:
                violators.append(record)
        return violators

    def error_string(self):
        places = ", ".join(
            f"{record.resource.path}:{record.lineno}" for record in self.violators
        )
        info = self.step.input_definition_info
        name = info.args_with_defaults[self.step.changer.index][0]
        return (
            f"Removing parameter <{name}> drops an explicitly passed"
            f" argument at: {places}"
        )


class CallSitesReceiveRequiredArgumentCondition(StepCondition):
    """Existing calls receive a value for the added parameter.

    Holds when the parameter has a default or an injected value;
    otherwise every rewritten call omits a required argument and fails
    with a TypeError at runtime.  Call sites forwarding ``**kwargs``
    are flagged too -- the check is deliberately conservative.
    """

    name = "call-sites-receive-required-argument"
    level = arch.BEHAVIOR_PRESERVING

    def _find_violators(self):
        changer = self.step.changer
        if changer.default is not None or changer.value is not None:
            return []
        analysis = self.step.composite.analysis
        analysis.ensure_ran()
        return list(analysis.call_records)

    def error_string(self):
        places = ", ".join(
            f"{record.resource.path}:{record.lineno}" for record in self.violators
        )
        return (
            f"Added parameter <{self.step.changer.name}> has no default"
            f" and no value; existing calls would fail at: {places}"
        )


class HierarchyOverridesUpdatedCondition(arch.TransformationCondition):
    """Every hierarchy definition of the method is updated.

    With ``in_hierarchy=False`` only the selected definition is
    rewritten; definitions of the same method elsewhere in the class
    hierarchy keep the old signature, breaking polymorphic call sites.
    This is the change-signature analog of the reference
    architecture's arity condition, which method rename has no use
    for.  Held at the composite level: it is a property of the shared
    occurrence scope, not of any single step.  Checking runs an extra
    definitions pass over the analyzed resources.
    """

    name = "hierarchy-overrides-updated"
    level = arch.BEHAVIOR_PRESERVING

    _shape_preserving = (ArgumentNormalizer, ArgumentDefaultInliner)


    def _changes_definition_shape(self):
        return any(
            not isinstance(step.changer, self._shape_preserving)
            for step in self.transformation.unwrapped_steps()
        )

    def _find_violators(self):
        transformation = self.transformation
        if not transformation.is_method() or transformation.in_hierarchy:
            return []
        if not self._changes_definition_shape():
            return []
        pyname = transformation.pyname

        def is_defined(occurrence):
            if not occurrence.is_defined():
                return False

        def not_self(occurrence):
            if occurrence.get_pyname().get_object() == pyname.get_object():
                return False

        finder = occurrences.Finder(
            transformation.project,
            transformation.name,
            filters=[is_defined, not_self, occurrences.InHierarchyFilter(pyname)],
        )
        violators = []
        for resource in transformation.resources:
            for occurrence in finder.find_occurrences(resource):
                violators.append((occurrence.resource, occurrence.lineno))
        return violators

    def error_string(self):
        places = ", ".join(
            f"{resource.path}:{lineno}" for resource, lineno in self.violators
        )
        return (
            f"<{self.transformation.name}> is defined elsewhere in the"
            f" class hierarchy and keeps the old signature: {places}"
        )


class RemoveParameterRefactoring(arch.Refactoring):
    """Behavior-preserving parameter removal (a composite step).

    Decorates a `SignatureStep`, not a transformation: the step is
    what carries conditions, and `generate_changes` is the composite's
    job.  Only the precondition half of the decorator is meaningful
    here.
    """

    def __init__(self, *args, changer=None):
        super().__init__(RemoveParameterStep(*args, changer=changer))

    def _build_breaking_change_preconditions(self):
        return [self.argument_value_lost_condition()]

    def argument_value_lost_condition(self):
        return NoArgumentValueLostCondition(self.transformation)


class AddParameterRefactoring(arch.Refactoring):
    """Behavior-preserving parameter addition (a composite step).

    Decorates a `SignatureStep`; see `RemoveParameterRefactoring`.
    """

    def __init__(self, *args, changer=None):
        super().__init__(AddParameterStep(*args, changer=changer))

    def _build_breaking_change_preconditions(self):
        return [self.required_argument_condition()]

    def required_argument_condition(self):
        return CallSitesReceiveRequiredArgumentCondition(self.transformation)


class ChangeSignatureRefactoring(arch.Refactoring):
    """Behavior-preserving signature change.

    Decorates the composite transformation.  The behavior-preserving
    commitment is assembled from two sources, as the reference
    architecture prescribes: conditions held at the composite level
    (properties of the shared occurrence scope and analysis) and
    conditions contributed by whichever steps are refactorings.
    """

    def __init__(self, *args, **kwds):
        super().__init__(ChangeSignatureTransformation(*args, **kwds))

    def _build_breaking_change_preconditions(self):
        conditions = [
            self.hierarchy_overrides_condition(),
            self.unsure_occurrences_condition(),
            self.analysis_coverage_condition(),
        ]
        for step in self.transformation.steps:
            if hasattr(step, "breaking_change_preconditions"):
                conditions.extend(step.breaking_change_preconditions())
        return conditions

    def hierarchy_overrides_condition(self):
        return HierarchyOverridesUpdatedCondition(self.transformation)

    def unsure_occurrences_condition(self):
        return arch.NoUnsureOccurrencesCondition(self.transformation)

    def analysis_coverage_condition(self):
        return arch.AnalysisCoversAllClientsCondition(self.transformation)


_STEP_CLASSES = [
    (ArgumentRemover, RemoveParameterRefactoring),
    (ArgumentAdder, AddParameterRefactoring),
    (ArgumentReorderer, ReorderParametersStep),
    (ArgumentDefaultInliner, InlineParameterDefaultStep),
    (ArgumentNormalizer, NormalizeParametersStep),
]


def steps_for_changers(changers):
    """Wrap legacy argument changers as composite steps.

    The caller's changer instance becomes the step's payload -- never
    reconstructed -- so stateful changers
    (`ArgumentDefaultInliner.remove`) and custom subclasses behave
    exactly as on the legacy path.  Adder and remover changers get
    their refactoring flavor so non-legacy drivers see their
    behavior-preserving conditions; the legacy policy never consults
    them.
    """
    return [_step_for_changer(changer) for changer in changers]


def _step_for_changer(changer):
    for changer_class, step_class in _STEP_CLASSES:
        if isinstance(changer, changer_class):
            return step_class(changer=changer)
    return GenericChangerStep(changer=changer)


class _CallRecord:
    """A violator: a call site matched by the shared occurrence pass."""

    def __init__(self, resource, lineno, primary, pyname, code):
        self.resource = resource
        self.lineno = lineno
        self.primary = primary
        self.pyname = pyname
        self.code = code


class _SignatureAnalysis(arch.OccurrenceAnalysis):
    """The change-signature pass, recording unsure and matched calls.

    Reproduces the legacy `ChangeSignature._change_calls` pass exactly
    -- the same finder configuration, the same `_FunctionChangers`
    edit chain, the same `_ChangeCallsInModule` rewriting -- so the
    composite's output is byte-identical to the legacy path.
    """

    def __init__(self, transformation):
        super().__init__(transformation)
        self.call_records = []
        self.function_changers = None

    def _build_finder(self):
        transformation = self.transformation
        finder = occurrences.create_finder(
            transformation.project,
            transformation.name,
            transformation.pyname,
            instance=transformation.primary,
            in_hierarchy=transformation.in_hierarchy and transformation.is_method(),
            unsure=self.record_unsure,
        )
        if transformation.others:
            name, pyname = transformation.others
            constructor_finder = occurrences.create_finder(
                transformation.project, name, pyname, only_calls=True
            )
            finder = legacy._MultipleFinders([finder, constructor_finder])
        self.function_changers = legacy._FunctionChangers(
            transformation.pyfunction,
            transformation.definition_info,
            [step.changer for step in transformation.unwrapped_steps()],
        )
        return finder

    def _rewrite(self, finder, resource):
        change_calls = legacy._ChangeCallsInModule(
            self.transformation.project,
            finder,
            resource,
            self.function_changers,
            observer=self._record,
        )
        return change_calls.get_changed_module()

    def _record(self, occurrence, code):
        if not occurrence.is_called():
            return
        primary, pyname = occurrence.get_primary_and_pyname()
        self.call_records.append(
            _CallRecord(occurrence.resource, occurrence.lineno, primary, pyname, code)
        )
