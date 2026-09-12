"""Architecture-facing change signature (research POC).

This module retrofits the reference architecture's *precondition*
layering onto rope's change signature.  It introduces no parallel
object model: the composite's children are rope's own
`_ArgumentChanger` instances, which now state their own applicability
and behavior-preserving conditions (see
`rope.refactor.change_signature`).  A `ChangeSignatureTransformation`
orders them and derives its applicability from theirs instead of
re-implementing it; a `ChangeSignatureRefactoring` decorates the
composite with behavior-preserving conditions, some held at the
composite level (properties of the shared occurrence scope) and some
contributed by the changers themselves.

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
parameter-level edits, each carrying reified conditions checked
against the signature the prior edits produce, folded into one
occurrence pass (`_SignatureAnalysis`).  That fold is also what keeps
the output byte-identical to the legacy path: the same finder
configuration, the same `_FunctionChangers` semantics and the same
`_ChangeCallsInModule` rewriting run exactly once.

Late configuration is what makes the conditions checkable.  A changer
cannot be validated up front -- an index in range, a name not
duplicated, are only meaningful against the signature produced by the
*prior* changers -- so the composite supplies each changer that
signature when it asks for its conditions.  The fold that produces
them lives in one place,
`ChangeSignatureTransformation.definition_infos`.

A changer that cannot be applied propagates its input signature
unchanged, so later index conditions may mis-report alongside the true
failure; the driver aggregates all failures, so the first reported
condition is always a real one.
"""

import copy

from rope.base import codeanalyze, exceptions, pyobjects, taskhandle, worder
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


class NoDuplicateParameterCondition(arch.Condition):
    """The added parameter name is not already in the signature.

    Checked against the signature produced by the prior steps; the
    error string reproduces the legacy `ArgumentAdder` message.
    """

    name = "no-duplicate-parameter"
    level = arch.APPLICABILITY

    def __init__(self, definition_info, name):
        super().__init__()
        self.definition_info = definition_info
        self.parameter_name = name

    def _find_violators(self):
        info = self.definition_info
        for pair in info.args_with_defaults:
            if pair[0] == self.parameter_name:
                return [SignatureViolation(self.parameter_name, info)]
        return []

    def error_string(self):
        return "Adding duplicate parameter: <%s>." % self.parameter_name


class ParameterExistsCondition(arch.Condition):
    """The removed index denotes an existing parameter slot.

    Replicates exactly the slots `ArgumentRemover` edits: a named
    parameter, the ``*args`` slot at ``len(args)``, or the ``**kwargs``
    slot just after it.  The legacy path silently no-ops outside these
    slots; here the missing slot is a reified applicability failure.
    """

    name = "parameter-exists"
    level = arch.APPLICABILITY

    def __init__(self, definition_info, index):
        super().__init__()
        self.definition_info = definition_info
        self.index = index

    def _find_violators(self):
        info = self.definition_info
        index = self.index
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
        info = self.definition_info
        return (
            f"No parameter at index <{self.index}> to remove;"
            f" the signature at this step is {info.to_string()}."
        )


class ReorderIndicesValidCondition(arch.Condition):
    """Every reorder index denotes an existing named parameter.

    The legacy path accepts prefix reorders (fewer indices than
    parameters), so only the indices themselves are validated; an
    invalid index was a raw IndexError at rewrite time.
    """

    name = "reorder-indices-valid"
    level = arch.APPLICABILITY

    def __init__(self, definition_info, new_order):
        super().__init__()
        self.definition_info = definition_info
        self.new_order = new_order

    def _find_violators(self):
        info = self.definition_info
        named_count = len(info.args_with_defaults)
        new_order = self.new_order
        violators = [
            SignatureViolation(index, info)
            for index in new_order
            if not 0 <= index < named_count
        ]
        if len(new_order) > named_count:
            violators.append(SignatureViolation(new_order, info))
        return violators

    def error_string(self):
        info = self.definition_info
        return (
            f"Invalid parameter ordering <{self.new_order}>;"
            f" the signature at this step is {info.to_string()}."
        )


class ParameterIndexInRangeCondition(arch.Condition):
    """The inlined index denotes an existing named parameter.

    The legacy path crashed with an IndexError during call rewriting;
    the missing precondition is regained as a reified condition.
    """

    name = "parameter-index-in-range"
    level = arch.APPLICABILITY

    def __init__(self, definition_info, index):
        super().__init__()
        self.definition_info = definition_info
        self.index = index

    def _find_violators(self):
        info = self.definition_info
        if 0 <= self.index < len(info.args_with_defaults):
            return []
        return [SignatureViolation(self.index, info)]

    def error_string(self):
        info = self.definition_info
        return (
            f"No parameter at index <{self.index}> to inline;"
            f" the signature at this step is {info.to_string()}."
        )


class ParameterTransformation(arch.Transformation):
    """One argument changer, executable on its own.

    A child of the composite, and a transformation in its own right:
    it resolves its target, states its applicability, and constructs
    its own `ChangeSet`.  It resolves and rewrites against the
    composite's `PendingChanges` view, so it sees the edits of the
    children that ran before it.
    """

    def __init__(self, project, resource, offset, changer, resources, in_hierarchy):
        self.project = project
        self.resource = resource
        self.offset = offset
        self.changer = changer
        self.resources = resources
        self.in_hierarchy = in_hierarchy
        self.pending = None
        self.changes = None
        self.definition_info = None
        self.call_records = []
        self.unsure_occurrences = []

    def prepare_for_execution(self):
        (self.name, self.primary, self.pyname, self.others) = (
            legacy._resolve_signature_target(
                self.project, self.resource, self.offset, pending=self.pending
            )
        )
        if (
            self.pyname is None
            or self.pyname.get_object() is None
            or not isinstance(self.pyname.get_object(), pyobjects.PyFunction)
        ):
            raise exceptions.RefactoringError(
                "Change method signature should be performed on functions"
            )
        self.pyfunction = self.pyname.get_object()
        self.definition_info = functionutils.DefinitionInfo.read(self.pyfunction)

    def is_method(self):
        return isinstance(self.pyfunction.parent, pyobjects.PyClass)

    def applicability_preconditions(self):
        return _conditions(self.changer, "applicability_conditions", self.definition_info)

    def breaking_change_preconditions(self):
        return _conditions(self.changer, "breaking_change_conditions", self)

    def _finder(self):
        finder = occurrences.create_finder(
            self.project,
            self.name,
            self.pyname,
            instance=self.primary,
            in_hierarchy=self.in_hierarchy and self.is_method(),
            unsure=self._record_unsure,
        )
        if self.others:
            name, pyname = self.others
            constructor_finder = occurrences.create_finder(
                self.project, name, pyname, only_calls=True
            )
            finder = legacy._MultipleFinders([finder, constructor_finder])
        return finder

    def _record_unsure(self, occurrence):
        self.unsure_occurrences.append(occurrence)
        return False

    def private_transform(self):
        finder = self._finder()
        changers = legacy._FunctionChangers(
            self.pyfunction, self.definition_info, [self.changer]
        )
        changes = ChangeSet("Changing signature of <%s>" % self.name)
        for file_ in self.resources:
            pymodule = self.pending.pymodule(file_)
            new_content = self._rewrite(finder, pymodule, changers)
            if new_content is not None and new_content != pymodule.source_code:
                changes.add_change(ChangeContents(file_, new_content))
        self.changes = changes
        return changes

    def _rewrite(self, finder, pymodule, changers):
        """`_ChangeCallsInModule.get_changed_module`, over the pending view."""
        source = pymodule.source_code
        word_finder = worder.Worder(source)
        collector = codeanalyze.ChangeCollector(source)
        for occurrence in finder.find_occurrences(pymodule=pymodule):
            if not occurrence.is_called() and not occurrence.is_defined():
                continue
            start, end = occurrence.get_primary_range()
            begin_parens, end_parens = word_finder.get_word_parens_range(end - 1)
            call = source[start:end_parens]
            if occurrence.is_called():
                primary, pyname = occurrence.get_primary_and_pyname()
                self.call_records.append(
                    _CallRecord(
                        occurrence.resource, occurrence.lineno, primary, pyname, call
                    )
                )
                changed = changers.change_call(primary, pyname, call)
            else:
                changed = changers.change_definition(call)
            if changed is not None:
                collector.add_change(start, end_parens, changed)
        return collector.get_changed()


class ChangeSignatureTransformation(arch.Transformation):
    """A signature change as an ordered sequence of executable children.

    Each child is a `ParameterTransformation` that runs against the
    `PendingChanges` view the composite carries, so a child is checked
    and rewritten against the program its predecessors produced --
    not against the original.  Applicability is therefore checked *by*
    the children at their own point in the sequence, never aggregated
    up front against a program that no longer describes what the
    child will meet.
    """

    def __init__(
        self,
        project,
        resource,
        offset,
        changers,
        in_hierarchy=False,
        resources=None,
        task_handle=taskhandle.DEFAULT_TASK_HANDLE,
    ):
        self.project = project
        self.resource = resource
        self.offset = offset
        self.changers = changers
        self.in_hierarchy = in_hierarchy
        self._given_resources = resources
        self.resources = None
        self.task_handle = task_handle
        self.children = []
        self.changes = None
        self._pending = None
        self._prepared = False

    def prepare_for_execution(self):
        if self._prepared:
            return
        self.resources = (
            self._given_resources
            if self._given_resources is not None
            else self.project.get_python_files()
        )
        self.children = [
            ParameterTransformation(
                self.project,
                self.resource,
                self.offset,
                changer,
                self.resources,
                self.in_hierarchy,
            )
            for changer in self.changers
        ]
        self._reject_non_functions()
        self._prepared = True

    def _reject_non_functions(self):
        """The target must be a function before any child runs.

        The composite's only hard applicability check, exactly as
        method-ness is for the rename transformation.  Everything else
        is checked by the children, each against the program its
        predecessors produced.
        """
        _, _, pyname, _ = legacy._resolve_signature_target(
            self.project, self.resource, self.offset
        )
        if (
            pyname is None
            or pyname.get_object() is None
            or not isinstance(pyname.get_object(), pyobjects.PyFunction)
        ):
            raise exceptions.RefactoringError(
                "Change method signature should be performed on functions"
            )

    def run(self):
        """Execute the children into a pending view, once."""
        if self._pending is None:
            pending = arch.PendingChanges(self.project)
            try:
                for child in self.children:
                    child.pending = pending
                    child.prepare_for_execution()
                    child.check_preconditions()
                    pending.absorb(child.private_transform())
            finally:
                pending.restore()
            self._pending = pending
        return self._pending

    def _target(self):
        """Resolve the target against the program as it stands.

        Composite-level conditions are properties of the *original*
        program, so the composite resolves its own target rather than
        borrowing a child's: a child's pyname belongs to a module the
        pending view has since discarded.
        """
        return legacy._resolve_signature_target(
            self.project, self.resource, self.offset
        )

    @property
    def name(self):
        return self._target()[0]

    @property
    def pyname(self):
        return self._target()[2]

    @property
    def definition_info(self):
        return functionutils.DefinitionInfo.read(self.pyname.get_object())

    def is_method(self):
        return isinstance(self.pyname.get_object().parent, pyobjects.PyClass)

    def unsure_occurrences(self):
        self.run()
        return [o for child in self.children for o in child.unsure_occurrences]

    def applicability_preconditions(self):
        """None: each child states and checks its own."""
        return []

    def check_preconditions(self):
        self.run()

    def private_transform(self):
        self.changes = self.run().as_changes(
            "Changing signature of <%s>" % self.name
        )
        return self.changes


class NoArgumentValueLostCondition(arch.Condition):
    """No call site supplies a value for the removed parameter.

    The legacy rewrite silently drops such a value.  The check replays
    the argument mapping of each recorded call site through the steps
    *before* this one and flags the sites where the removed parameter
    still receives an explicit value.  Only named-parameter removal is
    checked; the ``*args``/``**kwargs`` slots are out of scope.
    """

    name = "no-argument-value-lost"
    level = arch.BEHAVIOR_PRESERVING

    def __init__(self, child):
        super().__init__()
        self.child = child

    def _find_violators(self):
        info = self.child.definition_info
        index = self.child.changer.index
        if not 0 <= index < len(info.args_with_defaults):
            return []
        removed_name = info.args_with_defaults[index][0]
        violators = []
        for record in self.child.call_records:
            call_info = functionutils.CallInfo.read(
                record.primary, record.pyname, info, record.code
            )
            mapping = functionutils.ArgumentMapping(info, call_info)
            if removed_name in mapping.param_dict:
                violators.append(record)
        return violators

    def error_string(self):
        places = ", ".join(
            f"{record.resource.path}:{record.lineno}" for record in self.violators
        )
        info = self.child.definition_info
        name = info.args_with_defaults[self.child.changer.index][0]
        return (
            f"Removing parameter <{name}> drops an explicitly passed"
            f" argument at: {places}"
        )


class CallSitesReceiveRequiredArgumentCondition(arch.Condition):
    """Existing calls receive a value for the added parameter.

    Holds when the parameter has a default or an injected value;
    otherwise every rewritten call omits a required argument and fails
    with a TypeError at runtime.  Call sites forwarding ``**kwargs``
    are flagged too -- the check is deliberately conservative.
    """

    name = "call-sites-receive-required-argument"
    level = arch.BEHAVIOR_PRESERVING

    def __init__(self, child):
        super().__init__()
        self.child = child
        self.changer = child.changer

    def _find_violators(self):
        if self.changer.default is not None or self.changer.value is not None:
            return []
        return list(self.child.call_records)

    def error_string(self):
        places = ", ".join(
            f"{record.resource.path}:{record.lineno}" for record in self.violators
        )
        return (
            f"Added parameter <{self.changer.name}> has no default"
            f" and no value; existing calls would fail at: {places}"
        )


class HierarchyOverridesUpdatedCondition(arch.Condition):
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

    def __init__(self, transformation):
        """Takes the transformation: the check ranges over its target,
        its hierarchy, its resources and its steps at once."""
        super().__init__()
        self.transformation = transformation

    def _changes_definition_shape(self):
        return any(
            not isinstance(changer, self._shape_preserving)
            for changer in self.transformation.changers
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


class ChangeSignatureRefactoring(arch.Refactoring):
    """Behavior-preserving signature change.

    Decorates the composite transformation.  The behavior-preserving
    commitment is assembled from two sources, as the reference
    architecture prescribes: conditions held at the composite level
    (properties of the shared occurrence scope) and conditions each
    child contributes about its own edit.
    """

    def __init__(self, *args, **kwds):
        super().__init__(ChangeSignatureTransformation(*args, **kwds))

    def _build_breaking_change_preconditions(self):
        # the children must have run: each contributes conditions about
        # the call sites its own rewrite met
        self.transformation.run()
        conditions = [
            self.hierarchy_overrides_condition(),
            self.unsure_occurrences_condition(),
            self.analysis_coverage_condition(),
        ]
        for child in self.transformation.children:
            conditions.extend(child.breaking_change_preconditions())
        return conditions

    def hierarchy_overrides_condition(self):
        return HierarchyOverridesUpdatedCondition(self.transformation)

    def unsure_occurrences_condition(self):
        return arch.NoUnsureOccurrencesCondition(
            self.transformation, self.transformation.name
        )

    def analysis_coverage_condition(self):
        return arch.AnalysisCoversAllClientsCondition(
            self.transformation.project, self.transformation.resources
        )


def _conditions(changer, hook_name, *args):
    """The conditions a changer contributes through `hook_name`.

    Rope's public API accepts any object providing the two edit
    functions, not only `_ArgumentChanger` subclasses; a changer from
    outside that hierarchy simply contributes no conditions, exactly
    as on the legacy path.
    """
    hook = getattr(changer, hook_name, None)
    return hook(*args) if hook is not None else []


class _CallRecord:
    """A violator: a call site matched by the shared occurrence pass."""

    def __init__(self, resource, lineno, primary, pyname, code):
        self.resource = resource
        self.lineno = lineno
        self.primary = primary
        self.pyname = pyname
        self.code = code
