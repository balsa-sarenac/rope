"""Architecture-facing change signature (research POC).

This module realizes the *composite* form of the reference
architecture over rope's change signature.  It introduces no parallel
object model: the composite's children are rope's own
`_ArgumentChanger` instances, promoted to transformations in
`rope.refactor.change_signature`.  Each child states its own
applicability, resolves its own target and constructs its own
`ChangeSet`.

`ChangeSignatureTransformation` orders the children and executes them
one after another against an `arch.PendingChanges` view, so a child is
checked and rewritten against the program its predecessors produced
rather than against the original.  Applicability is therefore checked
*by* the children at their own point in the sequence and is never
aggregated up front: aggregation is unsound whenever one child's
applicability depends on what an earlier child creates.  The composite
keeps one check of its own -- that the target is a function.

The behavioral level is a class choice per child.  A plain changer is
a transformation; a changer wrapped in `AddParameterRefactoring` or
`RemoveParameterRefactoring` is a refactoring carrying the
corresponding commitment, and a composite may mix the two as the
reference realization does.  The composite runs every child at its
transformation level and hoists the refactoring-level children's
commitments into `ChangeSignatureRefactoring`, so the caller's policy
decides once what a warning means.  `ChangeSignature.get_changes`
passes plain changers, which is why the compatibility interface is
unaffected.

A changer that cannot be applied raises at its own check, so the
sequence stops at the first genuine failure rather than reporting
later children against a signature that was never produced.
"""


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

    def subjects(self):
        return [pair[0] for pair in self.definition_info.args_with_defaults]

    def _find_violators(self):
        return [
            name for name in self.subjects() if name == self.parameter_name
        ]

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

    def subjects(self):
        """Exactly the slots `ArgumentRemover` edits."""
        info = self.definition_info
        named_count = len(info.args_with_defaults)
        slots = list(range(named_count))
        if info.args_arg is not None:
            slots.append(named_count)
        if info.keywords_arg is not None:
            slots.append(named_count + (1 if info.args_arg is not None else 0))
        return slots

    def _find_violators(self):
        return [] if self.index in self.subjects() else [self.index]

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

    def subjects(self):
        return list(self.new_order)

    def _find_violators(self):
        named_count = len(self.definition_info.args_with_defaults)
        violators = [
            index for index in self.new_order if not 0 <= index < named_count
        ]
        # an order longer than the signature names slots that cannot be
        # reordered, whichever indices they carry
        violators.extend(self.new_order[named_count:])
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

    def subjects(self):
        return list(range(len(self.definition_info.args_with_defaults)))

    def _find_violators(self):
        return [] if self.index in self.subjects() else [self.index]

    def error_string(self):
        info = self.definition_info
        return (
            f"No parameter at index <{self.index}> to inline;"
            f" the signature at this step is {info.to_string()}."
        )


class ChangeSignatureTransformation(arch.Transformation):
    """A signature change as an ordered sequence of executable children.

    The children are rope's own argument changers, which are
    elementary transformations: each resolves its target, states its
    applicability, and constructs its own `ChangeSet`.  They run
    against the `PendingChanges` view the composite carries, so a
    child is checked and rewritten against the program its
    predecessors produced, not against the original.  Applicability is
    therefore checked *by* the children at their own point in the
    sequence, never aggregated up front against a program that no
    longer describes what the child will meet.

    A caller composes plain changers for the behavior-agnostic level
    and changers wrapped in `RemoveParameterRefactoring` or
    `AddParameterRefactoring` where it wants the behavior-preserving
    commitment; the composite runs every child at its transformation
    level and hoists the refactoring-level commitments to its own
    decorator.  Choosing the level is therefore a class choice per
    child, exactly as it is for whole operations.
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
        self.children = self.changers
        self._reject_foreign_children()
        self._reject_non_functions()
        self._prepared = True

    def _reject_foreign_children(self):
        """Every child must be one of rope's argument changers.

        `get_changes` documents `changers` as `_ArgumentChanger`s.  An
        object outside that hierarchy provides the two edit functions
        but cannot act as a child, which resolves its own target and
        constructs its own changes; it is refused here, as a
        preparation failure, rather than by an attribute error later.
        """
        for child in self.children:
            if not isinstance(_transformation_of(child), arch.Transformation):
                raise exceptions.RefactoringError(
                    "Change signature children must be argument changers;"
                    f" got {type(child).__name__}"
                )

    def _reject_non_functions(self):
        """The target must be a function before any child runs.

        A preparation failure, not a reified condition, mirroring the
        rename transformation's method check and the reference
        realization's `refactoringError` during `prepareForExecution`.
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
                    inner = _transformation_of(child)
                    inner.configure(
                        self.project,
                        self.resource,
                        self.offset,
                        self.resources,
                        self.in_hierarchy,
                        pending,
                    )
                    inner.prepare_for_execution()
                    inner.check_preconditions()
                    pending.absorb(inner.private_transform())
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

    def child_transformations(self):
        """The children at their transformation level.

        A child may be a refactoring decorating one; the composite
        constructs changes at the transformation level and leaves the
        commitments to its own decorator.
        """
        return [_transformation_of(child) for child in self.children]

    def unsure_occurrences(self):
        self.run()
        return [
            occurrence
            for child in self.child_transformations()
            for occurrence in child.unsure_occurrences
        ]

    def applicability_preconditions(self):
        """None: each child states and checks its own.

        The children's conditions are deliberately absent here: a
        child's applicability is meaningful against the program its
        predecessors produce, so checking them from this list -- up
        front, against the original -- would reject sequences that in
        fact succeed.  What they checked is reported through
        `internally_checked_preconditions`.
        """
        return []

    def internally_checked_preconditions(self):
        """What the children checked, each at its own point."""
        return [
            condition
            for child in self.child_transformations()
            for condition in child.checked_applicability
        ]

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

    def subjects(self):
        return list(self.child.call_records)

    def _find_violators(self):
        info = self.child.definition_info
        index = self.child.index
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
        name = info.args_with_defaults[self.child.index][0]
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
        self.changer = child

    def subjects(self):
        return list(self.child.call_records)

    def _find_violators(self):
        if self.changer.default is not None or self.changer.value is not None:
            return []
        return self.subjects()

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

    It states no `subjects`: its range is every definition of the name
    in the analyzed resources, which only a second pass without the
    self-exclusion filter would enumerate.
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


class RemoveParameterRefactoring(arch.Refactoring):
    """Behavior-preserving parameter removal.

    Decorates an argument changer, which constructs its own changes;
    only the commitment is added here.
    """

    def _build_breaking_change_preconditions(self):
        return [NoArgumentValueLostCondition(self.transformation)]


class AddParameterRefactoring(arch.Refactoring):
    """Behavior-preserving parameter addition."""

    def _build_breaking_change_preconditions(self):
        return [CallSitesReceiveRequiredArgumentCondition(self.transformation)]


def _transformation_of(child):
    """The transformation level of a child that may be a refactoring."""
    return getattr(child, "transformation", child)


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
        # The children must have run: each contributes conditions about
        # the call sites its own rewrite met, and a call site is only
        # known once the rewrite has looked for it.  Constructing
        # changes before consulting the commitments inverts the order
        # of the reference realization, but not its guarantee: changes
        # are constructed, never applied, until the caller's policy has
        # seen the warnings.
        self.transformation.run()
        conditions = [
            self.hierarchy_overrides_condition(),
            self.unsure_occurrences_condition(),
            self.analysis_coverage_condition(),
        ]
        for child in self.transformation.children:
            if isinstance(child, arch.Refactoring):
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

