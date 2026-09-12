"""Architecture-facing method rename (research POC).

This module retrofits a transformation/refactoring/driver architecture
onto rope's method rename while reusing rope's program model
(`Project`, pynames, pyobjects), occurrence engine
(`occurrences.create_finder`) and change model (`ChangeSet`).

It exposes two behavioral levels that share one change function:

* `RenameMethodTransformation` -- applicability preconditions plus
  change construction; the behavior-agnostic level.
* `RenameMethodRefactoring` -- a decorator over the transformation
  that adds behavior-preserving preconditions; violations are
  warnings, not hard failures.

Both levels share a three-layer execution API:

* ``execute()`` = ``generate_changes()`` + ``perform_changes()``
* ``generate_changes()`` = ``prepare_for_execution()`` +
  ``check_preconditions()`` + ``private_transform()``

Preconditions are reified `Condition` objects whose ``check()``
populates ``violators`` with program-model entities, so callers can
inspect the exact cause of a failure instead of receiving a boolean.
"""

from rope.base import evaluate, exceptions, pynames, pyobjects, taskhandle, worder
from rope.base.change import ChangeContents, ChangeSet
from rope.refactor import occurrences

# The generic architecture machinery lives in `rope.refactor.arch`;
# the names are re-exported here so existing callers keep working.
from rope.refactor.arch import (  # noqa: F401
    APPLICABILITY,
    BEHAVIOR_PRESERVING,
    FAIL_ON_WARNING,
    LEGACY,
    POLICIES,
    PROCEED_AFTER_WARNING,
    AnalysisCoversAllClientsCondition,
    BehaviorPreservationWarning,
    Condition,
    NegatedCondition,
    OccurrenceAnalysis,
    NoUnsureOccurrencesCondition,
    RefactoringDriver,
    RefactoringExecutionResult,
    Transformation,
    Refactoring,
    ValidNameCondition,
    check_applicability_preconditions,
)


class ConflictingDefinition:
    """A violator: an existing definition that the new name would clash with."""

    def __init__(self, pyname, pyclass):
        self.pyname = pyname
        self.pyclass = pyclass
        self.module, self.lineno = pyname.get_definition_location()


def _containing_class(pyname):
    """The class whose body defines `pyname`, if any."""
    if isinstance(pyname, pynames.DefinedName):
        scope = pyname.get_object().get_scope()
        parent = scope.parent
        if parent is not None and parent.get_kind() == "Class":
            return parent.pyobject
    return None


class HierarchyDoesNotDefineNameCondition(Condition):
    """The new name does not already resolve in any edited class.

    Checks the selected class and every class whose ``def`` header the
    occurrence analysis renames (with ``in_hierarchy=True`` that
    includes overriding descendants).  `PyClass.get_attributes()`
    merges superclass attributes, so each check covers that class's
    ancestors up to the root.  Checking triggers the shared occurrence
    analysis.
    """

    name = "hierarchy-does-not-define-name"
    level = BEHAVIOR_PRESERVING

    def __init__(self, analysis, pyclass, old_pyname, new_name):
        super().__init__()
        self.analysis = analysis
        self.pyclass = pyclass
        self.old_pyname = old_pyname
        self.new_name = new_name

    def _find_violators(self):
        self.analysis.ensure_ran()
        renamed_locations = {self.old_pyname.get_definition_location()}
        classes = [self.pyclass]
        for occurrence in self.analysis.defining_occurrences:
            pyname = occurrence.get_pyname()
            renamed_locations.add(pyname.get_definition_location())
            pyclass = _containing_class(pyname)
            if pyclass is not None:
                classes.append(pyclass)
        violators = []
        seen = set()
        for pyclass in classes:
            attributes = pyclass.get_attributes()
            if self.new_name not in attributes:
                continue
            definition = attributes[self.new_name]
            # a definition the rename itself edits holds the new name
            # legitimately (renaming to the same name), not conflictingly
            if definition.get_definition_location() in renamed_locations:
                continue
            conflict = ConflictingDefinition(definition, pyclass)
            # the defining module disambiguates same-named classes whose
            # conflicting definitions share a line number
            key = (pyclass.get_name(), conflict.module, conflict.lineno)
            if key not in seen:
                seen.add(key)
                violators.append(conflict)
        return violators

    def error_string(self):
        names = ", ".join(
            sorted({violator.pyclass.get_name() for violator in self.violators})
        )
        return (
            f"'{self.new_name}' is already defined in the class"
            f" hierarchy of: {names}."
        )


class RenameMethodTransformation(Transformation):
    """Behavior-agnostic method rename.

    Owns the applicability preconditions and the change-construction
    function; produces an ordinary rope `ChangeSet`.
    """

    def __init__(
        self,
        project,
        resource,
        offset,
        new_name=None,
        resources=None,
        in_hierarchy=False,
        unsure=None,
        docs=False,
        task_handle=taskhandle.DEFAULT_TASK_HANDLE,
        require_identifier=True,
    ):
        self.project = project
        self.resource = resource
        self.offset = offset
        self.new_name = new_name
        self._given_resources = resources
        self.resources = None
        self.in_hierarchy = in_hierarchy
        self.unsure = unsure
        self.docs = docs
        self.task_handle = task_handle
        self.require_identifier = require_identifier
        self.old_name = None
        self.old_pyname = None
        self.old_instance = None
        self.changes = None
        self.analysis = None
        self._prepared = False

    def prepare_for_execution(self):
        """Resolve inputs into program-model entities; validate the target.

        This is the late-configuration hook of the architecture: it
        checks that the operation is invoked on a supported node (a
        method) and builds the entities the rest of the execution uses.
        Failure here is a hard rejection.
        """
        if self._prepared:
            return
        self.old_name = worder.get_name_at(self.resource, self.offset)
        this_pymodule = self.project.get_pymodule(self.resource)
        self.old_instance, self.old_pyname = evaluate.eval_location2(
            this_pymodule, self.offset
        )
        if self.old_pyname is None:
            raise exceptions.RefactoringError(
                "Rename refactoring should be performed"
                " on resolvable python identifiers."
            )
        if not self._is_method():
            raise exceptions.RefactoringError(
                "Rename method refactoring should be performed on a method."
            )
        if self._given_resources is None:
            self.resources = self.project.get_python_files()
        else:
            self.resources = self._given_resources
        self.analysis = _OccurrenceAnalysis(self)
        self._prepared = True

    def _is_method(self):
        pyname = self.old_pyname
        return (
            isinstance(pyname, pynames.DefinedName)
            and isinstance(pyname.get_object(), pyobjects.PyFunction)
            and isinstance(pyname.get_object().parent, pyobjects.PyClass)
        )

    def get_pyclass(self):
        return self.old_pyname.get_object().parent

    def applicability_preconditions(self):
        return [
            ValidNameCondition(
                self.new_name, require_identifier=self.require_identifier
            ),
        ]

    def private_transform(self):
        """Construct the `ChangeSet` from the shared occurrence analysis."""
        self.analysis.ensure_ran()
        changes = ChangeSet(f"Renaming <{self.old_name}> to <{self.new_name}>")
        for file_, new_content in self.analysis.new_contents:
            changes.add_change(ChangeContents(file_, new_content))
        self.changes = changes
        return changes


class RenameMethodRefactoring(Refactoring):
    """Behavior-preserving method rename.

    A decorator over `RenameMethodTransformation`: it shares the
    transformation's applicability preconditions and change function
    and adds behavior-preserving preconditions.  Violations of those
    partition the applicability domain instead of restricting it --
    the same change is still constructible through the transformation.
    """

    def __init__(self, *args, **kwds):
        super().__init__(RenameMethodTransformation(*args, **kwds))

    @property
    def old_name(self):
        return self.transformation.old_name

    @property
    def new_name(self):
        return self.transformation.new_name

    def _build_breaking_change_preconditions(self):
        return [
            self.hierarchy_conflict_condition(),
            self.unsure_occurrences_condition(),
            self.analysis_coverage_condition(),
        ]

    def hierarchy_conflict_condition(self):
        transformation = self.transformation
        return HierarchyDoesNotDefineNameCondition(
            transformation.analysis,
            transformation.get_pyclass(),
            transformation.old_pyname,
            transformation.new_name,
        )

    def unsure_occurrences_condition(self):
        return NoUnsureOccurrencesCondition(
            self.transformation.analysis, self.transformation.old_name
        )

    def analysis_coverage_condition(self):
        return AnalysisCoversAllClientsCondition(
            self.transformation.project, self.transformation.resources
        )


# Method rename predates the shared driver; the old name stays usable.
RenameMethodDriver = RefactoringDriver


class _OccurrenceAnalysis(OccurrenceAnalysis):
    """The rename pass, recording unsure and definition occurrences."""

    def __init__(self, transformation):
        super().__init__(transformation)
        self.defining_occurrences = []

    def _build_finder(self):
        transformation = self.transformation
        return _DefinitionRecordingFinder(
            occurrences.create_finder(
                transformation.project,
                transformation.old_name,
                transformation.old_pyname,
                unsure=self.record_unsure,
                docs=transformation.docs,
                instance=transformation.old_instance,
                in_hierarchy=transformation.in_hierarchy,
            ),
            self.defining_occurrences.append,
        )

    def _rewrite(self, finder, resource):
        # imported here: rope.refactor.rename delegates to this module
        from rope.refactor.rename import rename_in_module

        return rename_in_module(
            finder, self.transformation.new_name, resource=resource
        )

    def record_unsure(self, occurrence):
        """Record, then preserve the caller's own `unsure` decision."""
        self.unsure_occurrences.append(occurrence)
        if self.transformation.unsure is None:
            return False
        return self.transformation.unsure(occurrence)


class _DefinitionRecordingFinder:
    """Re-yields a finder's occurrences, recording renamed `def` headers."""

    def __init__(self, finder, record):
        self.finder = finder
        self.record = record

    def find_occurrences(self, resource=None, pymodule=None):
        for occurrence in self.finder.find_occurrences(resource, pymodule):
            if occurrence.is_defined():
                self.record(occurrence)
            yield occurrence
