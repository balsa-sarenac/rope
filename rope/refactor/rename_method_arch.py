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

import ast
from keyword import iskeyword

from rope.base import evaluate, exceptions, pynames, pyobjects, taskhandle, worder
from rope.base.change import ChangeContents, ChangeSet
from rope.refactor import occurrences

APPLICABILITY = "applicability"
BEHAVIOR_PRESERVING = "behavior_preserving"


class BehaviorPreservationWarning(exceptions.RefactoringError):
    """Signals failed behavior-preserving preconditions.

    The reference architecture reports these violations through a
    resumable warning signal; Python has no resumable exceptions, so
    callers that want to proceed anyway use a `RenameMethodDriver`
    policy or the underlying transformation instead of resuming.
    """

    def __init__(self, conditions):
        self.conditions = conditions
        super().__init__("\n".join(c.error_string() for c in conditions))


class Condition:
    """A reified precondition over rope's program model.

    ``check()`` evaluates the condition and populates ``violators``
    with the entities that caused a failure (occurrences, pynames,
    resources), keeping the exact cause inspectable.
    """

    name = "condition"
    level = None

    def __init__(self):
        self.violators = []

    def check(self):
        """Evaluate the condition; return `True` when it holds."""
        self.violators = list(self._find_violators())
        return not self.violators

    def _find_violators(self):
        raise NotImplementedError

    def error_string(self):
        raise NotImplementedError


class ValidNameCondition(Condition):
    """The target name is structurally valid.

    With ``require_identifier=False`` only the keyword check is
    performed, which reproduces `Rename.validate_changes()` exactly
    (legacy compatibility).
    """

    name = "valid-name"
    level = APPLICABILITY

    def __init__(self, new_name, require_identifier=True):
        super().__init__()
        self.new_name = new_name
        self.require_identifier = require_identifier

    def _find_violators(self):
        if self.new_name is None:
            return [self.new_name]
        if iskeyword(self.new_name):
            return [self.new_name]
        if self.require_identifier and not self.new_name.isidentifier():
            return [self.new_name]
        return []

    def error_string(self):
        if self.new_name is None:
            return "Invalid refactoring target name. No name was given."
        if iskeyword(self.new_name):
            return (
                f"Invalid refactoring target name. "
                f"'{self.new_name}' is a Python keyword."
            )
        return (
            f"Invalid refactoring target name. "
            f"'{self.new_name}' is not a valid Python identifier."
        )


class RenameMethodTransformation:
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

    def check_preconditions(self):
        check_applicability_preconditions(self)

    def private_transform(self):
        """Construct the `ChangeSet` from the shared occurrence analysis."""
        self.analysis.ensure_ran()
        changes = ChangeSet(f"Renaming <{self.old_name}> to <{self.new_name}>")
        for file_, new_content in self.analysis.new_contents:
            changes.add_change(ChangeContents(file_, new_content))
        self.changes = changes
        return changes

    def generate_changes(self):
        self.prepare_for_execution()
        self.check_preconditions()
        return self.private_transform()

    def perform_changes(self):
        self.project.do(self.changes)

    def execute(self):
        self.generate_changes()
        self.perform_changes()
        return self.changes


def check_applicability_preconditions(operation):
    """Check and hard-fail: applicability violations stop the operation."""
    failed = [
        condition
        for condition in operation.applicability_preconditions()
        if not condition.check()
    ]
    if failed:
        raise exceptions.RefactoringError(
            "\n".join(condition.error_string() for condition in failed)
        )


class _OccurrenceAnalysis:
    """One shared occurrence-analysis pass over the selected resources.

    Both the behavior-preserving conditions and change construction
    need rope's occurrence search.  Unsure occurrences are reported
    only through a callback invoked *during* the search, so warnings
    cannot be computed cheaper than the full analysis; running the
    search once here keeps the layered API affordable and records the
    unsure occurrences as inspectable objects.
    """

    def __init__(self, transformation):
        self.transformation = transformation
        self.unsure_occurrences = []
        self.new_contents = []
        self._ran = False

    def ensure_ran(self):
        if self._ran:
            return
        # imported here: rope.refactor.rename delegates to this module
        from rope.refactor.rename import rename_in_module

        transformation = self.transformation
        finder = occurrences.create_finder(
            transformation.project,
            transformation.old_name,
            transformation.old_pyname,
            unsure=self._record_unsure,
            docs=transformation.docs,
            instance=transformation.old_instance,
            in_hierarchy=transformation.in_hierarchy,
        )
        job_set = transformation.task_handle.create_jobset(
            "Collecting Changes", len(transformation.resources)
        )
        for file_ in transformation.resources:
            job_set.started_job(file_.path)
            new_content = rename_in_module(
                finder, transformation.new_name, resource=file_
            )
            if new_content is not None:
                self.new_contents.append((file_, new_content))
            job_set.finished_job()
        self._ran = True

    def _record_unsure(self, occurrence):
        self.unsure_occurrences.append(occurrence)
        if self.transformation.unsure is None:
            return False
        return self.transformation.unsure(occurrence)
