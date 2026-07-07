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

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation
        self.new_name = transformation.new_name

    def _find_violators(self):
        analysis = self.transformation.analysis
        analysis.ensure_ran()
        classes = [self.transformation.get_pyclass()]
        for occurrence in analysis.defining_occurrences:
            pyclass = _containing_class(occurrence.get_pyname())
            if pyclass is not None:
                classes.append(pyclass)
        violators = []
        seen = set()
        for pyclass in classes:
            attributes = pyclass.get_attributes()
            if self.new_name not in attributes:
                continue
            conflict = ConflictingDefinition(attributes[self.new_name], pyclass)
            key = (pyclass.get_name(), conflict.lineno)
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


class NoUnsureOccurrencesCondition(Condition):
    """No occurrence of the old name has an unresolvable receiver.

    Violators are the unsure `Occurrence` objects recorded during the
    shared occurrence analysis; checking this condition triggers that
    analysis, so warnings are never cheaper than the search itself.
    """

    name = "no-unsure-occurrences"
    level = BEHAVIOR_PRESERVING

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def _find_violators(self):
        analysis = self.transformation.analysis
        analysis.ensure_ran()
        return analysis.unsure_occurrences

    def error_string(self):
        places = ", ".join(
            f"{occurrence.resource.path}:{occurrence.lineno}"
            for occurrence in self.violators
        )
        return (
            f"{len(self.violators)} occurrence(s) of"
            f" '{self.transformation.old_name}' could not be resolved"
            f" statically: {places}"
        )


class ReflectiveReference:
    """A violator: a reflective or textual reference to the old name."""

    def __init__(self, resource, lineno, kind):
        self.resource = resource
        self.lineno = lineno
        self.kind = kind


class NoReflectiveReferencesCondition(Condition):
    """No reflective or textual reference to the old name is visible.

    A conservative AST scan for ``getattr``/``setattr``/``hasattr``/
    ``delattr`` calls, ``methodcaller`` and string constants equal to
    the old name.  With ``docs=True`` rope renames textual occurrences
    itself, but only where the name appears contiguously in the
    source; references the textual finder cannot see (folded implicit
    concatenations, escapes) are still reported.  This is a partial
    approximation of Python's reflection facilities.
    """

    name = "no-reflective-references"
    level = BEHAVIOR_PRESERVING

    _reflective_builtins = {"getattr", "setattr", "hasattr", "delattr"}

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def _find_violators(self):
        violators = []
        for resource in self.transformation.resources:
            source = resource.read()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            violators.extend(self._scan_module(resource, tree, source))
        return violators

    def _scan_module(self, resource, tree, source):
        old_name = self.transformation.old_name
        reflective_arguments = set()
        violators = []
        for node in ast.walk(tree):
            argument = self._match_call(node, old_name)
            if argument is not None:
                reflective_arguments.add(id(argument))
                if self._is_covered_by_docs_rename(argument, source):
                    continue
                violators.append(
                    ReflectiveReference(resource, node.lineno, self._kind(node))
                )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value == old_name
                and id(node) not in reflective_arguments
                and not self._is_covered_by_docs_rename(node, source)
            ):
                violators.append(ReflectiveReference(resource, node.lineno, "string"))
        return violators

    def _is_covered_by_docs_rename(self, node, source):
        """Whether rope's docs-mode textual rename rewrites this constant.

        The textual finder only sees the name written contiguously, so
        a folded implicit concatenation stays a violator even with
        ``docs=True``.
        """
        if not self.transformation.docs:
            return False
        segment = ast.get_source_segment(source, node)
        return segment is not None and self.transformation.old_name in segment

    def _match_call(self, node, old_name):
        """Return the string argument node when `node` reflects `old_name`."""
        if not isinstance(node, ast.Call):
            return None
        kind = self._kind(node)
        if kind in self._reflective_builtins:
            arg_index = 1
        elif kind == "methodcaller":
            arg_index = 0
        else:
            return None
        if len(node.args) <= arg_index:
            return None
        argument = node.args[arg_index]
        if isinstance(argument, ast.Constant) and argument.value == old_name:
            return argument
        return None

    def _kind(self, node):
        function = node.func
        if isinstance(function, ast.Name):
            return function.id
        if isinstance(function, ast.Attribute):
            return function.attr
        return None

    def error_string(self):
        places = ", ".join(
            f"{reference.resource.path}:{reference.lineno} ({reference.kind})"
            for reference in self.violators
        )
        return (
            f"'{self.transformation.old_name}' is referenced reflectively"
            f" or textually and will not be renamed: {places}"
        )


class AnalysisCoversAllClientsCondition(Condition):
    """The analysis covers every python file of the project.

    When `resources` restricts the analysis, clients outside the
    selection keep calling the old name; the excluded files are the
    violators.
    """

    name = "analysis-covers-all-clients"
    level = BEHAVIOR_PRESERVING

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def _find_violators(self):
        analyzed = set(self.transformation.resources)
        return [
            file_
            for file_ in self.transformation.project.get_python_files()
            if file_ not in analyzed
        ]

    def error_string(self):
        places = ", ".join(file_.path for file_ in self.violators)
        return (
            f"The analysis is restricted; possible clients are not"
            f" updated: {places}"
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


class RenameMethodRefactoring:
    """Behavior-preserving method rename.

    A decorator over `RenameMethodTransformation`: it shares the
    transformation's applicability preconditions and change function
    and adds behavior-preserving preconditions.  Violations of those
    partition the applicability domain instead of restricting it --
    the same change is still constructible through the transformation.
    """

    def __init__(self, *args, **kwds):
        self.transformation = RenameMethodTransformation(*args, **kwds)
        self._breaking_change_preconditions = None

    @property
    def project(self):
        return self.transformation.project

    @property
    def old_name(self):
        return self.transformation.old_name

    @property
    def new_name(self):
        return self.transformation.new_name

    @property
    def changes(self):
        return self.transformation.changes

    def prepare_for_execution(self):
        self.transformation.prepare_for_execution()

    def applicability_preconditions(self):
        return self.transformation.applicability_preconditions()

    def breaking_change_preconditions(self):
        if self._breaking_change_preconditions is None:
            self._breaking_change_preconditions = [
                self.hierarchy_conflict_condition(),
                self.unsure_occurrences_condition(),
                self.reflective_references_condition(),
                self.analysis_coverage_condition(),
            ]
        return self._breaking_change_preconditions

    def hierarchy_conflict_condition(self):
        return HierarchyDoesNotDefineNameCondition(self.transformation)

    def unsure_occurrences_condition(self):
        return NoUnsureOccurrencesCondition(self.transformation)

    def reflective_references_condition(self):
        return NoReflectiveReferencesCondition(self.transformation)

    def analysis_coverage_condition(self):
        return AnalysisCoversAllClientsCondition(self.transformation)

    def check_preconditions(self):
        self.transformation.check_preconditions()
        self.check_breaking_change_preconditions()

    def check_breaking_change_preconditions(self):
        failed = [
            condition
            for condition in self.breaking_change_preconditions()
            if not condition.check()
        ]
        if failed:
            raise BehaviorPreservationWarning(failed)

    def private_transform(self):
        return self.transformation.private_transform()

    def generate_changes(self):
        self.prepare_for_execution()
        self.check_preconditions()
        return self.private_transform()

    def perform_changes(self):
        self.transformation.perform_changes()

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


LEGACY = "legacy"
FAIL_ON_WARNING = "fail_on_warning"
PROCEED_AFTER_WARNING = "proceed_after_warning"

POLICIES = (LEGACY, FAIL_ON_WARNING, PROCEED_AFTER_WARNING)


class RefactoringExecutionResult:
    """What a driver run produced.

    * `changes`: the `ChangeSet`, or `None` when the policy rejected it.
    * `applicability_results`: the checked applicability conditions.
    * `warning_results`: the failed behavior-preserving conditions,
      with their violators.
    * `mode`: the policy that ran.
    """

    def __init__(self, changes, applicability_results, warning_results, mode):
        self.changes = changes
        self.applicability_results = applicability_results
        self.warning_results = warning_results
        self.mode = mode


class RenameMethodDriver:
    """Headless orchestrator for method rename.

    The interactive driver of the reference architecture resolves
    warnings with the user; this headless equivalent resolves them
    with an explicit policy.  Like its interactive counterpart it uses
    the fine-grained layer of the API (individual conditions and
    `private_transform`) rather than `generate_changes`.

    * `legacy`: reproduce `Rename.get_changes()` semantics --
      behavior-preserving conditions are not consulted.
    * `fail_on_warning`: return warnings and no changes.
    * `proceed_after_warning`: return warnings and the changes.
    """

    def __init__(self, refactoring, policy=LEGACY):
        if policy not in POLICIES:
            raise ValueError(f"Unknown warning policy: {policy!r}")
        self.refactoring = refactoring
        self.policy = policy

    def run(self):
        refactoring = self.refactoring
        refactoring.prepare_for_execution()
        applicability = refactoring.applicability_preconditions()
        failed = [condition for condition in applicability if not condition.check()]
        if failed:
            raise exceptions.RefactoringError(
                "\n".join(condition.error_string() for condition in failed)
            )
        warnings = []
        if self.policy != LEGACY:
            warnings = [
                condition
                for condition in refactoring.breaking_change_preconditions()
                if not condition.check()
            ]
            if warnings and self.policy == FAIL_ON_WARNING:
                return RefactoringExecutionResult(
                    None, applicability, warnings, self.policy
                )
        changes = refactoring.private_transform()
        return RefactoringExecutionResult(
            changes, applicability, warnings, self.policy
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
        self.defining_occurrences = []
        self.new_contents = []
        self._ran = False

    def ensure_ran(self):
        if self._ran:
            return
        # imported here: rope.refactor.rename delegates to this module
        from rope.refactor.rename import rename_in_module

        transformation = self.transformation
        finder = _DefinitionRecordingFinder(
            occurrences.create_finder(
                transformation.project,
                transformation.old_name,
                transformation.old_pyname,
                unsure=self._record_unsure,
                docs=transformation.docs,
                instance=transformation.old_instance,
                in_hierarchy=transformation.in_hierarchy,
            ),
            self.defining_occurrences.append,
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
