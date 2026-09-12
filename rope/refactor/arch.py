"""Shared architecture-facing refactoring machinery (research POC).

This module holds the parts of the transformation/refactoring/driver
architecture that are independent of any concrete refactoring: reified
`Condition` objects, the warning signal, the execution result, and the
headless `RefactoringDriver`.  Concrete operations (method rename,
change signature) build on these while reusing rope's program model,
occurrence engine and change model.

Operations expose two behavioral levels sharing one change function --
a transformation (applicability preconditions plus change
construction) and a refactoring decorating it with behavior-preserving
preconditions -- and a three-layer execution API:

* ``execute()`` = ``generate_changes()`` + ``perform_changes()``
* ``generate_changes()`` = ``prepare_for_execution()`` +
  ``check_preconditions()`` + ``private_transform()``

The shared behavior-preserving conditions are parameterized by the
operation instance, exactly as the reference architecture's conditions
are parameterized by their refactoring.  They rely on a small duck
protocol rather than an ABC; the operation must provide:

* ``analysis`` -- exposing ``ensure_ran()`` and ``unsure_occurrences``
* ``old_name`` -- the searched name
* ``resources`` -- the analyzed files
* ``project`` -- the rope project
* ``docs`` -- whether textual (docs-mode) occurrences are rewritten
"""

from keyword import iskeyword

from rope.base import exceptions

APPLICABILITY = "applicability"
BEHAVIOR_PRESERVING = "behavior_preserving"


class BehaviorPreservationWarning(exceptions.RopeError):
    """Signals failed behavior-preserving preconditions.

    The reference architecture reports these violations through a
    resumable warning signal; Python has no resumable exceptions, so
    callers that want to proceed anyway use a driver policy or the
    underlying transformation instead of resuming.

    It is deliberately *not* a `RefactoringError`: an applicability
    failure stops the operation, a behavior-preserving failure
    delegates a decision, and a script tells the two apart by type --
    the two catchable channels of the reference architecture.  No
    legacy caller is affected, because rope's existing entry points run
    the driver under the LEGACY policy, which never raises this.  Both
    still share `RopeError` as their root.
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

    def subjects(self):
        """Every entity the condition ranges over.

        `violators` is the failing subset; a negated condition needs
        the complement, so a condition that can be negated must say
        what it ranged over.
        """
        raise NotImplementedError

    def not_(self):
        if type(self).subjects is Condition.subjects:
            raise TypeError(
                f"<{self.name}> states no subjects and cannot be negated"
            )
        return NegatedCondition(self)


class NegatedCondition(Condition):
    """Holds exactly when the condition it wraps fails.

    Mirrors `RBNegatedCondition`: the violators of a negated condition
    are the *non*-violators of the inner one, so a failure still names
    the entities responsible rather than only reporting a boolean.
    """

    def __init__(self, condition):
        super().__init__()
        self.condition = condition
        self.name = "not-" + condition.name
        self.level = condition.level

    def check(self):
        self.condition.check()
        self.violators = list(self.non_violators())
        return not self.violators

    def non_violators(self):
        violators = self.condition.violators
        return [
            subject for subject in self.condition.subjects()
            if subject not in violators
        ]

    def subjects(self):
        return self.condition.subjects()

    def error_string(self):
        return f"Expected <{self.condition.name}> to fail, but it held."


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

    def subjects(self):
        return [self.new_name]

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


class NoUnsureOccurrencesCondition(Condition):
    """No occurrence of the searched name has an unresolvable receiver.

    Violators are the unsure `Occurrence` objects recorded during the
    shared occurrence analysis; checking this condition triggers that
    analysis, so warnings are never cheaper than the search itself.

    It states no `subjects`: rope reports unresolvable receivers
    through a callback invoked only on failure, so the occurrences it
    *could* resolve are never recorded.  Enumerating them would mean
    retaining every occurrence of the name, which the analysis has no
    other use for.
    """

    name = "no-unsure-occurrences"
    level = BEHAVIOR_PRESERVING

    def __init__(self, analysis, name):
        super().__init__()
        self.analysis = analysis
        self.searched_name = name

    def _find_violators(self):
        analysis = self.analysis
        if hasattr(analysis, "ensure_ran"):
            analysis.ensure_ran()
            return analysis.unsure_occurrences
        return analysis.unsure_occurrences()

    def error_string(self):
        places = ", ".join(
            f"{occurrence.resource.path}:{occurrence.lineno}"
            for occurrence in self.violators
        )
        return (
            f"{len(self.violators)} occurrence(s) of"
            f" '{self.searched_name}' could not be resolved"
            f" statically: {places}"
        )


class AnalysisCoversAllClientsCondition(Condition):
    """The analysis covers every python file of the project.

    When `resources` restricts the analysis, clients outside the
    selection keep calling the old name; the excluded files are the
    violators.
    """

    name = "analysis-covers-all-clients"
    level = BEHAVIOR_PRESERVING

    def __init__(self, project, resources):
        super().__init__()
        self.project = project
        self.resources = resources

    def subjects(self):
        return self.project.get_python_files()

    def _find_violators(self):
        analyzed = set(self.resources)
        return [file_ for file_ in self.subjects() if file_ not in analyzed]

    def error_string(self):
        places = ", ".join(file_.path for file_ in self.violators)
        return (
            f"The analysis is restricted; possible clients are not"
            f" updated: {places}"
        )


class Transformation:
    """Boilerplate for a behavior-agnostic transformation.

    The polymorphic core of the three-layer API: subclasses supply
    `prepare_for_execution()`, `applicability_preconditions()` and
    `private_transform()`, and must expose `project` and `changes`;
    the layer composition and execution control are shared.
    """

    checked_applicability = ()

    def prepare_for_execution(self):
        raise NotImplementedError

    def applicability_preconditions(self):
        raise NotImplementedError

    def private_transform(self):
        raise NotImplementedError

    def check_preconditions(self):
        self.checked_applicability = check_applicability_preconditions(self)

    def internally_checked_preconditions(self):
        """Conditions this operation checked itself, if any.

        A composite checks each child's applicability during
        execution, at the child's own point in the sequence, so those
        conditions never pass through the driver's own gate; reporting
        them here keeps the driver's result complete.
        """
        return []

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


class Refactoring(Transformation):
    """A transformation plus a behavior-preserving commitment.

    Mirrors `ReRefactoring`, which both *inherits* the three-layer API
    from `ReAbstractTransformation` and *holds* a transformation to
    delegate to.  Inheriting means only the commitment is written here:
    `generate_changes`, `perform_changes` and `execute` are the shared
    ones, and they reach the inner transformation through the
    delegated hooks below.

    Subclasses supply `_build_breaking_change_preconditions()`.
    """

    def __init__(self, transformation):
        self.transformation = transformation
        self._breaking_change_preconditions = None

    @property
    def project(self):
        return self.transformation.project

    @property
    def changes(self):
        return self.transformation.changes

    def prepare_for_execution(self):
        self.transformation.prepare_for_execution()

    def applicability_preconditions(self):
        return self.transformation.applicability_preconditions()

    def private_transform(self):
        return self.transformation.private_transform()

    def internally_checked_preconditions(self):
        return self.transformation.internally_checked_preconditions()

    def breaking_change_preconditions(self):
        if self._breaking_change_preconditions is None:
            self._breaking_change_preconditions = (
                self._build_breaking_change_preconditions()
            )
        return self._breaking_change_preconditions

    def _build_breaking_change_preconditions(self):
        raise NotImplementedError

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


class PendingChanges:
    """The program as it would be after the changes absorbed so far.

    A composite's children each construct changes; a later child must
    analyze the program *including* its predecessors' edits, without
    anything being written.  This is the view that makes that
    possible -- the role `RBNamespace` plays in the reference
    architecture.

    Rope keeps parsed modules in `pycore.module_cache.module_map`,
    keyed by resource, and every name resolution goes through it.  A
    pending module is therefore installed into that cache rather than
    consulted beside it: a module built from pending source would
    otherwise be a different object than the one other modules resolve
    through, and cross-module occurrences would stop matching.

    `restore()` drops the pending modules again, and must run whether
    or not the children succeed.

    This reaches into `pycore.module_cache` and
    `_invalidate_resource_cache`, which rope does not publish.  The
    view is therefore sound but coupled to the engine's internals: the
    parts needed to build it exist, which is the portability claim,
    but they are not a supported interface and an upstream change to
    the caching strategy would break it.
    """

    def __init__(self, project):
        self.project = project
        self.sources = {}

    def absorb(self, changes):
        """Take the changes into the view without writing anything."""
        for change in changes.changes:
            self.sources[change.resource] = change.new_contents
            self._install(change.resource)

    def _install(self, resource):
        from rope.base import libutils

        pycore = self.project.pycore
        pycore._invalidate_resource_cache(resource)
        module = libutils.get_string_module(
            self.project, self.sources[resource], resource
        )
        pycore.module_cache.module_map[resource] = module

    def restore(self):
        """Forget the pending modules; the project is untouched."""
        pycore = self.project.pycore
        for resource in self.sources:
            pycore._invalidate_resource_cache(resource)

    def source(self, resource):
        if resource in self.sources:
            return self.sources[resource]
        return resource.read()

    def pymodule(self, resource):
        return self.project.get_pymodule(resource)

    def as_changes(self, description):
        from rope.base.change import ChangeContents, ChangeSet

        changes = ChangeSet(description)
        for resource, source in self.sources.items():
            changes.add_change(ChangeContents(resource, source))
        return changes


class OccurrenceAnalysis:
    """One shared occurrence-and-rewrite pass over the selected resources.

    Both operations need the same pass for the same two reasons: the
    `ChangeSet` is built from it, and the behavior-preserving
    conditions inspect what it saw.  Rope reports unresolvable
    receivers through an `unsure` callback invoked *during* the
    search, so a warning can never be cheaper than the analysis
    itself; running it once here is what keeps the layered API
    affordable.

    Subclasses supply `_build_finder()` and `_rewrite()`.
    """

    def __init__(self, transformation):
        self.transformation = transformation
        self.unsure_occurrences = []
        self.new_contents = []
        self._ran = False

    def ensure_ran(self):
        if self._ran:
            return
        transformation = self.transformation
        finder = self._build_finder()
        job_set = transformation.task_handle.create_jobset(
            "Collecting Changes", len(transformation.resources)
        )
        for file_ in transformation.resources:
            job_set.started_job(file_.path)
            new_content = self._rewrite(finder, file_)
            if new_content is not None:
                self.new_contents.append((file_, new_content))
            job_set.finished_job()
        self._ran = True

    def _build_finder(self):
        raise NotImplementedError

    def _rewrite(self, finder, resource):
        raise NotImplementedError

    def record_unsure(self, occurrence):
        """The `unsure` callback: record, and answer rope's question."""
        self.unsure_occurrences.append(occurrence)
        return False


def check_applicability_preconditions(operation):
    """Check and hard-fail: applicability violations stop the operation.

    Returns the conditions checked, so a caller that needs to report
    them keeps the instances whose `violators` were populated.
    """
    conditions = operation.applicability_preconditions()
    failed = [condition for condition in conditions if not condition.check()]
    if failed:
        raise exceptions.RefactoringError(
            "\n".join(condition.error_string() for condition in failed)
        )
    return conditions


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


class RefactoringDriver:
    """Headless orchestrator for an architecture-facing refactoring.

    The interactive driver of the reference architecture resolves
    warnings with the user; this headless equivalent resolves them
    with an explicit policy.  Like its interactive counterpart it uses
    the fine-grained layer of the API (individual conditions and
    `private_transform`) rather than `generate_changes`.

    * `legacy`: reproduce the legacy ``get_changes()`` semantics --
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
        applicability = check_applicability_preconditions(refactoring)
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
            changes,
            applicability + refactoring.internally_checked_preconditions(),
            warnings,
            self.policy,
        )
