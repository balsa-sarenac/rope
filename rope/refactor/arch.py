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

import ast
from keyword import iskeyword

from rope.base import exceptions

APPLICABILITY = "applicability"
BEHAVIOR_PRESERVING = "behavior_preserving"


class BehaviorPreservationWarning(exceptions.RefactoringError):
    """Signals failed behavior-preserving preconditions.

    The reference architecture reports these violations through a
    resumable warning signal; Python has no resumable exceptions, so
    callers that want to proceed anyway use a driver policy or the
    underlying transformation instead of resuming.
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


class NoUnsureOccurrencesCondition(Condition):
    """No occurrence of the searched name has an unresolvable receiver.

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
    """No reflective or textual reference to the searched name is visible.

    A conservative AST scan for ``getattr``/``setattr``/``hasattr``/
    ``delattr`` calls, ``methodcaller`` and string constants equal to
    the searched name.  With ``docs=True`` rope renames textual
    occurrences itself, but only where the name appears contiguously in
    the source; references the textual finder cannot see (folded
    implicit concatenations, escapes) are still reported.  This is a
    partial approximation of Python's reflection facilities.
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


class Transformation:
    """Boilerplate for a behavior-agnostic transformation.

    The polymorphic core of the three-layer API: subclasses supply
    `prepare_for_execution()`, `applicability_preconditions()` and
    `private_transform()`, and must expose `project` and `changes`;
    the layer composition and execution control are shared.
    """

    def prepare_for_execution(self):
        raise NotImplementedError

    def applicability_preconditions(self):
        raise NotImplementedError

    def private_transform(self):
        raise NotImplementedError

    def check_preconditions(self):
        check_applicability_preconditions(self)

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


class TransformationDecorator:
    """Boilerplate for a refactoring decorating a transformation.

    Subclasses supply the behavior-preserving commitment through
    `_build_breaking_change_preconditions()`; everything else --
    the three-layer API and the delegation to the inner
    transformation's applicability and change function -- is shared.
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
