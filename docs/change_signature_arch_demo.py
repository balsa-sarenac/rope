"""Demonstration scenario for the change-signature composite POC.

Builds a small synthetic project with a class hierarchy and call
sites, then runs one composite signature change (add a parameter,
reorder, remove one):

1. showing the signature flow between the composite's changers,
2. at the transformation level (behavior-agnostic),
3. at the refactoring level (warns, non-resumably),
4. under the three driver policies, and
5. apply--test--undo through rope's change model and history.

Run from the repository root:

    python docs/change_signature_arch_demo.py
"""

import tempfile
from textwrap import dedent, indent

from rope.base.project import Project
from rope.refactor import arch
from rope.refactor import change_signature_arch as arch_cs
from rope.refactor.change_signature import (
    ArgumentAdder,
    ArgumentRemover,
    ArgumentReorderer,
)

REPORTS = dedent('''\
    class Report:
        def render(self, title, footer):
            return f"{title} / {footer}"


    class FancyReport(Report):
        def render(self, title, footer):
            return f"** {title} ** / {footer}"
''')

CLIENTS = dedent('''\
    from reports import Report

    report = Report()
    print(report.render("summary", "page 1"))
''')


def changers():
    # Removing index 3 is invalid against the original three-slot
    # signature; it only becomes applicable after the add changer --
    # the late-configuration point the composite exists for.
    # the level is a class choice per child: a reorder cannot break
    # behavior, an add and a remove can
    return [
        arch_cs.AddParameterRefactoring(ArgumentAdder(3, "header")),
        ArgumentReorderer([0, 1, 3, 2]),
        arch_cs.RemoveParameterRefactoring(ArgumentRemover(3)),
    ]


def show(title, result):
    print(f"--- {title} ---")
    if result.warning_results:
        for condition in result.warning_results:
            print(f"warning [{condition.name}]:")
            print(indent(condition.error_string(), "  "))
    else:
        print("no warnings")
    if result.changes is None:
        print("changes: rejected by policy")
    else:
        print("changes:")
        print(indent(result.changes.get_description().rstrip(), "  "))
    print()


def main():
    root = tempfile.mkdtemp(prefix="rope-arch-demo-")
    project = Project(root)
    reports = project.root.create_file("reports.py")
    reports.write(REPORTS)
    clients = project.root.create_file("clients.py")
    clients.write(CLIENTS)
    offset = REPORTS.index("render")

    def refactoring():
        return arch_cs.ChangeSignatureRefactoring(project, reports, offset, changers())

    print("=== The composite's children, at the level each warrants ===\n")
    transformation = arch_cs.ChangeSignatureTransformation(
        project, reports, offset, changers()
    )
    transformation.prepare_for_execution()
    transformation.check_preconditions()
    for child, inner in zip(
        transformation.children, transformation.child_transformations()
    ):
        print(
            f"{type(inner).__name__}"
            f" [{type(child).__name__}]:"
            f" {inner.definition_info.to_string()}"
        )
    print()

    print("=== Two behavioral levels, one change function ===\n")
    changes = transformation.generate_changes()
    print("transformation level: changes constructed, no warnings consulted")
    print(indent(changes.get_description().rstrip(), "  ") + "\n")

    try:
        refactoring().generate_changes()
    except arch.BehaviorPreservationWarning as warning:
        print("refactoring level: BehaviorPreservationWarning")
        print(indent(str(warning), "  ") + "\n")

    print("=== Driver policies ===\n")
    show("legacy", arch.RefactoringDriver(refactoring(), arch.LEGACY).run())
    show(
        "fail_on_warning",
        arch.RefactoringDriver(refactoring(), arch.FAIL_ON_WARNING).run(),
    )
    result = arch.RefactoringDriver(
        refactoring(), arch.PROCEED_AFTER_WARNING
    ).run()
    show("proceed_after_warning", result)

    print("=== Apply, test, undo (execution control) ===\n")
    project.do(result.changes)
    compiled = compile(clients.read(), "clients.py", "exec")
    print("applied; rewritten clients module still compiles:", bool(compiled))
    print(
        "resolved call site rewritten:",
        'report.render("summary", "page 1")' not in clients.read(),
    )
    print(
        "warned hierarchy override kept the old signature:",
        "def render(self, title, footer)" in reports.read(),
    )
    project.history.undo()
    print("undone; original restored:", clients.read() == CLIENTS)
    project.close()


if __name__ == "__main__":
    main()
