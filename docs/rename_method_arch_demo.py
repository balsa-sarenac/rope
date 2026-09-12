"""Demonstration scenario for the method-rename architecture POC.

Builds a small synthetic project with inheritance, duck typing and
duck-typed receivers, then runs the same method rename:

1. at the transformation level (behavior-agnostic),
2. at the refactoring level (warns, non-resumably),
3. under the three driver policies, and
4. apply--test--undo through rope's change model and history.

Run from the repository root:

    python docs/rename_method_arch_demo.py
"""

import tempfile
from textwrap import dedent, indent

from rope.base.project import Project
from rope.refactor import rename_method_arch as arch

SHAPES = dedent('''\
    class Shape:
        def area(self):
            raise NotImplementedError

        def describe(self):
            return f"area={self.area()}"


    class Square(Shape):
        def __init__(self, side):
            self.side = side

        def area(self):
            return self.side * self.side
''')

CLIENTS = dedent('''\
    from shapes import Square


    def total(shapes):
        # duck typing: rope cannot resolve the receiver statically
        return sum(shape.area() for shape in shapes)


    square = Square(3)
    assert square.area() == 9
''')


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
    shapes = project.root.create_file("shapes.py")
    shapes.write(SHAPES)
    clients = project.root.create_file("clients.py")
    clients.write(CLIENTS)
    offset = SHAPES.index("area")

    def refactoring():
        return arch.RenameMethodRefactoring(
            project, shapes, offset, "surface", in_hierarchy=True
        )

    print("=== Two behavioral levels, one change function ===\n")
    transformation = arch.RenameMethodTransformation(
        project, shapes, offset, "surface", in_hierarchy=True
    )
    changes = transformation.generate_changes()
    print("transformation level: changes constructed, no warnings consulted")
    print(indent(changes.get_description().rstrip(), "  ") + "\n")

    try:
        refactoring().generate_changes()
    except arch.BehaviorPreservationWarning as warning:
        print("refactoring level: BehaviorPreservationWarning")
        print(indent(str(warning), "  ") + "\n")

    print("=== Driver policies ===\n")
    show("legacy", arch.RenameMethodDriver(refactoring(), arch.LEGACY).run())
    show(
        "fail_on_warning",
        arch.RenameMethodDriver(refactoring(), arch.FAIL_ON_WARNING).run(),
    )
    result = arch.RenameMethodDriver(
        refactoring(), arch.PROCEED_AFTER_WARNING
    ).run()
    show("proceed_after_warning", result)

    print("=== Apply, test, undo (execution control) ===\n")
    project.do(result.changes)
    compiled = compile(clients.read(), "clients.py", "exec")
    print("applied; renamed clients module still compiles:", bool(compiled))
    print("resolved call site renamed:", "square.surface()" in clients.read())
    print(
        "warned duck-typed call site kept the old name:",
        "shape.area()" in clients.read(),
    )
    project.history.undo()
    print("undone; original restored:", clients.read() == CLIENTS)
    project.close()


if __name__ == "__main__":
    main()
