import unittest
from textwrap import dedent

from rope.base import exceptions
from rope.refactor import rename_method_arch as arch
from ropetest import testutils


class RenameMethodArchMixin:
    def setUp(self):
        super().setUp()
        self.project = testutils.sample_project()

    def tearDown(self):
        testutils.remove_project(self.project)
        super().tearDown()

    def _write_module(self, name, code):
        module = testutils.create_module(self.project, name)
        module.write(code)
        return module


SIMPLE_CLASS = dedent("""\
    class A(object):
        def a_method(self):
            pass
    a = A()
    a.a_method()
""")

SIMPLE_CLASS_RENAMED = dedent("""\
    class A(object):
        def new_method(self):
            pass
    a = A()
    a.new_method()
""")


class ConditionTest(RenameMethodArchMixin, unittest.TestCase):
    def test_valid_name_condition_passes(self):
        condition = arch.ValidNameCondition("new_method")
        self.assertTrue(condition.check())
        self.assertEqual([], condition.violators)

    def test_valid_name_condition_rejects_keywords(self):
        condition = arch.ValidNameCondition("lambda")
        self.assertFalse(condition.check())
        self.assertEqual(["lambda"], condition.violators)
        self.assertIn("keyword", condition.error_string())

    def test_valid_name_condition_rejects_non_identifiers(self):
        condition = arch.ValidNameCondition("foo bar")
        self.assertFalse(condition.check())
        self.assertIn("identifier", condition.error_string())

    def test_valid_name_condition_legacy_mode_allows_non_identifiers(self):
        condition = arch.ValidNameCondition("foo bar", require_identifier=False)
        self.assertTrue(condition.check())

    def test_valid_name_condition_rejects_missing_name(self):
        condition = arch.ValidNameCondition(None)
        self.assertFalse(condition.check())

    def test_condition_levels(self):
        self.assertEqual(
            arch.APPLICABILITY, arch.ValidNameCondition("x").level
        )


class RenameMethodTransformationTest(RenameMethodArchMixin, unittest.TestCase):
    def _transformation(self, module, code, new_name="new_method", **kwds):
        return arch.RenameMethodTransformation(
            self.project, module, code.index("a_method"), new_name, **kwds
        )

    def test_prepare_for_execution_resolves_the_method(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS)
        transformation.prepare_for_execution()
        self.assertEqual("a_method", transformation.old_name)
        self.assertIsNotNone(transformation.old_pyname)

    def test_prepare_for_execution_rejects_non_methods(self):
        code = "a_method = 1\n"
        mod = self._write_module("mod1", code)
        transformation = arch.RenameMethodTransformation(
            self.project, mod, code.index("a_method"), "new_method"
        )
        with self.assertRaises(exceptions.RefactoringError):
            transformation.prepare_for_execution()

    def test_prepare_for_execution_rejects_functions(self):
        code = "def a_method():\n    pass\n"
        mod = self._write_module("mod1", code)
        transformation = arch.RenameMethodTransformation(
            self.project, mod, code.index("a_method"), "new_method"
        )
        with self.assertRaises(exceptions.RefactoringError):
            transformation.prepare_for_execution()

    def test_check_preconditions_rejects_keyword_names(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS, new_name="lambda")
        transformation.prepare_for_execution()
        with self.assertRaises(exceptions.RefactoringError):
            transformation.check_preconditions()

    def test_generate_changes_builds_an_ordinary_changeset(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS)
        changes = transformation.generate_changes()
        self.assertEqual(
            "Renaming <a_method> to <new_method>", changes.description
        )
        self.project.do(changes)
        self.assertEqual(SIMPLE_CLASS_RENAMED, mod.read())

    def test_execute_generates_and_performs(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS)
        transformation.execute()
        self.assertEqual(SIMPLE_CLASS_RENAMED, mod.read())

    def test_execute_supports_undo_through_history(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS)
        transformation.execute()
        self.project.history.undo()
        self.assertEqual(SIMPLE_CLASS, mod.read())

    def test_rename_with_call_sites_across_modules(self):
        mod1 = self._write_module(
            "mod1",
            dedent("""\
                class A(object):
                    def a_method(self):
                        pass
            """),
        )
        mod2 = self._write_module(
            "mod2",
            dedent("""\
                import mod1
                a = mod1.A()
                a.a_method()
            """),
        )
        code = mod1.read()
        transformation = arch.RenameMethodTransformation(
            self.project, mod1, code.index("a_method"), "new_method"
        )
        self.project.do(transformation.generate_changes())
        self.assertIn("def new_method", mod1.read())
        self.assertIn("a.new_method()", mod2.read())

    def test_rename_in_hierarchy(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            class B(A):
                def a_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        transformation = arch.RenameMethodTransformation(
            self.project,
            mod,
            code.index("a_method"),
            "new_method",
            in_hierarchy=True,
        )
        self.project.do(transformation.generate_changes())
        self.assertEqual(2, mod.read().count("def new_method"))

    def test_transformation_has_no_behavior_preserving_conditions(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        transformation = self._transformation(mod, SIMPLE_CLASS)
        transformation.prepare_for_execution()
        levels = {
            condition.level
            for condition in transformation.applicability_preconditions()
        }
        self.assertEqual({arch.APPLICABILITY}, levels)


if __name__ == "__main__":
    unittest.main()
