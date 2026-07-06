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


DUCK_TYPED = dedent("""\
    class A(object):
        def a_method(self):
            pass
    def f(arg):
        arg.a_method()
""")


class BehaviorPreservingConditionTest(RenameMethodArchMixin, unittest.TestCase):
    def _refactoring(self, module, code, new_name="new_method", **kwds):
        return arch.RenameMethodRefactoring(
            self.project, module, code.index("a_method"), new_name, **kwds
        )

    def test_hierarchy_condition_passes_for_fresh_names(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS)
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertTrue(condition.check())

    def test_hierarchy_condition_finds_conflict_in_same_class(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
                def new_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code)
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertFalse(condition.check())
        self.assertEqual(1, len(condition.violators))
        self.assertIsNotNone(condition.violators[0].pyname)
        self.assertIn("new_method", condition.error_string())

    def test_hierarchy_condition_finds_conflict_in_superclass(self):
        code = dedent("""\
            class Base(object):
                def new_method(self):
                    pass
            class A(Base):
                def a_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code)
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertFalse(condition.check())

    def test_unsure_condition_reports_duck_typed_occurrences(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        refactoring = self._refactoring(mod, DUCK_TYPED)
        refactoring.prepare_for_execution()
        condition = refactoring.unsure_occurrences_condition()
        self.assertFalse(condition.check())
        occurrence = condition.violators[0]
        self.assertEqual(mod, occurrence.resource)
        self.assertEqual(5, occurrence.lineno)

    def test_unsure_condition_passes_for_known_receivers(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS)
        refactoring.prepare_for_execution()
        self.assertTrue(refactoring.unsure_occurrences_condition().check())

    def test_reflective_condition_finds_getattr(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            getattr(A(), "a_method")()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code)
        refactoring.prepare_for_execution()
        condition = refactoring.reflective_references_condition()
        self.assertFalse(condition.check())
        reference = condition.violators[0]
        self.assertEqual("getattr", reference.kind)
        self.assertEqual(4, reference.lineno)

    def test_reflective_condition_finds_methodcaller_and_strings(self):
        code = dedent("""\
            from operator import methodcaller
            class A(object):
                def a_method(self):
                    pass
            call = methodcaller("a_method")
            name = "a_method"
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code)
        refactoring.prepare_for_execution()
        condition = refactoring.reflective_references_condition()
        self.assertFalse(condition.check())
        kinds = {reference.kind for reference in condition.violators}
        self.assertEqual({"methodcaller", "string"}, kinds)

    def test_reflective_condition_trivially_passes_with_docs(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            getattr(A(), "a_method")()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code, docs=True)
        refactoring.prepare_for_execution()
        self.assertTrue(refactoring.reflective_references_condition().check())

    def test_coverage_condition_reports_excluded_files(self):
        mod1 = self._write_module("mod1", SIMPLE_CLASS)
        mod2 = self._write_module("mod2", "import mod1\n")
        refactoring = self._refactoring(mod1, SIMPLE_CLASS, resources=[mod1])
        refactoring.prepare_for_execution()
        condition = refactoring.analysis_coverage_condition()
        self.assertFalse(condition.check())
        self.assertEqual([mod2], condition.violators)

    def test_coverage_condition_passes_without_restriction(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS)
        refactoring.prepare_for_execution()
        self.assertTrue(refactoring.analysis_coverage_condition().check())


class RenameMethodRefactoringTest(RenameMethodArchMixin, unittest.TestCase):
    def _refactoring(self, module, code, new_name="new_method", **kwds):
        return arch.RenameMethodRefactoring(
            self.project, module, code.index("a_method"), new_name, **kwds
        )

    def test_refactoring_delegates_applicability_to_transformation(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS)
        refactoring.prepare_for_execution()
        self.assertEqual(
            [condition.name for condition in
             refactoring.transformation.applicability_preconditions()],
            [condition.name for condition in
             refactoring.applicability_preconditions()],
        )

    def test_breaking_change_preconditions_are_behavior_preserving(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS)
        refactoring.prepare_for_execution()
        conditions = refactoring.breaking_change_preconditions()
        self.assertTrue(conditions)
        self.assertEqual(
            {arch.BEHAVIOR_PRESERVING},
            {condition.level for condition in conditions},
        )

    def test_two_levels_share_the_change_function(self):
        mod1 = self._write_module("mod1", SIMPLE_CLASS)
        transformation = arch.RenameMethodTransformation(
            self.project, mod1, SIMPLE_CLASS.index("a_method"), "new_method"
        )
        transformation_changes = transformation.generate_changes()
        mod2 = self._write_module("mod2", SIMPLE_CLASS)
        refactoring = arch.RenameMethodRefactoring(
            self.project, mod2, SIMPLE_CLASS.index("a_method"), "new_method"
        )
        refactoring_changes = refactoring.generate_changes()
        self.assertEqual(
            transformation_changes.description, refactoring_changes.description
        )
        self.assertEqual(
            [change.new_contents for change in transformation_changes.changes],
            [change.new_contents for change in refactoring_changes.changes],
        )

    def test_generate_changes_warns_on_unsure_occurrences(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        refactoring = self._refactoring(mod, DUCK_TYPED)
        with self.assertRaises(arch.BehaviorPreservationWarning) as caught:
            refactoring.generate_changes()
        names = {condition.name for condition in caught.exception.conditions}
        self.assertIn("no-unsure-occurrences", names)

    def test_transformation_level_sets_the_warning_aside(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        transformation = arch.RenameMethodTransformation(
            self.project, mod, DUCK_TYPED.index("a_method"), "new_method"
        )
        changes = transformation.generate_changes()
        self.assertIsNotNone(changes)

    def test_refactoring_still_hard_fails_on_applicability(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS, new_name="lambda")
        with self.assertRaises(exceptions.RefactoringError):
            refactoring.generate_changes()


class RenameMethodDriverTest(RenameMethodArchMixin, unittest.TestCase):
    def _driver(self, module, code, policy, new_name="new_method", **kwds):
        refactoring = arch.RenameMethodRefactoring(
            self.project, module, code.index("a_method"), new_name, **kwds
        )
        return arch.RenameMethodDriver(refactoring, policy=policy)

    def test_rejects_unknown_policies(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = arch.RenameMethodRefactoring(
            self.project, mod, SIMPLE_CLASS.index("a_method"), "new_method"
        )
        with self.assertRaises(ValueError):
            arch.RenameMethodDriver(refactoring, policy="ask_the_user")

    def test_legacy_policy_returns_changes_without_warnings(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        result = self._driver(mod, DUCK_TYPED, arch.LEGACY).run()
        self.assertEqual(arch.LEGACY, result.mode)
        self.assertEqual([], result.warning_results)
        self.assertEqual(
            "Renaming <a_method> to <new_method>", result.changes.description
        )

    def test_legacy_policy_skips_behavior_preserving_checks(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        driver = self._driver(mod, DUCK_TYPED, arch.LEGACY)
        result = driver.run()
        self.assertIsNone(
            driver.refactoring._breaking_change_preconditions
        )
        self.assertIsNotNone(result.changes)

    def test_legacy_policy_hard_fails_on_keywords(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        driver = self._driver(mod, SIMPLE_CLASS, arch.LEGACY, new_name="lambda")
        with self.assertRaises(exceptions.RefactoringError):
            driver.run()

    def test_fail_on_warning_returns_no_changes(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        result = self._driver(mod, DUCK_TYPED, arch.FAIL_ON_WARNING).run()
        self.assertIsNone(result.changes)
        names = {condition.name for condition in result.warning_results}
        self.assertIn("no-unsure-occurrences", names)

    def test_fail_on_warning_returns_changes_when_clean(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        result = self._driver(mod, SIMPLE_CLASS, arch.FAIL_ON_WARNING).run()
        self.assertIsNotNone(result.changes)
        self.assertEqual([], result.warning_results)

    def test_proceed_after_warning_returns_both(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        result = self._driver(mod, DUCK_TYPED, arch.PROCEED_AFTER_WARNING).run()
        self.assertIsNotNone(result.changes)
        self.assertTrue(result.warning_results)
        self.assertTrue(result.warning_results[0].violators)

    def test_result_records_checked_applicability_conditions(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        result = self._driver(mod, SIMPLE_CLASS, arch.LEGACY).run()
        self.assertEqual(
            ["valid-name"],
            [condition.name for condition in result.applicability_results],
        )


if __name__ == "__main__":
    unittest.main()
