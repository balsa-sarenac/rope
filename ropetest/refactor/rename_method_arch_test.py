import unittest
from textwrap import dedent

from rope.base import exceptions
from rope.refactor import rename_method_arch as arch
from rope.refactor.rename import Rename
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

    def test_negated_condition_holds_when_the_inner_one_fails(self):
        negated = arch.NegatedCondition(arch.ValidNameCondition("lambda"))
        self.assertTrue(negated.check())
        self.assertEqual([], negated.violators)

    def test_negated_condition_fails_when_the_inner_one_holds(self):
        negated = arch.ValidNameCondition("new_method").not_()
        self.assertFalse(negated.check())
        self.assertIn("to fail", negated.error_string())

    def test_negated_condition_reports_non_violators(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        self._write_module("mod2", "")
        refactoring = arch.RenameMethodRefactoring(
            self.project,
            mod,
            SIMPLE_CLASS.index("a_method"),
            "new_method",
            resources=[mod],
        )
        refactoring.prepare_for_execution()
        coverage = refactoring.analysis_coverage_condition()
        self.assertFalse(coverage.check())
        self.assertEqual(
            ["mod2.py"], [resource.path for resource in coverage.violators]
        )
        negated = coverage.not_()
        self.assertFalse(negated.check())
        self.assertIn("mod1.py", [r.path for r in negated.violators])

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

    def test_hierarchy_condition_accepts_renaming_to_the_same_name(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        refactoring = self._refactoring(mod, SIMPLE_CLASS, new_name="a_method")
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertTrue(condition.check())

    def test_hierarchy_condition_accepts_no_op_rename_of_overrides(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            class B(A):
                def a_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, new_name="a_method", in_hierarchy=True
        )
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertTrue(condition.check())

    def test_hierarchy_condition_keeps_same_named_classes_apart(self):
        base_code = dedent("""\
            class Base(object):
                def a_method(self):
                    pass
        """)
        subclass_code = dedent("""\
            import mod1
            class C(mod1.Base):
                def a_method(self):
                    pass
                def new_method(self):
                    pass
        """)
        mod1 = self._write_module("mod1", base_code)
        self._write_module("mod2", subclass_code)
        self._write_module("mod3", subclass_code)
        refactoring = self._refactoring(mod1, base_code, in_hierarchy=True)
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertFalse(condition.check())
        # same class name, same conflict line, different modules:
        # both definitions are distinct violators
        self.assertEqual(2, len(condition.violators))
        self.assertNotEqual(
            condition.violators[0].module, condition.violators[1].module
        )

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

    def test_hierarchy_condition_finds_conflict_in_edited_subclass(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            class B(A):
                def a_method(self):
                    pass
                def new_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code, in_hierarchy=True)
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_conflict_condition()
        self.assertFalse(condition.check())
        self.assertEqual(
            ["B"],
            [violator.pyclass.get_name() for violator in condition.violators],
        )

    def test_fail_on_warning_rejects_edited_subclass_conflicts(self):
        code = dedent("""\
            class A(object):
                def a_method(self):
                    pass
            class B(A):
                def a_method(self):
                    pass
                def new_method(self):
                    pass
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(mod, code, in_hierarchy=True)
        result = arch.RenameMethodDriver(
            refactoring, policy=arch.FAIL_ON_WARNING
        ).run()
        self.assertIsNone(result.changes)
        names = {condition.name for condition in result.warning_results}
        self.assertIn("hierarchy-does-not-define-name", names)

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

    def test_fail_on_warning_accepts_a_no_op_rename(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        result = self._driver(
            mod, SIMPLE_CLASS, arch.FAIL_ON_WARNING, new_name="a_method"
        ).run()
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


class LegacyDelegationTest(RenameMethodArchMixin, unittest.TestCase):
    def test_get_changes_matches_the_legacy_driver_for_methods(self):
        mod = self._write_module("mod1", DUCK_TYPED)
        legacy_changes = Rename(
            self.project, mod, DUCK_TYPED.index("a_method")
        ).get_changes("new_method")
        refactoring = arch.RenameMethodRefactoring(
            self.project,
            mod,
            DUCK_TYPED.index("a_method"),
            "new_method",
            require_identifier=False,
        )
        driver_changes = arch.RenameMethodDriver(
            refactoring, policy=arch.LEGACY
        ).run().changes
        self.assertEqual(legacy_changes.description, driver_changes.description)
        self.assertEqual(
            [change.new_contents for change in legacy_changes.changes],
            [change.new_contents for change in driver_changes.changes],
        )

    def test_get_changes_keeps_keyword_error_for_methods(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        renamer = Rename(self.project, mod, SIMPLE_CLASS.index("a_method"))
        with self.assertRaises(exceptions.RefactoringError) as caught:
            renamer.get_changes("lambda")
        self.assertIn("keyword", str(caught.exception))

    def test_get_changes_keeps_legacy_name_leniency_for_methods(self):
        mod = self._write_module("mod1", SIMPLE_CLASS)
        renamer = Rename(self.project, mod, SIMPLE_CLASS.index("a_method"))
        changes = renamer.get_changes("not an identifier")
        self.assertIsNotNone(changes)

    def test_get_changes_unchanged_for_non_methods(self):
        code = "a_var = 1\nprint(a_var)\n"
        mod = self._write_module("mod1", code)
        changes = Rename(self.project, mod, code.index("a_var")).get_changes(
            "new_var"
        )
        self.project.do(changes)
        self.assertEqual("new_var = 1\nprint(new_var)\n", mod.read())


if __name__ == "__main__":
    unittest.main()
