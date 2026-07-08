import unittest
from textwrap import dedent

from rope.base import exceptions
from rope.refactor import change_signature_arch as arch_cs
from rope.refactor import arch
from rope.refactor import functionutils
from rope.refactor.change_signature import (
    ArgumentAdder,
    ArgumentRemover,
    ChangeSignature,
)
from ropetest import testutils


class ChangeSignatureArchMixin:
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

    def _transformation(self, module, code, steps, **kwds):
        return arch_cs.ChangeSignatureTransformation(
            self.project, module, code.index("a_func") + 1, steps, **kwds
        )

    def _refactoring(self, module, code, steps, **kwds):
        return arch_cs.ChangeSignatureRefactoring(
            self.project, module, code.index("a_func") + 1, steps, **kwds
        )


def _definfo(args_with_defaults, args_arg=None, keywords_arg=None):
    return functionutils.DefinitionInfo(
        "a_func", False, list(args_with_defaults), args_arg, keywords_arg
    )


TWO_PARAMS = dedent("""\
    def a_func(p1, p2):
        pass
    a_func(1, 2)
""")


class StepConditionTest(ChangeSignatureArchMixin, unittest.TestCase):
    def _bound(self, step, info):
        step.bind(None, info)
        return step

    def test_unbound_step_cannot_be_checked(self):
        step = arch_cs.RemoveParameterTransformation(0)
        with self.assertRaises(exceptions.RefactoringError):
            step.check_preconditions()

    def test_remove_of_named_parameter_is_applicable(self):
        step = self._bound(
            arch_cs.RemoveParameterTransformation(0), _definfo([("p1", None)])
        )
        step.check_preconditions()

    def test_remove_of_missing_parameter_is_rejected(self):
        step = self._bound(
            arch_cs.RemoveParameterTransformation(2), _definfo([("p1", None)])
        )
        with self.assertRaises(exceptions.RefactoringError):
            step.check_preconditions()
        condition = step.applicability_preconditions()[0]
        self.assertFalse(condition.check())
        self.assertEqual(2, condition.violators[0].subject)

    def test_remove_of_star_args_slot_is_applicable(self):
        step = self._bound(
            arch_cs.RemoveParameterTransformation(1),
            _definfo([("p1", None)], args_arg="args"),
        )
        step.check_preconditions()

    def test_remove_of_keywords_slot_is_applicable(self):
        step = self._bound(
            arch_cs.RemoveParameterTransformation(2),
            _definfo([("p1", None)], args_arg="args", keywords_arg="kwds"),
        )
        step.check_preconditions()

    def test_add_of_duplicate_parameter_is_rejected_with_legacy_message(self):
        step = self._bound(
            arch_cs.AddParameterTransformation(0, "p1"), _definfo([("p1", None)])
        )
        condition = step.applicability_preconditions()[1]
        self.assertFalse(condition.check())
        self.assertEqual(
            "Adding duplicate parameter: <p1>.", condition.error_string()
        )

    def test_add_of_keyword_named_parameter_is_rejected(self):
        step = self._bound(
            arch_cs.AddParameterTransformation(0, "lambda"), _definfo([])
        )
        with self.assertRaises(exceptions.RefactoringError):
            step.check_preconditions()

    def test_reorder_with_invalid_index_is_rejected(self):
        step = self._bound(
            arch_cs.ReorderParametersTransformation([1, 0, 2]),
            _definfo([("p1", None), ("p2", None)]),
        )
        with self.assertRaises(exceptions.RefactoringError):
            step.check_preconditions()

    def test_reorder_of_a_prefix_is_applicable(self):
        step = self._bound(
            arch_cs.ReorderParametersTransformation([0]),
            _definfo([("p1", None), ("p2", None)]),
        )
        step.check_preconditions()

    def test_inline_of_missing_parameter_is_rejected(self):
        step = self._bound(
            arch_cs.InlineParameterDefaultTransformation(1),
            _definfo([("p1", "1")]),
        )
        with self.assertRaises(exceptions.RefactoringError):
            step.check_preconditions()

    def test_step_conditions_are_applicability_level(self):
        step = self._bound(
            arch_cs.RemoveParameterTransformation(0), _definfo([("p1", None)])
        )
        levels = {c.level for c in step.applicability_preconditions()}
        self.assertEqual({arch.APPLICABILITY}, levels)


class CompositeTransformationTest(ChangeSignatureArchMixin, unittest.TestCase):
    def test_prepare_rejects_non_functions(self):
        code = "a_func = 1\n"
        mod = self._write_module("mod1", code)
        transformation = self._transformation(mod, code, [])
        with self.assertRaises(exceptions.RefactoringError):
            transformation.prepare_for_execution()

    def test_late_configuration_binds_each_step_to_prior_output(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        steps = [
            arch_cs.AddParameterTransformation(2, "p3"),
            arch_cs.ReorderParametersTransformation([1, 0, 2]),
        ]
        transformation = self._transformation(mod, TWO_PARAMS, steps)
        transformation.prepare_for_execution()
        transformation.check_preconditions()
        self.assertEqual(
            ["p1", "p2", "p3"],
            [pair[0] for pair in steps[1].input_definition_info.args_with_defaults],
        )

    def test_step_invalid_against_original_signature_is_rejected(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        steps = [arch_cs.ReorderParametersTransformation([1, 0, 2])]
        transformation = self._transformation(mod, TWO_PARAMS, steps)
        transformation.prepare_for_execution()
        with self.assertRaises(exceptions.RefactoringError):
            transformation.check_preconditions()

    def test_applicability_is_aggregated_from_the_steps(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        steps = [
            arch_cs.AddParameterTransformation(2, "p3"),
            arch_cs.RemoveParameterTransformation(0),
        ]
        transformation = self._transformation(mod, TWO_PARAMS, steps)
        transformation.prepare_for_execution()
        aggregated = [
            type(c) for c in transformation.applicability_preconditions()
        ]
        per_step = [
            type(c)
            for step in steps
            for c in step.applicability_preconditions()
        ]
        self.assertEqual(per_step, aggregated)

    def test_generate_changes_builds_an_ordinary_changeset(self):
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod = self._write_module("mod1", code)
        transformation = self._transformation(
            mod, code, [arch_cs.RemoveParameterTransformation(0)]
        )
        changes = transformation.generate_changes()
        self.assertEqual("Changing signature of <a_func>", changes.description)
        self.project.do(changes)
        self.assertEqual(
            dedent("""\
                def a_func():
                    pass
                a_func()
            """),
            mod.read(),
        )

    def test_execute_supports_undo_through_history(self):
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod = self._write_module("mod1", code)
        transformation = self._transformation(
            mod, code, [arch_cs.RemoveParameterTransformation(0)]
        )
        transformation.execute()
        self.project.history.undo()
        self.assertEqual(code, mod.read())

    def test_duplicate_add_raises_from_generate_changes(self):
        code = dedent("""\
            def a_func(p1):
                pass
        """)
        mod = self._write_module("mod1", code)
        transformation = self._transformation(
            mod, code, [arch_cs.AddParameterTransformation(0, "p1")]
        )
        with self.assertRaises(exceptions.RefactoringError):
            transformation.generate_changes()


class TwoLevelEquivalenceTest(ChangeSignatureArchMixin, unittest.TestCase):
    def _assert_same_changes(self, code, make_steps, **kwds):
        mod = self._write_module("mod1", code)
        transformation = self._transformation(mod, code, make_steps(), **kwds)
        refactoring = self._refactoring(mod, code, make_steps(), **kwds)
        transformation_changes = transformation.generate_changes()
        refactoring_changes = refactoring.generate_changes()
        self.assertEqual(
            transformation_changes.description, refactoring_changes.description
        )
        self.assertEqual(
            [change.new_contents for change in transformation_changes.changes],
            [change.new_contents for change in refactoring_changes.changes],
        )

    def test_both_levels_share_one_change_function(self):
        code = dedent("""\
            def a_func(p1, p2):
                pass
            a_func(1, 2)
        """)
        self._assert_same_changes(
            code,
            lambda: [
                arch_cs.AddParameterTransformation(2, "p3", "None"),
                arch_cs.ReorderParametersTransformation([1, 0, 2]),
            ],
        )

    def test_legacy_entry_point_matches_the_transformation(self):
        code = dedent("""\
            class A(object):
                def a_func(self, p1, p2):
                    pass
            A().a_func(1, p2=2)
        """)
        mod = self._write_module("mod1", code)
        legacy_changes = ChangeSignature(
            self.project, mod, code.index("a_func") + 1
        ).get_changes([ArgumentRemover(2), ArgumentAdder(2, "p3", None, "3")])
        transformation = self._transformation(
            mod,
            code,
            [
                arch_cs.RemoveParameterTransformation(2),
                arch_cs.AddParameterTransformation(2, "p3", None, "3"),
            ],
        )
        transformation_changes = transformation.generate_changes()
        self.assertEqual(
            [change.new_contents for change in legacy_changes.changes],
            [change.new_contents for change in transformation_changes.changes],
        )


UNSURE_CALL = dedent("""\
    class A(object):
        def a_func(self, p1):
            pass
    def f(arg):
        arg.a_func(1)
""")

HIERARCHY = dedent("""\
    class A(object):
        def a_func(self, p1):
            pass
    class B(A):
        def a_func(self, p1):
            pass
""")


class BehaviorPreservingConditionTest(ChangeSignatureArchMixin, unittest.TestCase):
    def test_removed_parameter_with_supplied_value_is_reported(self):
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.RemoveParameterRefactoring(0)]
        )
        refactoring.prepare_for_execution()
        conditions = [
            condition
            for condition in refactoring.breaking_change_preconditions()
            if condition.name == "no-argument-value-lost"
        ]
        condition = conditions[0]
        self.assertFalse(condition.check())
        record = condition.violators[0]
        self.assertEqual(mod, record.resource)
        self.assertEqual(3, record.lineno)
        self.assertIn("mod1.py:3", condition.error_string())

    def test_removed_parameter_never_passed_is_clean(self):
        code = dedent("""\
            def a_func(p1=1):
                pass
            a_func()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.RemoveParameterRefactoring(0)]
        )
        refactoring.prepare_for_execution()
        refactoring.check_breaking_change_preconditions()

    def test_required_parameter_add_reports_every_call_site(self):
        code = dedent("""\
            def a_func():
                pass
            a_func()
            a_func()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.AddParameterRefactoring(0, "p1")]
        )
        refactoring.prepare_for_execution()
        conditions = [
            condition
            for condition in refactoring.breaking_change_preconditions()
            if condition.name == "call-sites-receive-required-argument"
        ]
        condition = conditions[0]
        self.assertFalse(condition.check())
        self.assertEqual([3, 4], [record.lineno for record in condition.violators])

    def test_parameter_add_with_default_is_clean(self):
        code = dedent("""\
            def a_func():
                pass
            a_func()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.AddParameterRefactoring(0, "p1", "None")]
        )
        refactoring.prepare_for_execution()
        refactoring.check_breaking_change_preconditions()

    def test_hierarchy_override_left_behind_is_reported(self):
        mod = self._write_module("mod1", HIERARCHY)
        refactoring = self._refactoring(
            mod, HIERARCHY, [arch_cs.RemoveParameterRefactoring(1)]
        )
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_overrides_condition()
        self.assertFalse(condition.check())
        resource, lineno = condition.violators[0]
        self.assertEqual(mod, resource)
        self.assertEqual(5, lineno)

    def test_in_hierarchy_covers_the_overrides(self):
        mod = self._write_module("mod1", HIERARCHY)
        refactoring = self._refactoring(
            mod,
            HIERARCHY,
            [arch_cs.RemoveParameterRefactoring(1)],
            in_hierarchy=True,
        )
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_overrides_condition()
        self.assertTrue(condition.check())

    def test_unsure_receiver_is_reported(self):
        mod = self._write_module("mod1", UNSURE_CALL)
        refactoring = self._refactoring(
            mod, UNSURE_CALL, [arch_cs.RemoveParameterRefactoring(1)]
        )
        refactoring.prepare_for_execution()
        condition = refactoring.unsure_occurrences_condition()
        self.assertFalse(condition.check())
        self.assertEqual(5, condition.violators[0].lineno)

    def test_reflective_reference_is_reported(self):
        code = dedent("""\
            class A(object):
                def a_func(self, p1):
                    pass
            getattr(A(), "a_func")(1)
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.RemoveParameterRefactoring(1)]
        )
        refactoring.prepare_for_execution()
        condition = refactoring.reflective_references_condition()
        self.assertFalse(condition.check())
        self.assertEqual("getattr", condition.violators[0].kind)

    def test_restricted_analysis_reports_excluded_clients(self):
        code = dedent("""\
            def a_func(p1):
                pass
        """)
        mod1 = self._write_module("mod1", code)
        mod2 = self._write_module("mod2", "import mod1\nmod1.a_func(1)\n")
        refactoring = self._refactoring(
            mod1,
            code,
            [arch_cs.RemoveParameterRefactoring(0)],
            resources=[mod1],
        )
        refactoring.prepare_for_execution()
        condition = refactoring.analysis_coverage_condition()
        self.assertFalse(condition.check())
        self.assertEqual([mod2], condition.violators)


class DriverPolicyTest(ChangeSignatureArchMixin, unittest.TestCase):
    WARNING_CODE = dedent("""\
        def a_func(p1):
            pass
        a_func(1)
    """)

    def _driver(self, policy):
        mod = self._write_module("mod1", self.WARNING_CODE)
        refactoring = self._refactoring(
            mod, self.WARNING_CODE, [arch_cs.RemoveParameterRefactoring(0)]
        )
        return arch.RefactoringDriver(refactoring, policy=policy)

    def test_legacy_policy_skips_behavior_preserving_conditions(self):
        result = self._driver(arch.LEGACY).run()
        self.assertIsNotNone(result.changes)
        self.assertEqual([], result.warning_results)
        self.assertEqual(arch.LEGACY, result.mode)

    def test_fail_on_warning_returns_conditions_and_no_changes(self):
        result = self._driver(arch.FAIL_ON_WARNING).run()
        self.assertIsNone(result.changes)
        self.assertTrue(result.warning_results)

    def test_proceed_after_warning_returns_both(self):
        result = self._driver(arch.PROCEED_AFTER_WARNING).run()
        self.assertIsNotNone(result.changes)
        self.assertTrue(result.warning_results)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            arch.RefactoringDriver(object(), policy="whatever")


class LegacyCompatibilityCharacterizationTest(
    ChangeSignatureArchMixin, unittest.TestCase
):
    def test_remove_then_re_add_keeps_the_stale_argument(self):
        """Pins the legacy ArgumentRemover mapping bug.

        `ArgumentRemover.change_argument_mapping` looks up
        ``args_with_defaults[0]`` (a tuple) instead of the removed
        parameter's name, so the supplied value is never purged from
        the mapping.  Removing `p1` and re-adding it therefore keeps
        the old call argument.  Preserved for characterization
        fidelity; making parameter flow explicit is what surfaced it.
        """
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod = self._write_module("mod1", code)
        signature = ChangeSignature(self.project, mod, code.index("a_func") + 1)
        changes = signature.get_changes(
            [ArgumentRemover(0), ArgumentAdder(0, "p1")]
        )
        self.project.do(changes)
        self.assertEqual(
            dedent("""\
                def a_func(p1):
                    pass
                a_func(1)
            """),
            mod.read(),
        )

    def test_custom_changer_instances_are_passed_through(self):
        class UpperCaseAdder(ArgumentAdder):
            pass

        changer = UpperCaseAdder(0, "p1")
        steps = arch_cs.steps_for_changers([changer])
        self.assertIsInstance(steps[0], arch_cs.AddParameterRefactoring)
        self.assertIs(changer, steps[0].transformation.changer)

    def test_unknown_changer_gets_the_generic_step(self):
        class Custom:
            def change_definition_info(self, definition_info):
                pass

            def change_argument_mapping(self, definition_info, mapping):
                pass

        changer = Custom()
        steps = arch_cs.steps_for_changers([changer])
        self.assertIsInstance(steps[0], arch_cs.GenericChangerTransformation)
        self.assertIs(changer, steps[0].changer)


if __name__ == "__main__":
    unittest.main()
