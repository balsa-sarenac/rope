import unittest
from textwrap import dedent

from rope.base import exceptions
from rope.refactor import change_signature_arch as arch_cs
from rope.refactor.change_signature_arch import (
    NoDuplicateParameterCondition,
    ParameterExistsCondition,
)
from rope.refactor import arch
from rope.refactor import functionutils
from rope.refactor.change_signature import (
    ArgumentAdder,
    ArgumentDefaultInliner,
    ArgumentRemover,
    ArgumentReorderer,
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

    def _transformation(self, module, code, changers, **kwds):
        return arch_cs.ChangeSignatureTransformation(
            self.project, module, code.index("a_func") + 1, changers, **kwds
        )

    def _refactoring(self, module, code, changers, **kwds):
        return arch_cs.ChangeSignatureRefactoring(
            self.project, module, code.index("a_func") + 1, changers, **kwds
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


class ChangerConditionTest(ChangeSignatureArchMixin, unittest.TestCase):
    """Each argument changer states its own applicability conditions."""

    def _check(self, changer, info):
        arch.check_applicability_preconditions(_against(changer, info))

    def test_remove_of_named_parameter_is_applicable(self):
        self._check(ArgumentRemover(0), _definfo([("p1", None)]))

    def test_remove_of_missing_parameter_is_rejected(self):
        info = _definfo([("p1", None)])
        changer = ArgumentRemover(2)
        with self.assertRaises(exceptions.RefactoringError):
            self._check(changer, info)
        condition = _against(changer, info).applicability_preconditions()[0]
        self.assertFalse(condition.check())
        self.assertEqual([2], condition.violators)

    def test_remove_of_star_args_slot_is_applicable(self):
        self._check(
            ArgumentRemover(1), _definfo([("p1", None)], args_arg="args")
        )

    def test_remove_of_keywords_slot_is_applicable(self):
        self._check(
            ArgumentRemover(2),
            _definfo([("p1", None)], args_arg="args", keywords_arg="kwds"),
        )

    def test_add_of_duplicate_parameter_is_rejected_with_legacy_message(self):
        info = _definfo([("p1", None)])
        condition = _against(ArgumentAdder(0, "p1"), info).applicability_preconditions()[1]
        self.assertFalse(condition.check())
        self.assertEqual(
            "Adding duplicate parameter: <p1>.", condition.error_string()
        )

    def test_add_of_keyword_named_parameter_is_rejected(self):
        with self.assertRaises(exceptions.RefactoringError):
            self._check(ArgumentAdder(0, "lambda"), _definfo([]))

    def test_reorder_with_invalid_index_is_rejected(self):
        with self.assertRaises(exceptions.RefactoringError):
            self._check(
                ArgumentReorderer([1, 0, 2]),
                _definfo([("p1", None), ("p2", None)]),
            )

    def test_reorder_of_a_prefix_is_applicable(self):
        self._check(
            ArgumentReorderer([0]), _definfo([("p1", None), ("p2", None)])
        )

    def test_inline_of_missing_parameter_is_rejected(self):
        with self.assertRaises(exceptions.RefactoringError):
            self._check(ArgumentDefaultInliner(1), _definfo([("p1", "1")]))

    def test_changer_conditions_are_applicability_level(self):
        conditions = _against(
            ArgumentRemover(0), _definfo([("p1", None)])
        ).applicability_preconditions()
        self.assertEqual({arch.APPLICABILITY}, {c.level for c in conditions})

    def test_a_changer_subclass_inherits_its_conditions(self):
        class UpperCaseAdder(ArgumentAdder):
            pass

        mod = self._write_module("mod1", TWO_PARAMS)
        changer = UpperCaseAdder(0, "p3")
        transformation = self._transformation(mod, TWO_PARAMS, [changer])
        transformation.prepare_for_execution()
        transformation.check_preconditions()
        child = transformation.child_transformations()[0]
        self.assertIs(changer, child)
        self.assertEqual(
            [arch.ValidNameCondition, NoDuplicateParameterCondition],
            [type(c) for c in child.applicability_preconditions()],
        )

    def test_a_changer_without_conditions_contributes_none(self):
        from rope.refactor.change_signature import ArgumentNormalizer

        self.assertEqual(
            [],
            _against(ArgumentNormalizer(), _definfo([])).applicability_preconditions(),
        )


def _against(changer, info):
    """A changer holding the signature its conditions read."""
    changer.definition_info = info
    return changer


class CompositeTransformationTest(ChangeSignatureArchMixin, unittest.TestCase):
    def test_prepare_rejects_non_functions(self):
        code = "a_func = 1\n"
        mod = self._write_module("mod1", code)
        transformation = self._transformation(mod, code, [])
        with self.assertRaises(exceptions.RefactoringError):
            transformation.prepare_for_execution()

    def test_each_child_sees_its_predecessors_edits(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        changers = [
            ArgumentAdder(2, "p3"),
            ArgumentReorderer([1, 0, 2]),
        ]
        transformation = self._transformation(mod, TWO_PARAMS, changers)
        transformation.prepare_for_execution()
        transformation.check_preconditions()
        self.assertEqual(
            ["p1", "p2", "p3"],
            [
                pair[0]
                for pair in transformation.child_transformations()[1]
                .definition_info.args_with_defaults
            ],
        )

    def test_target_offset_follows_a_shrinking_edit_before_the_def(self):
        code = dedent("""\
            def caller():
                return a_func(1, 222222222)
            def a_func(p1, p2):
                pass
        """)
        mod = self._write_module("mod1", code)
        transformation = arch_cs.ChangeSignatureTransformation(
            self.project,
            mod,
            code.index("def a_func") + 5,
            [ArgumentRemover(1), ArgumentAdder(1, "p3", default="0")],
        )
        self.project.do(transformation.generate_changes())
        self.assertEqual(
            dedent("""\
                def caller():
                    return a_func(1)
                def a_func(p1, p3=0):
                    pass
            """),
            mod.read(),
        )

    def test_target_offset_follows_a_growing_edit_before_the_def(self):
        code = dedent("""\
            def caller():
                return a_func(1, 2)
            def a_func(p1, p2):
                pass
        """)
        mod = self._write_module("mod1", code)
        transformation = arch_cs.ChangeSignatureTransformation(
            self.project,
            mod,
            code.index("def a_func") + 5,
            [ArgumentAdder(2, "p3", value="333333333"), ArgumentRemover(0)],
        )
        self.project.do(transformation.generate_changes())
        self.assertEqual(
            dedent("""\
                def caller():
                    return a_func(2, 333333333)
                def a_func(p2, p3):
                    pass
            """),
            mod.read(),
        )

    def test_step_invalid_against_original_signature_is_rejected(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        steps = [ArgumentReorderer([1, 0, 2])]
        transformation = self._transformation(mod, TWO_PARAMS, steps)
        transformation.prepare_for_execution()
        with self.assertRaises(exceptions.RefactoringError):
            transformation.check_preconditions()

    def test_each_child_states_its_own_applicability(self):
        mod = self._write_module("mod1", TWO_PARAMS)
        changers = [
            ArgumentAdder(2, "p3"),
            ArgumentRemover(0),
        ]
        transformation = self._transformation(mod, TWO_PARAMS, changers)
        transformation.prepare_for_execution()
        # the composite holds none of its own; each child checks at its
        # own point in the sequence, against what its predecessors left
        self.assertEqual([], transformation.applicability_preconditions())
        transformation.check_preconditions()
        self.assertEqual(
            [ParameterExistsCondition],
            [
                type(c)
                for c in transformation.child_transformations()[1]
                .applicability_preconditions()
            ],
        )

    def test_generate_changes_builds_an_ordinary_changeset(self):
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod = self._write_module("mod1", code)
        transformation = self._transformation(
            mod, code, [ArgumentRemover(0)]
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
            mod, code, [ArgumentRemover(0)]
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
            mod, code, [ArgumentAdder(0, "p1")]
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
                ArgumentAdder(2, "p3", "None"),
                ArgumentReorderer([1, 0, 2]),
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
                ArgumentRemover(2),
                ArgumentAdder(2, "p3", None, "3"),
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
            mod, code, [arch_cs.RemoveParameterRefactoring(ArgumentRemover(0))]
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

    def _parameter_unused_condition(self, code, index, module="mod1"):
        mod = self._write_module(module, code)
        refactoring = self._refactoring(
            mod,
            code,
            [arch_cs.RemoveParameterRefactoring(ArgumentRemover(index))],
        )
        refactoring.prepare_for_execution()
        (condition,) = [
            condition
            for condition in refactoring.breaking_change_preconditions()
            if condition.name == "removed-parameter-unused"
        ]
        return mod, condition

    def test_removed_parameter_read_in_the_body_is_reported(self):
        code = dedent("""\
            def a_func(p1, p2):
                return p1 + p2
        """)
        mod, condition = self._parameter_unused_condition(code, 1)
        self.assertFalse(condition.check())
        self.assertEqual([2], [occurrence.lineno for occurrence in condition.violators])
        self.assertEqual(mod, condition.violators[0].resource)
        self.assertIn("<p2>", condition.error_string())
        self.assertIn("mod1.py:2", condition.error_string())

    def test_removed_parameter_not_read_in_the_body_is_clean(self):
        code = dedent("""\
            def a_func(p1, p2):
                return p1
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertTrue(condition.check())
        self.assertEqual([], condition.subjects())

    def test_removed_parameter_rebound_in_the_body_is_reported_conservatively(self):
        code = dedent("""\
            def a_func(p1, p2):
                p2 = 1
                return p2
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertFalse(condition.check())
        self.assertEqual([2, 3], [occurrence.lineno for occurrence in condition.violators])

    def test_keyword_argument_at_a_call_site_is_not_a_body_read(self):
        code = dedent("""\
            def a_func(p1, p2):
                return p1
            a_func(1, p2=2)
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertTrue(condition.check())

    def test_positional_call_before_the_def_is_not_a_body_read(self):
        code = dedent("""\
            def caller(p2):
                return a_func(1, p2)
            def a_func(p1, p2):
                return p1
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertTrue(condition.check())

    def test_body_read_is_found_after_a_non_ascii_default(self):
        code = dedent("""\
            def a_func(p1, p2="é"):
                return p2
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertFalse(condition.check())
        self.assertEqual([2], [o.lineno for o in condition.violators])

    def test_body_read_is_found_after_a_docstring(self):
        code = dedent("""\
            def a_func(p1, p2):
                \"\"\"p2 is documented, not read, here.\"\"\"
                return p2
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertFalse(condition.check())
        self.assertEqual([3], [o.lineno for o in condition.violators])

    def test_body_read_is_found_in_a_one_line_def(self):
        code = "def a_func(p1, p2): return p2\n"
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertFalse(condition.check())
        self.assertEqual([1], [o.lineno for o in condition.violators])

    def test_body_read_condition_can_be_negated(self):
        """One scan per check keeps subjects and violators the same objects.

        Only the read direction is meaningful: when the parameter is
        not read the range is empty, and `NegatedCondition` reports the
        complement of an empty range, which holds vacuously.
        """
        code = dedent("""\
            def a_func(p1, p2):
                return p2
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        negated = condition.not_()
        self.assertTrue(negated.check())
        self.assertEqual([], negated.violators)

    def test_removed_star_args_slot_is_out_of_scope(self):
        code = dedent("""\
            def a_func(p1, *args):
                return args
        """)
        _, condition = self._parameter_unused_condition(code, 1)
        self.assertTrue(condition.check())

    def test_remove_parameter_refactoring_names_both_conditions(self):
        code = dedent("""\
            def a_func(p1):
                pass
        """)
        mod = self._write_module("mod1", code)
        child = arch_cs.RemoveParameterRefactoring(ArgumentRemover(0))
        refactoring = self._refactoring(mod, code, [child])
        refactoring.prepare_for_execution()
        self.assertEqual(
            ["no-argument-value-lost", "removed-parameter-unused"],
            [c.name for c in child.breaking_change_preconditions()],
        )
        self.assertIs(
            type(child.argument_value_condition()), arch_cs.NoArgumentValueLostCondition
        )
        self.assertIs(
            type(child.parameter_unused_condition()),
            arch_cs.RemovedParameterUnusedCondition,
        )

    def test_removed_parameter_never_passed_is_clean(self):
        code = dedent("""\
            def a_func(p1=1):
                pass
            a_func()
        """)
        mod = self._write_module("mod1", code)
        refactoring = self._refactoring(
            mod, code, [arch_cs.RemoveParameterRefactoring(ArgumentRemover(0))]
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
            mod, code, [arch_cs.AddParameterRefactoring(ArgumentAdder(0, "p1"))]
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
            mod, code, [arch_cs.AddParameterRefactoring(ArgumentAdder(0, "p1", "None"))]
        )
        refactoring.prepare_for_execution()
        refactoring.check_breaking_change_preconditions()

    def test_hierarchy_override_left_behind_is_reported(self):
        mod = self._write_module("mod1", HIERARCHY)
        refactoring = self._refactoring(
            mod, HIERARCHY, [ArgumentRemover(1)]
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
            [ArgumentRemover(1)],
            in_hierarchy=True,
        )
        refactoring.prepare_for_execution()
        condition = refactoring.hierarchy_overrides_condition()
        self.assertTrue(condition.check())

    def test_unsure_receiver_is_reported(self):
        mod = self._write_module("mod1", UNSURE_CALL)
        refactoring = self._refactoring(
            mod, UNSURE_CALL, [ArgumentRemover(1)]
        )
        refactoring.prepare_for_execution()
        condition = refactoring.unsure_occurrences_condition()
        self.assertFalse(condition.check())
        self.assertEqual(5, condition.violators[0].lineno)

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
            [arch_cs.RemoveParameterRefactoring(ArgumentRemover(0))],
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
            mod, self.WARNING_CODE, [arch_cs.RemoveParameterRefactoring(ArgumentRemover(0))]
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


class LegacyEntryPointBehaviorChangeTest(
    ChangeSignatureArchMixin, unittest.TestCase
):
    """`ChangeSignature.get_changes` runs the composite, not the folded pass.

    These pin the four inputs on which the legacy entry point now
    behaves differently from rope's original single pass, so the
    difference is a recorded decision rather than an accident.  The
    deprecated convenience methods still run the folded pass.
    """

    def _signature(self, code):
        mod = self._write_module("mod1", code)
        return mod, ChangeSignature(self.project, mod, code.index("a_func") + 1)

    def test_remove_then_re_add_drops_the_stale_argument(self):
        """Executing the children in sequence removes a folded-pass artifact.

        `ArgumentRemover.change_argument_mapping` looks up
        ``args_with_defaults[0]`` (a tuple) instead of the removed
        parameter's name, so the supplied value was never purged from
        the mapping.  Folding both changers into one pass therefore
        left the old call argument in place.  Each child now rewrites
        the program its predecessor produced, so the removal really
        happens before the addition is analyzed and the stale argument
        cannot survive.
        """
        code = dedent("""\
            def a_func(p1):
                pass
            a_func(1)
        """)
        mod, signature = self._signature(code)
        changes = signature.get_changes(
            [ArgumentRemover(0), ArgumentAdder(0, "p1")]
        )
        self.project.do(changes)
        self.assertEqual(
            dedent("""\
                def a_func(p1):
                    pass
                a_func()
            """),
            mod.read(),
        )

    def test_children_must_come_from_ropes_changer_hierarchy(self):
        """`get_changes` documents `changers` as `_ArgumentChanger`s.

        An object outside that hierarchy provides the two edit
        functions but cannot act as a composite child, which resolves
        its own target and constructs its own changes.  The legacy
        folded pass tolerated such an object; composition refuses it.
        """

        class Custom:
            def change_definition_info(self, definition_info):
                pass

            def change_argument_mapping(self, definition_info, mapping):
                pass

        _, signature = self._signature(TWO_PARAMS)
        with self.assertRaisesRegex(exceptions.RefactoringError, "Custom"):
            signature.get_changes([Custom()])

    def test_removing_a_missing_index_is_rejected(self):
        """The folded pass silently did nothing; the slot is now checked."""
        _, signature = self._signature(TWO_PARAMS)
        with self.assertRaisesRegex(exceptions.RefactoringError, "index <5>"):
            signature.get_changes([ArgumentRemover(5)])

    def test_inlining_a_missing_index_is_rejected(self):
        """The folded pass raised a bare IndexError from call rewriting."""
        _, signature = self._signature(TWO_PARAMS)
        with self.assertRaisesRegex(exceptions.RefactoringError, "index <3>"):
            signature.get_changes([ArgumentDefaultInliner(3)])

    def test_removing_a_negative_index_is_rejected(self):
        """The folded pass honoured ``-1`` through Python's list indexing."""
        _, signature = self._signature(TWO_PARAMS)
        with self.assertRaisesRegex(exceptions.RefactoringError, "index <-1>"):
            signature.get_changes([ArgumentRemover(-1)])


if __name__ == "__main__":
    unittest.main()
