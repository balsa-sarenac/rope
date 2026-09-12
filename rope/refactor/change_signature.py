import copy

import rope.base.exceptions
from rope.base import codeanalyze, evaluate, pyobjects, taskhandle, utils, worder
from rope.base.change import ChangeContents, ChangeSet
from rope.refactor import arch, functionutils, occurrences


def _resolve_signature_target(project, resource, offset, pending=None):
    """Resolve the selected offset to a signature-change target.

    Returns ``(name, primary, pyname, others)``.  A class selection is
    redirected to its ``__init__``; for constructors `others` carries
    the ``(class_name, class_pyname)`` pair used to update
    ``ClassName(...)`` call sites.  Both the legacy path and the
    architecture path resolve through here, so they cannot drift.

    With a `pending` view (`arch.PendingChanges`), the target is
    resolved against the program including changes not yet applied,
    which is what lets a composite's later children see the edits of
    their predecessors.
    """
    if pending is not None:
        source = pending.source(resource)
        name = worder.Worder(source).get_word_at(offset)
        this_pymodule = pending.pymodule(resource)
    else:
        name = worder.get_name_at(resource, offset)
        this_pymodule = project.get_pymodule(resource)
    primary, pyname = evaluate.eval_location2(this_pymodule, offset)
    if pyname is None:
        return name, primary, None, None
    pyobject = pyname.get_object()
    if isinstance(pyobject, pyobjects.PyClass) and "__init__" in pyobject:
        pyname = pyobject["__init__"]
        name = "__init__"
    pyobject = pyname.get_object()
    others = None
    if (
        name == "__init__"
        and isinstance(pyobject, pyobjects.PyFunction)
        and isinstance(pyobject.parent, pyobjects.PyClass)
    ):
        pyclass = pyobject.parent
        others = (pyclass.get_name(), pyclass.parent[pyclass.get_name()])
    return name, primary, pyname, others


class ChangeSignature:
    def __init__(self, project, resource, offset):
        self.project = project
        self.resource = resource
        self.offset = offset
        self._set_name_and_pyname()
        if (
            self.pyname is None
            or self.pyname.get_object() is None
            or not isinstance(self.pyname.get_object(), pyobjects.PyFunction)
        ):
            raise rope.base.exceptions.RefactoringError(
                "Change method signature should be performed on functions"
            )

    def _set_name_and_pyname(self):
        (self.name, self.primary, self.pyname, self.others) = (
            _resolve_signature_target(self.project, self.resource, self.offset)
        )

    def _change_calls(
        self,
        call_changer,
        in_hierarchy=None,
        resources=None,
        handle=taskhandle.DEFAULT_TASK_HANDLE,
    ):
        if resources is None:
            resources = self.project.get_python_files()
        changes = ChangeSet("Changing signature of <%s>" % self.name)
        job_set = handle.create_jobset("Collecting Changes", len(resources))
        finder = occurrences.create_finder(
            self.project,
            self.name,
            self.pyname,
            instance=self.primary,
            in_hierarchy=in_hierarchy and self.is_method(),
        )
        if self.others:
            name, pyname = self.others
            constructor_finder = occurrences.create_finder(
                self.project, name, pyname, only_calls=True
            )
            finder = _MultipleFinders([finder, constructor_finder])
        for file in resources:
            job_set.started_job(file.path)
            change_calls = _ChangeCallsInModule(
                self.project, finder, file, call_changer
            )
            changed_file = change_calls.get_changed_module()
            if changed_file is not None:
                changes.add_change(ChangeContents(file, changed_file))
            job_set.finished_job()
        return changes

    def get_args(self):
        """Get function arguments.

        Return a list of ``(name, default)`` tuples for all but star
        and double star arguments.  For arguments that don't have a
        default, `None` will be used.
        """
        return self._definfo().args_with_defaults

    def is_method(self):
        pyfunction = self.pyname.get_object()
        return isinstance(pyfunction.parent, pyobjects.PyClass)

    @utils.deprecated("Use `ChangeSignature.get_args()` instead")
    def get_definition_info(self):
        return self._definfo()

    def _definfo(self):
        return functionutils.DefinitionInfo.read(self.pyname.get_object())

    @utils.deprecated()
    def normalize(self):
        changer = _FunctionChangers(
            self.pyname.get_object(), self.get_definition_info(), [ArgumentNormalizer()]
        )
        return self._change_calls(changer)

    @utils.deprecated()
    def remove(self, index):
        changer = _FunctionChangers(
            self.pyname.get_object(),
            self.get_definition_info(),
            [ArgumentRemover(index)],
        )
        return self._change_calls(changer)

    @utils.deprecated()
    def add(self, index, name, default=None, value=None):
        changer = _FunctionChangers(
            self.pyname.get_object(),
            self.get_definition_info(),
            [ArgumentAdder(index, name, default, value)],
        )
        return self._change_calls(changer)

    @utils.deprecated()
    def inline_default(self, index):
        changer = _FunctionChangers(
            self.pyname.get_object(),
            self.get_definition_info(),
            [ArgumentDefaultInliner(index)],
        )
        return self._change_calls(changer)

    @utils.deprecated()
    def reorder(self, new_ordering):
        changer = _FunctionChangers(
            self.pyname.get_object(),
            self.get_definition_info(),
            [ArgumentReorderer(new_ordering)],
        )
        return self._change_calls(changer)

    def get_changes(
        self,
        changers,
        in_hierarchy=False,
        resources=None,
        task_handle=taskhandle.DEFAULT_TASK_HANDLE,
    ):
        """Get changes caused by this refactoring

        `changers` is a list of `_ArgumentChanger`.  If `in_hierarchy`
        is `True` the changers are applied to all matching methods in
        the class hierarchy.
        `resources` can be a list of `rope.base.resource.File` that
        should be searched for occurrences; if `None` all python files
        in the project are searched.

        The changers run as a composite (`change_signature_arch`), each
        against the signature its predecessors produced, under the
        LEGACY warning policy.  Four inputs therefore behave differently
        from the folded single pass this method used to run; the
        deprecated `remove`, `add`, `reorder`, `inline_default` and
        `normalize` still run that pass:

        * removing a parameter and re-adding one at the same index no
          longer leaves the removed parameter's argument in the calls;
        * an object outside the `_ArgumentChanger` hierarchy is
          refused with a `RefactoringError` instead of being accepted;
        * removing a non-existent index raises a `RefactoringError`
          instead of silently doing nothing;
        * inlining the default of a non-existent index raises a
          `RefactoringError` instead of an `IndexError`.

        """
        # imported here: change_signature_arch builds on this module
        from rope.refactor import arch, change_signature_arch

        refactoring = change_signature_arch.ChangeSignatureRefactoring(
            self.project,
            self.resource,
            self.offset,
            changers=changers,
            in_hierarchy=in_hierarchy,
            resources=resources,
            task_handle=task_handle,
        )
        driver = arch.RefactoringDriver(refactoring, policy=arch.LEGACY)
        return driver.run().changes


class _FunctionChangers:
    def __init__(self, pyfunction, definition_info, changers=None):
        self.pyfunction = pyfunction
        self.definition_info = definition_info
        self.changers = changers
        self.changed_definition_infos = self._get_changed_definition_infos()

    def _get_changed_definition_infos(self):
        definition_info = self.definition_info
        result = [definition_info]
        for changer in self.changers:
            definition_info = copy.deepcopy(definition_info)
            changer.change_definition_info(definition_info)
            result.append(definition_info)
        return result

    def change_definition(self, call):
        return self.changed_definition_infos[-1].to_string()

    def change_call(self, primary, pyname, call):
        call_info = functionutils.CallInfo.read(
            primary, pyname, self.definition_info, call
        )
        mapping = functionutils.ArgumentMapping(self.definition_info, call_info)

        for definition_info, changer in zip(
            self.changed_definition_infos, self.changers
        ):
            changer.change_argument_mapping(definition_info, mapping)

        return mapping.to_call_info(self.changed_definition_infos[-1]).to_string()


class _ArgumentChanger(arch.Transformation):
    """An elementary signature edit, executable on its own.

    Besides the two edit functions, a changer states the conditions
    under which it can be applied, and constructs its own `ChangeSet`
    -- it is the elementary transformation a composite signature
    change is built from.  What such an edit can *break* is a
    commitment of the refactoring wrapping it, not of the edit itself,
    so a caller composes plain changers for the behavior-agnostic
    level and decorated ones where it wants the commitment.

    A changer is constructed without execution context, as rope's
    public API has always allowed; the composite supplies that context
    through `configure()` before running it.
    """

    def change_definition_info(self, definition_info):
        pass

    def change_argument_mapping(self, definition_info, argument_mapping):
        pass

    def configure(self, project, resource, offset, resources, in_hierarchy, pending):
        """Late configuration: the composite supplies the context."""
        self.project = project
        self.resource = resource
        self.offset = offset
        self.resources = resources
        self.in_hierarchy = in_hierarchy
        self.pending = pending
        self.changes = None
        self.definition_info = None
        self.call_records = []
        self.unsure_occurrences = []

    def prepare_for_execution(self):
        (self.target_name, self.primary, self.pyname, self.others) = (
            _resolve_signature_target(
                self.project, self.resource, self.offset, pending=self.pending
            )
        )
        if (
            self.pyname is None
            or self.pyname.get_object() is None
            or not isinstance(self.pyname.get_object(), pyobjects.PyFunction)
        ):
            raise rope.base.exceptions.RefactoringError(
                "Change method signature should be performed on functions"
            )
        self.pyfunction = self.pyname.get_object()
        self.definition_info = functionutils.DefinitionInfo.read(self.pyfunction)

    def is_method(self):
        return isinstance(self.pyfunction.parent, pyobjects.PyClass)

    def applicability_preconditions(self):
        """Preconditions for constructing a structurally valid edit."""
        return []

    def private_transform(self):
        finder = self._finder()
        changers = _FunctionChangers(
            self.pyfunction, self.definition_info, [self]
        )
        changes = ChangeSet("Changing signature of <%s>" % self.target_name)
        for file_ in self.resources:
            pymodule = self.pending.pymodule(file_)
            new_content = self._rewrite(finder, pymodule, changers)
            if new_content is not None and new_content != pymodule.source_code:
                changes.add_change(ChangeContents(file_, new_content))
        self.changes = changes
        return changes

    def _finder(self):
        finder = occurrences.create_finder(
            self.project,
            self.target_name,
            self.pyname,
            instance=self.primary,
            in_hierarchy=self.in_hierarchy and self.is_method(),
            unsure=self._record_unsure,
        )
        if self.others:
            name, pyname = self.others
            constructor_finder = occurrences.create_finder(
                self.project, name, pyname, only_calls=True
            )
            finder = _MultipleFinders([finder, constructor_finder])
        return finder

    def _record_unsure(self, occurrence):
        self.unsure_occurrences.append(occurrence)
        return False

    def _rewrite(self, finder, pymodule, changers):
        """`_ChangeCallsInModule.get_changed_module`, over the pending view."""
        source = pymodule.source_code
        word_finder = worder.Worder(source)
        collector = codeanalyze.ChangeCollector(source)
        for occurrence in finder.find_occurrences(pymodule=pymodule):
            if not occurrence.is_called() and not occurrence.is_defined():
                continue
            start, end = occurrence.get_primary_range()
            begin_parens, end_parens = word_finder.get_word_parens_range(end - 1)
            call = source[start:end_parens]
            if occurrence.is_called():
                primary, pyname = occurrence.get_primary_and_pyname()
                self.call_records.append(
                    _CallRecord(
                        occurrence.resource, occurrence.lineno, primary, pyname, call
                    )
                )
                changed = changers.change_call(primary, pyname, call)
            else:
                changed = changers.change_definition(call)
            if changed is not None:
                collector.add_change(start, end_parens, changed)
        return collector.get_changed()


class _CallRecord:
    """A violator: a call site matched by a changer's own pass."""

    def __init__(self, resource, lineno, primary, pyname, code):
        self.resource = resource
        self.lineno = lineno
        self.primary = primary
        self.pyname = pyname
        self.code = code


class ArgumentNormalizer(_ArgumentChanger):
    pass


class ArgumentRemover(_ArgumentChanger):
    def __init__(self, index):
        self.index = index

    def change_definition_info(self, call_info):
        if self.index < len(call_info.args_with_defaults):
            del call_info.args_with_defaults[self.index]
        elif (
            self.index == len(call_info.args_with_defaults)
            and call_info.args_arg is not None
        ):
            call_info.args_arg = None
        elif (
            self.index == len(call_info.args_with_defaults)
            and call_info.args_arg is None
            and call_info.keywords_arg is not None
        ) or (
            self.index == len(call_info.args_with_defaults) + 1
            and call_info.args_arg is not None
            and call_info.keywords_arg is not None
        ):
            call_info.keywords_arg = None

    def change_argument_mapping(self, definition_info, mapping):
        if self.index < len(definition_info.args_with_defaults):
            name = definition_info.args_with_defaults[0]
            if name in mapping.param_dict:
                del mapping.param_dict[name]

    def applicability_preconditions(self):
        from rope.refactor import change_signature_arch as conditions

        return [
            conditions.ParameterExistsCondition(self.definition_info, self.index)
        ]


class ArgumentAdder(_ArgumentChanger):
    def __init__(self, index, name, default=None, value=None):
        self.index = index
        self.name = name
        self.default = default
        self.value = value

    def change_definition_info(self, definition_info):
        for pair in definition_info.args_with_defaults:
            if pair[0] == self.name:
                raise rope.base.exceptions.RefactoringError(
                    "Adding duplicate parameter: <%s>." % self.name
                )
        definition_info.args_with_defaults.insert(self.index, (self.name, self.default))

    def change_argument_mapping(self, definition_info, mapping):
        if self.value is not None:
            mapping.param_dict[self.name] = self.value

    def applicability_preconditions(self):
        from rope.refactor import change_signature_arch as conditions

        return [
            arch.ValidNameCondition(self.name),
            conditions.NoDuplicateParameterCondition(
                self.definition_info, self.name
            ),
        ]


class ArgumentDefaultInliner(_ArgumentChanger):
    def __init__(self, index):
        self.index = index
        self.remove = False

    def change_definition_info(self, definition_info):
        if self.remove:
            definition_info.args_with_defaults[self.index] = (
                definition_info.args_with_defaults[self.index][0],
                None,
            )

    def change_argument_mapping(self, definition_info, mapping):
        default = definition_info.args_with_defaults[self.index][1]
        name = definition_info.args_with_defaults[self.index][0]
        if default is not None and name not in mapping.param_dict:
            mapping.param_dict[name] = default

    def applicability_preconditions(self):
        from rope.refactor import change_signature_arch as conditions

        return [
            conditions.ParameterIndexInRangeCondition(
                self.definition_info, self.index
            )
        ]


class ArgumentReorderer(_ArgumentChanger):
    def __init__(self, new_order, autodef=None):
        """Construct an `ArgumentReorderer`

        Note that the `new_order` is a list containing the new
        position of parameters; not the position each parameter
        is going to be moved to. (changed in ``0.5m4``)

        For example changing ``f(a, b, c)`` to ``f(c, a, b)``
        requires passing ``[2, 0, 1]`` and *not* ``[1, 2, 0]``.

        The `autodef` (automatic default) argument, forces rope to use
        it as a default if a default is needed after the change.  That
        happens when an argument without default is moved after
        another that has a default value.  Note that `autodef` should
        be a string or `None`; the latter disables adding automatic
        default.

        """
        self.new_order = new_order
        self.autodef = autodef

    def change_definition_info(self, definition_info):
        new_args = list(definition_info.args_with_defaults)
        for new_index, index in enumerate(self.new_order):
            new_args[new_index] = definition_info.args_with_defaults[index]
        seen_default = False
        for index, (arg, default) in enumerate(list(new_args)):
            if default is not None:
                seen_default = True
            if seen_default and default is None and self.autodef is not None:
                new_args[index] = (arg, self.autodef)
        definition_info.args_with_defaults = new_args

    def applicability_preconditions(self):
        from rope.refactor import change_signature_arch as conditions

        return [
            conditions.ReorderIndicesValidCondition(
                self.definition_info, self.new_order
            )
        ]


class _ChangeCallsInModule:
    def __init__(self, project, occurrence_finder, resource, call_changer):
        self.project = project
        self.occurrence_finder = occurrence_finder
        self.resource = resource
        self.call_changer = call_changer

    def get_changed_module(self):
        word_finder = worder.Worder(self.source)
        change_collector = codeanalyze.ChangeCollector(self.source)
        for occurrence in self.occurrence_finder.find_occurrences(self.resource):
            if not occurrence.is_called() and not occurrence.is_defined():
                continue
            start, end = occurrence.get_primary_range()
            begin_parens, end_parens = word_finder.get_word_parens_range(end - 1)
            if occurrence.is_called():
                primary, pyname = occurrence.get_primary_and_pyname()
                changed_call = self.call_changer.change_call(
                    primary, pyname, self.source[start:end_parens]
                )
            else:
                changed_call = self.call_changer.change_definition(
                    self.source[start:end_parens]
                )
            if changed_call is not None:
                change_collector.add_change(start, end_parens, changed_call)
        return change_collector.get_changed()

    @property
    @utils.saveit
    def pymodule(self):
        return self.project.get_pymodule(self.resource)

    @property
    @utils.saveit
    def source(self):
        if self.resource is not None:
            return self.resource.read()
        else:
            return self.pymodule.source_code

    @property
    @utils.saveit
    def lines(self):
        return self.pymodule.lines


class _MultipleFinders:
    def __init__(self, finders):
        self.finders = finders

    def find_occurrences(self, resource=None, pymodule=None):
        all_occurrences = []
        for finder in self.finders:
            all_occurrences.extend(finder.find_occurrences(resource, pymodule))
        all_occurrences.sort(key=lambda x: x.get_primary_range())
        return all_occurrences
