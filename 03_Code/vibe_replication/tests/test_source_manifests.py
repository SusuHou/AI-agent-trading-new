"""Audit tests for the explicit formal-session source scopes.

正式 session 显式源码范围的审计测试。

These tests deliberately derive the Step-28 dependency closure from Python's
AST instead of copying the production list a second time. / 这些测试使用
Python AST 独立推导 Step 28 的依赖闭包，而不是再抄一遍生产清单。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import unittest
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import src.source_manifests as source_manifests


def _local_module_index() -> dict[str, str]:
    """Map every import spelling used by this project to one relative file.

    把本项目使用的每种本地 import 写法映射到一个相对文件。
    """

    index: dict[str, str] = {}
    for directory_name in ("src", "steps"):
        for path in (PROJECT_ROOT / directory_name).glob("*.py"):
            relative_name = path.relative_to(PROJECT_ROOT).as_posix()
            qualified_name = relative_name[:-3].replace("/", ".")
            index[qualified_name] = relative_name
            # The step files add ``steps`` to sys.path and therefore import
            # one another as ``step_25_...``. / step 文件把 ``steps``
            # 加入 sys.path，所以也会用 ``step_25_...`` 这种短名。
            if directory_name == "steps":
                index[path.stem] = relative_name
    return index


def _direct_local_imports(
    relative_name: str,
    module_index: dict[str, str],
) -> set[str]:
    """Read direct project-local imports from one file. / 读取一个文件的直接本地依赖。"""

    tree = ast.parse(
        (PROJECT_ROOT / relative_name).read_text(encoding="utf-8"),
        filename=relative_name,
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in module_index:
                    imported.add(module_index[alias.name])
        elif isinstance(node, ast.ImportFrom) and node.module in module_index:
            imported.add(module_index[node.module])
    return imported


def _local_import_closure(relative_name: str) -> set[str]:
    """Return the recursive local-import closure including its root.

    返回包含根文件本身的递归本地 import 闭包。
    """

    module_index = _local_module_index()
    pending = [relative_name]
    closure: set[str] = set()
    while pending:
        current = pending.pop()
        if current in closure:
            continue
        closure.add(current)
        pending.extend(_direct_local_imports(current, module_index))
    return closure


class SourceManifestTests(unittest.TestCase):
    """Freeze A23's scientific-code boundary. / 锁定 A23 的科学代码边界。"""

    def test_lists_have_exact_counts_are_sorted_disjoint_and_exist(self) -> None:
        """The two explicit lists must be small, deterministic, and real.

        两份明确清单必须规模固定、顺序确定，且文件真实存在。
        """

        execution = source_manifests.EXECUTION_SOURCE_FILES
        result = source_manifests.RESULT_PIPELINE_SOURCE_FILES

        self.assertEqual(len(execution), 27)
        self.assertEqual(len(result), 19)
        self.assertEqual(execution, tuple(sorted(execution)))
        self.assertEqual(result, tuple(sorted(result)))
        self.assertEqual(len(execution), len(set(execution)))
        self.assertEqual(len(result), len(set(result)))
        self.assertTrue(set(execution).isdisjoint(result))
        for relative_name in execution + result:
            with self.subTest(relative_name=relative_name):
                self.assertTrue((PROJECT_ROOT / relative_name).is_file())

    def test_execution_scope_is_exact_step28_local_import_closure(self) -> None:
        """No market dependency may be omitted or silently added.

        市场转移依赖既不得遗漏，也不得被静默添加。
        """

        derived = _local_import_closure("steps/step_28_session_phases.py")
        self.assertEqual(len(derived), 27)
        self.assertEqual(derived, set(source_manifests.EXECUTION_SOURCE_FILES))

    def test_combined_scope_covers_every_result_root_local_dependency(self) -> None:
        """Every result root is closed over local scientific dependencies.

        每个结果管线根文件的本地科学依赖都必须被覆盖。
        """

        covered = set(source_manifests.EXECUTION_SOURCE_FILES) | set(
            source_manifests.RESULT_PIPELINE_SOURCE_FILES
        )
        # The manifest hashes itself separately, so it is intentionally not
        # duplicated in either source list. / manifest 文件有独立哈希，
        # 因此不在两份源码清单中重复出现。
        covered.add(source_manifests.SOURCE_SCOPE_MANIFEST_FILE)
        for result_root in source_manifests.RESULT_PIPELINE_SOURCE_FILES:
            with self.subTest(result_root=result_root):
                self.assertLessEqual(_local_import_closure(result_root), covered)

    def test_unlisted_root_orchestration_file_does_not_change_source_hashes(self) -> None:
        """An unrelated future launcher must not stale scientific sessions.

        以后新增的无关调度入口不应让科学 session 无故过期。
        """

        before = (
            source_manifests.LOADED_EXECUTION_SOURCE_SHA256,
            source_manifests.LOADED_RESULT_PIPELINE_SOURCE_SHA256,
            source_manifests.LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
            source_manifests.LOADED_COMBINED_SESSION_SOURCE_SHA256,
        )
        relative_name = f"unrelated_orchestration_{uuid4().hex}.py"
        path = PROJECT_ROOT / relative_name
        try:
            path.write_text(
                '"""Unrelated test-only launcher. / 仅测试用的无关调度入口。"""\n',
                encoding="utf-8",
            )
            self.assertNotIn(relative_name, source_manifests.EXECUTION_SOURCE_FILES)
            self.assertNotIn(
                relative_name,
                source_manifests.RESULT_PIPELINE_SOURCE_FILES,
            )
            reloaded = importlib.reload(source_manifests)
            after = (
                reloaded.LOADED_EXECUTION_SOURCE_SHA256,
                reloaded.LOADED_RESULT_PIPELINE_SOURCE_SHA256,
                reloaded.LOADED_SOURCE_SCOPE_MANIFEST_SHA256,
                reloaded.LOADED_COMBINED_SESSION_SOURCE_SHA256,
            )
            self.assertEqual(after, before)
        finally:
            if path.exists():
                path.unlink()
            # Restore the module constants from the clean workspace too.
            # / 从已清理的 workspace 再恢复一次模块常量。
            importlib.reload(source_manifests)


if __name__ == "__main__":
    unittest.main()
