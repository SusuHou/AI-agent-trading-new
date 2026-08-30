"""Canonical package for the step-by-step replication modules.

逐步复现模块的统一 Python package。

New public integration boundaries should import shared receipt/checkpoint types
through ``steps.step_...`` so Python does not load those public classes twice
under two module names. The older Step 01-28 internal import cluster is being
migrated incrementally. / 新的公共整合边界应通过 ``steps.step_...`` 导入共享的
凭证与 checkpoint 类型，防止 Python 用两个模块名重复加载公共 class；旧的第
01-28 步内部导入簇会逐步迁移。
"""
