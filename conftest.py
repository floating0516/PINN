"""
pytest 配置文件 - 在测试输出中显示中文描述
"""
import pytest


def pytest_runtest_protocol(item, nextitem):
    """在每个测试运行时显示中文描述"""
    if item.obj.__doc__:
        doc = item.obj.__doc__.strip().split('\n')[0]
        # print(f"\n{item.nodeid}")
        print(f"  {doc}")
    return None  # 继续正常的测试流程
