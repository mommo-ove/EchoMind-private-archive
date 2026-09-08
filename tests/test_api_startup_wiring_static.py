import ast
from pathlib import Path


def test_tool_manager_startup_call_only_passes_supported_constructor_keywords():
    root = Path(__file__).resolve().parents[1]
    api_tree = ast.parse((root / "api" / "main.py").read_text(encoding="utf-8"))
    manager_tree = ast.parse((root / "mcp" / "tool_manager.py").read_text(encoding="utf-8"))

    manager_class = next(
        node for node in manager_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MCPToolManager"
    )
    constructor = next(
        node for node in manager_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    supported = {argument.arg for argument in constructor.args.args if argument.arg != "self"}
    calls = [
        node for node in ast.walk(api_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MCPToolManager"
    ]

    assert len(calls) == 1
    passed = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    assert passed <= supported
    assert "embedder" not in passed
