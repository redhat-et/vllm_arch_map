"""Source inspection and conservative trace attribution for architecture reports."""

import ast
import inspect
import textwrap


def analyze_source(source, filename=None, start_line=1):
    """Describe Python statements without claiming they executed."""
    source = textwrap.dedent(source)
    function = ast.parse(source).body[0]
    operations = []

    def visit(statements, context):
        for statement in statements:
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if isinstance(statement, ast.If):
                condition = ast.unparse(statement.test)
                visit(statement.body, context + [f"if {condition}"])
                visit(statement.orelse, context + [f"else ({condition})"])
                continue
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                header = ast.unparse(statement).splitlines()[0]
                visit(statement.body, context + [header])
                visit(statement.orelse, context + [f"else {header}"])
                continue
            if isinstance(statement, (ast.With, ast.AsyncWith, ast.Try)):
                visit(statement.body, context + [type(statement).__name__])
                for handler in getattr(statement, "handlers", []):
                    visit(handler.body, context + ["exception handler"])
                visit(getattr(statement, "orelse", []), context + ["try else"])
                visit(getattr(statement, "finalbody", []), context + ["finally"])
                continue
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ):
                continue
            calls = [
                ast.unparse(n.func)
                for n in ast.walk(statement)
                if isinstance(n, ast.Call)
            ]
            reads = sorted(
                {
                    n.id
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Load)
                    and n.id != "self"
                }
            )
            writes = sorted(
                {
                    n.id
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
            )
            kind = "return" if isinstance(statement, ast.Return) else "statement"
            if calls:
                kind = "call"
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                kind = "assignment"
            operations.append(
                {
                    "id": f"s{len(operations)}",
                    "op": kind,
                    "line": ast.get_source_segment(source, statement)
                    or ast.unparse(statement),
                    "line_number": start_line + statement.lineno - 1,
                    "calls": calls,
                    "reads": reads,
                    "writes": writes,
                    "context": context,
                    "evidence": "source",
                }
            )

    visit(function.body, [])
    parameters = [arg.arg for arg in function.args.args if arg.arg != "self"]
    definitions = {name: {"input:" + name} for name in parameters}
    edges = []
    for operation in operations:
        for name in operation["reads"]:
            for producer in sorted(definitions.get(name, [])):
                edges.append(
                    {
                        "from": producer,
                        "to": operation["id"],
                        "variable": name,
                        "evidence": "possible_source_dependency",
                    }
                )
        for name in operation["writes"]:
            if operation["context"]:
                definitions.setdefault(name, set()).add(operation["id"])
            else:
                definitions[name] = {operation["id"]}
    return {
        "edges": edges,
        "source": source,
        "source_file": filename,
        "start_line": start_line,
        "evidence": "source",
        "operations": operations,
        "parameters": [arg.arg for arg in function.args.args if arg.arg != "self"],
        "limitations": [
            "Source order and variable references, not observed tensor edges.",
            (
                "Branches and loops have not been evaluated; dependencies are "
                "conservative, may include infeasible branches, and omit aliases "
                "and loop-carried dependencies."
            ),
        ],
    }


def inspect_method(method):
    try:
        lines, start = inspect.getsourcelines(method)
        return analyze_source("".join(lines), inspect.getsourcefile(method), start)
    except (TypeError, OSError, SyntaxError, AttributeError, IndexError):
        return {"evidence": "unavailable", "operations": []}


def module_marker(name):
    """Accept explicit module markers, including vLLM's literal dictionary format."""
    if name.startswith("Module:"):
        return name[7:].strip()
    try:
        value = ast.literal_eval(name)
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, dict) and isinstance(value.get("Module"), str):
        return value["Module"]
    return None


def correlate_trace(events):
    """Link kernels through CUDA launches; never use GPU/CPU time overlap."""
    launches = {}
    ranges = []
    stacks = {}
    kernels = []
    cpu_ops = []
    for event in sorted(events, key=lambda e: e.get("ts", 0)):
        cat = event.get("cat", "").lower()
        key = (event.get("pid"), event.get("tid"))
        name = module_marker(event.get("name", ""))
        if name and event.get("ph") == "X":
            ranges.append({**event, "module": name})
        elif name and event.get("ph") == "B":
            stacks.setdefault(key, []).append({**event, "module": name})
        elif event.get("ph") == "E" and stacks.get(key):
            begin = stacks[key].pop()
            ranges.append({**begin, "dur": event.get("ts", 0) - begin.get("ts", 0)})
        correlation = event.get("args", {}).get("correlation")
        if cat in ("cuda_runtime", "cuda_driver") and correlation is not None:
            launches.setdefault((event.get("pid"), correlation), []).append(event)
        if cat == "cpu_op" and event.get("ph") == "X":
            cpu_ops.append(event)
        if cat == "kernel" and event.get("ph") == "X":
            kernels.append(event)
    runtime_pids = {pid for pid, _ in launches}
    output = []
    for kernel in kernels:
        correlation = kernel.get("args", {}).get("correlation")
        args = kernel.get("args", {})
        process = args.get("process_id")
        if process is None and len(runtime_pids) == 1:
            process = next(iter(runtime_pids))
        candidates = launches.get((process, correlation), [])
        if len(candidates) != 1:
            output.append({"kernel": kernel, "evidence": "unresolved"})
            continue
        launch = candidates[0]
        owners = [
            r
            for r in ranges
            if r.get("pid") == launch.get("pid")
            and r.get("tid") == launch.get("tid")
            and r.get("ts", 0) <= launch.get("ts", 0)
            and launch.get("ts", 0) + launch.get("dur", 0)
            <= r.get("ts", 0) + r.get("dur", 0)
        ]
        owner = min(owners, key=lambda r: r["dur"]) if owners else None
        enclosing_ops = [
            op
            for op in cpu_ops
            if op.get("pid") == launch.get("pid")
            and op.get("tid") == launch.get("tid")
            and op.get("ts", 0) <= launch.get("ts", 0)
            and launch.get("ts", 0) + launch.get("dur", 0)
            <= op.get("ts", 0) + op.get("dur", 0)
        ]
        cpu_op = min(enclosing_ops, key=lambda op: op["dur"]) if enclosing_ops else None
        output.append(
            {
                "kernel": kernel,
                "launch": launch,
                "cpu_op": cpu_op,
                "module": owner["module"] if owner else None,
                "evidence": "launch_correlation" if owner else "unresolved",
            }
        )
    return output


def validate_identity(logs, tree):
    if not isinstance(tree, dict) or not tree.get("model"):
        raise ValueError("Model tree must contain a model identifier; recapture it.")
    if logs.get("model") != tree["model"]:
        raise ValueError(
            f"Model mismatch: logs={logs.get('model')!r}, tree={tree['model']!r}"
        )
    architectures = tree.get("config", {}).get("architectures", [])
    if (
        logs.get("architecture")
        and architectures
        and logs["architecture"] not in architectures
    ):
        raise ValueError("Architecture mismatch between logs and model tree")
