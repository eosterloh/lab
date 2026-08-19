from __future__ import annotations

from typing import Any, Callable

from lab.pack import ArtifactPack
from lab.types import ToolResult, err, ok

Args = dict[str, Any]
Handler = Callable[[Any, Args], ToolResult]


def read_notebook(sup: Any, args: Args) -> ToolResult:
    n = int(args.get("n", 20))
    return ok(entries=sup.notebook.tail(n), beliefs=sup.notebook.beliefs())


def write_note(sup: Any, args: Args) -> ToolResult:
    text = str(args.get("text") or args.get("verdict") or "").strip()
    if not text:
        return err("text is required")
    row = sup.notebook.append(
        cycle=sup.state.cycle,
        phase=sup.state.phase,
        kind=str(args.get("kind", "note")),
        pack_hash=sup.state.pack_hash,
        hypothesis=args.get("hypothesis"),
        metrics=args.get("metrics"),
        verdict=args.get("verdict"),
        text=text,
    )
    return ok(entry=row)


def write_beliefs(sup: Any, args: Args) -> ToolResult:
    text = str(args.get("text", "")).strip()
    if not text:
        return err("text is required")
    sup.notebook.set_beliefs(text)
    return ok(beliefs=text)


def halt(sup: Any, args: Args) -> ToolResult:
    reason = str(args.get("reason", "policy halt"))
    sup.halt(reason)
    return ok(halted=True, reason=reason)


def run_eval(sup: Any, args: Args) -> ToolResult:
    result = sup.run_eval()
    return ok(eval=result)


def read_metrics(sup: Any, args: Args) -> ToolResult:
    return ok(metrics=sup.state.last_metrics, job_id=sup.state.current_job_id)


def read_samples(sup: Any, args: Args) -> ToolResult:
    samples = (sup.state.last_eval or {}).get("samples")
    return ok(samples=samples or [])


def enter_research(sup: Any, args: Args) -> ToolResult:
    return sup.transition_research()


def list_files(sup: Any, args: Args) -> ToolResult:
    rel = str(args.get("path", "."))
    return ok(files=sup.sandbox.list_files(rel))


def read_file(sup: Any, args: Args) -> ToolResult:
    rel = str(args.get("path", ""))
    try:
        return ok(path=rel, content=sup.sandbox.read_file(rel))
    except Exception as e:
        return err(str(e))


def write_file(sup: Any, args: Args) -> ToolResult:
    rel = str(args.get("path", ""))
    content = str(args.get("content", ""))
    try:
        path = sup.sandbox.write_file(rel, content)
        return ok(path=path)
    except Exception as e:
        return err(str(e))


def exec_cmd(sup: Any, args: Args) -> ToolResult:
    argv = args.get("argv")
    if not isinstance(argv, list):
        return err("argv must be a list of strings")
    try:
        return ok(**sup.sandbox.exec([str(x) for x in argv]))
    except Exception as e:
        return err(str(e))


def web_fetch(sup: Any, args: Args) -> ToolResult:
    url = str(args.get("url", ""))
    dest = args.get("path")
    try:
        return ok(**sup.sandbox.fetch(url, dest=dest))
    except Exception as e:
        return err(str(e))


def web_search(sup: Any, args: Args) -> ToolResult:
    query = str(args.get("query", ""))
    try:
        return ok(hits=sup.sandbox.search(query))
    except Exception as e:
        return err(str(e))


def write_pack(sup: Any, args: Args) -> ToolResult:
    payload = args.get("pack") or args
    try:
        pack = ArtifactPack.from_dict(payload if "hypothesis" in payload else args)
        digest = sup.save_pack(pack)
        return ok(pack_hash=digest, pack=pack.canonical())
    except Exception as e:
        return err(str(e))


def enter_train(sup: Any, args: Args) -> ToolResult:
    return sup.transition_train()


def job_status(sup: Any, args: Args) -> ToolResult:
    job = sup.poll_job()
    if job is None:
        return err("no current job")
    return ok(job=job.to_dict())


def cancel_job(sup: Any, args: Args) -> ToolResult:
    job = sup.cancel_job()
    if job is None:
        return err("no current job")
    return ok(job=job.to_dict())


def enter_eval(sup: Any, args: Args) -> ToolResult:
    return sup.transition_eval()


REGISTRY: dict[str, Handler] = {
    "read_notebook": read_notebook,
    "write_note": write_note,
    "write_beliefs": write_beliefs,
    "halt": halt,
    "run_eval": run_eval,
    "read_metrics": read_metrics,
    "read_samples": read_samples,
    "enter_research": enter_research,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "exec": exec_cmd,
    "web_fetch": web_fetch,
    "web_search": web_search,
    "write_pack": write_pack,
    "enter_train": enter_train,
    "job_status": job_status,
    "cancel_job": cancel_job,
    "enter_eval": enter_eval,
}
