"""Outputs of a Quill session: edited file, tracked-changes copy, HTML review, disclosure JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path

from scholarlens.edit.diff import add_latex_diff_preamble, diff, render_html, render_latex_tracked
from scholarlens.edit.disclosure import build_manifest
from scholarlens.edit.engine import EditSession, Status

_EXT = {"latex": ".tex", "markdown": ".md", "text": ".txt"}

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e1d9;--ins:#d7f0dd;--ins-fg:#14532d;
--del:#fbe0dd;--del-fg:#7f1d1d;--ok:#15803d;--warn:#b45309;--bad:#b91c1c;--chip:#f1efe9}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a19a;--card:#1f1f1c;--line:#34332e;
--ins:#173d24;--ins-fg:#bbf7d0;--del:#4a1d1a;--del-fg:#fecaca;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--chip:#2a2925}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:24px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px;color:var(--muted);margin-bottom:8px}
.chip{background:var(--chip);border-radius:999px;padding:1px 9px}.accepted{color:var(--ok)}.rejected,.error,.refused{color:var(--bad)}
.unchanged,.skipped{color:var(--muted)}.text{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-serif,Georgia,serif}
ins{background:var(--ins);color:var(--ins-fg);text-decoration:none;border-radius:3px}
del{background:var(--del);color:var(--del-fg);border-radius:3px}.note{font-size:13px;color:var(--warn);margin-top:6px}
blockquote{margin:24px 0 0;padding:12px 16px;border-left:3px solid var(--line);background:var(--card);border-radius:0 8px 8px 0}
"""


def render_review_html(session: EditSession, manifest: dict) -> str:
    s = manifest["summary"]
    title = html.escape(session.document.meta.get("title") or Path(str(manifest["input"]["path"] or "document")).name)
    stats = [("accepted", s["accepted"]), ("unchanged", s["unchanged"]), ("rejected", s["rejected"]),
             ("skipped", s["skipped"]), ("errors / refusals", s["error"] + s["refused"]),
             ("words changed", s["words_changed"])]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>Quill review</title><style>{_CSS}</style>"
             f"</head><body><main><h1>Quill review — {title}</h1>",
             f"<p class='sub'>{html.escape(session.config.mode)} mode · {html.escape(json.dumps(session.model))}"
             f" · {html.escape(session.finished_at)}</p><div class='grid'>"]
    parts += [f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for k, v in stats]
    parts.append("</div>")
    for e in session.edits:
        if e.status == Status.SKIPPED and e.reason in ("not selected", "section not selected"):
            continue
        meta = [f"<span class='chip'>{e.node_id}</span>", f"<span class='chip'>{html.escape(e.section or '—')}</span>",
                f"<b class='{e.status.value}'>{e.status.value}</b>"]
        if e.similarity is not None:
            meta.append(f"<span>similarity {e.similarity:.3f}</span>")
        if e.attempts:
            meta.append(f"<span>{len(e.attempts)} attempt{'s' if len(e.attempts) > 1 else ''}</span>")
        if e.status == Status.ACCEPTED and e.mask is not None and e.edited_masked is not None:
            body = render_html(diff(e.mask.masked, e.edited_masked), e.mask)
        else:
            body = html.escape(e.original)
        notes = ""
        if e.reason and e.status != Status.UNCHANGED:
            notes += f"<div class='note'>{html.escape(e.reason)}</div>"
        for w in e.warnings:
            notes += f"<div class='note'>⚠ {html.escape(w)}</div>"
        parts.append(f"<div class='card'><div class='meta'>{''.join(meta)}</div><div class='text'>{body}</div>{notes}</div>")
    parts.append(f"<blockquote><b>Draft disclosure statement</b> (adapt and confirm before use)<br>"
                 f"{html.escape(manifest['statement_draft'])}</blockquote></main></body></html>")
    return "".join(parts)


def write_outputs(session: EditSession, out_dir: str | Path, *, input_path: str | Path | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = session.document
    stem = Path(str(input_path or doc.meta.get("path") or "document")).stem
    files: dict[str, Path] = {}

    ext = _EXT.get(doc.source_format, ".md")
    edited_path = out / f"{stem}.edited{ext}"
    edited_path.write_text(session.output_text(), encoding="utf-8")
    files["edited"] = edited_path

    if doc.source_format == "latex":
        tracked = {e.node_id: render_latex_tracked(diff(e.mask.masked, e.edited_masked), e.mask)
                   for e in session.edits
                   if e.status == Status.ACCEPTED and e.mask is not None and e.edited_masked is not None}
        tracked_path = out / f"{stem}.tracked.tex"
        tracked_path.write_text(add_latex_diff_preamble(doc.reconstruct(tracked)), encoding="utf-8")
        files["tracked"] = tracked_path

    docx_result = None
    if doc.source_format == "docx" and (input_path or doc.meta.get("path")):
        from scholarlens.edit.docx_writer import write_docx

        src = input_path or doc.meta["path"]
        files["edited_text"] = edited_path
        files["edited"] = out / f"{stem}.edited.docx"
        files["tracked"] = out / f"{stem}.tracked.docx"
        write_docx(session, src, files["edited"], tracked=False)
        docx_result = write_docx(session, src, files["tracked"], tracked=True)
        edited_path = files["edited"]

    manifest = build_manifest(session, input_path=input_path, output_path=edited_path)
    if docx_result is not None:
        manifest["docx"] = docx_result
    manifest_path = out / f"{stem}.disclosure.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files["disclosure"] = manifest_path

    from scholarlens.edit.session_io import save_session

    files["session"] = save_session(session, out / f"{stem}.session.json", input_path=input_path)

    review_path = out / f"{stem}.review.html"
    review_path.write_text(render_review_html(session, manifest), encoding="utf-8")
    files["review"] = review_path
    return files
