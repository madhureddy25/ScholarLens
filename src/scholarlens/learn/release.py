"""Publish improved models as GitHub Releases and install the latest one.

Releases are tagged ``lens-vN`` (detector) and ``quill-vN`` (editor LoRA adapter).
Each carries a .tar.gz and its SHA-256 in the release notes; ``update`` refuses
an archive whose checksum does not match, and extracts only regular files inside
the target directory.

Security note: Lens models are scikit-learn objects saved with joblib, and loading
a joblib file can execute code. Only install releases from the official
repository (``$SCHOLARLENS_REPO``, default madhureddy25/ScholarLens).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib import request

from scholarlens.learn.feedback import home

KINDS = ("lens", "quill", "bench")
_INCLUDE = {
    "lens": ("fusion/", "learn_report.json", "report.md", "report.json"),
    "quill": ("adapter_config.json", "adapter_model.safetensors", "scholarlens_adapter.json", "learn_report.json",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"),
    "bench": ("bench.jsonl", "manifest.json"),
}


def repo_slug() -> str:
    return os.environ.get("SCHOLARLENS_REPO", "madhureddy25/ScholarLens")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(kind: str, src: str | Path, dest_dir: str | Path, tag: str) -> Path:
    src, dest = Path(src), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{tag}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src).as_posix()
            if p.is_file() and any(rel == inc or rel.startswith(inc) for inc in _INCLUDE[kind]):
                tar.add(p, arcname=rel)
    return archive


def next_tag(kind: str, existing: list[str]) -> str:
    nums = [int(m.group(1)) for t in existing if (m := re.fullmatch(rf"{kind}-v(\d+)", t))]
    return f"{kind}-v{max(nums, default=0) + 1}"


def release_notes(kind: str, src: Path, digest: str) -> str:
    lines = [f"Automated {kind} release from ScholarLens continuous learning.", "", f"sha256: {digest}", ""]
    rep = src / "learn_report.json"
    if rep.is_file():
        r = json.loads(rep.read_text(encoding="utf-8"))
        if kind == "lens" and r.get("candidate"):
            lines.append("| family | AUROC | TPR@1%FPR |")
            lines.append("|---|---|---|")
            for fam, row in r["candidate"]["attacks"].items():
                tpr = next((v for k, v in row.items() if k.startswith("tpr@")), float("nan"))
                lines.append(f"| {fam} | {row['auroc']:.4f} | {tpr:.3f} |")
            lines.append("")
            lines += [f"- false-positive rate on `{fam}`: {row['fpr']:.4f}" for fam, row in r["candidate"]["human_fpr"].items()]
            lines.append(f"- contributions used: {r.get('n_contributions', 0)}; hard examples: {r.get('n_hard_examples', 0)}")
        elif kind == "quill" and r.get("candidate"):
            c = r["candidate"]
            lines.append(f"- base model: {r.get('base_model')}")
            lines.append(f"- integrity rate {c['integrity_rate']:.3f} · mean reward {c['mean_reward']:.3f} · "
                         f"similarity {c.get('mean_similarity')}")
            lines.append(f"- contributions: {r.get('n_contributions', 0)}; DPO pairs: {r.get('n_dpo_pairs', 0)}")
    return "\n".join(lines)


def publish(kind: str, src: str | Path, *, force: bool = False, repo: str | None = None, runner=subprocess.run) -> str:
    if kind not in KINDS:
        raise ValueError(kind)
    src, repo = Path(src), repo or repo_slug()
    rep = src / "learn_report.json"
    if kind != "bench" and not force and (not rep.is_file() or not json.loads(rep.read_text(encoding="utf-8")).get("accepted")):
        raise ValueError("refusing to publish: the learn report does not mark this candidate as accepted")
    tags = runner(["gh", "release", "list", "--repo", repo, "--limit", "200", "--json", "tagName", "--jq", ".[].tagName"],
                  check=True, capture_output=True, text=True).stdout.split()
    tag = next_tag(kind, tags)
    with tempfile.TemporaryDirectory() as tmp:
        archive = package(kind, src, tmp, tag)
        notes = release_notes(kind, src, sha256(archive))
        runner(["gh", "release", "create", tag, str(archive), "--repo", repo, "--title", tag, "--notes", notes],
               check=True)
    return tag


def _api(url: str):
    req = request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "scholarlens"})
    with request.urlopen(req, timeout=60) as r:  # noqa: S310
        return json.loads(r.read())


def latest_release(kind: str, repo: str | None = None, fetch=_api) -> dict | None:
    releases = fetch(f"https://api.github.com/repos/{repo or repo_slug()}/releases?per_page=100")
    for rel in releases:  # newest first
        if re.fullmatch(rf"{kind}-v\d+", rel.get("tag_name", "")) and not rel.get("draft"):
            return rel
    return None


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not m.isfile() or not str(target).startswith(str(dest.resolve()) + os.sep):
                raise ValueError(f"unsafe archive member: {m.name}")
        tar.extractall(dest, filter="data")


def install(kind: str, release: dict, *, download=None) -> Path:
    asset = next(a for a in release["assets"] if a["name"].endswith(".tar.gz"))
    m = re.search(r"sha256:\s*([0-9a-f]{64})", release.get("body") or "")
    if not m:
        raise ValueError("release has no sha256 in its notes")
    root = home() / "models" / kind
    dest = root / release["tag_name"]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset["name"]
        if download is None:
            with request.urlopen(asset["browser_download_url"], timeout=600) as r, archive.open("wb") as f:  # noqa: S310
                shutil.copyfileobj(r, f)
        else:
            download(asset["browser_download_url"], archive)
        if sha256(archive) != m.group(1):
            raise ValueError("checksum mismatch: refusing to install")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        _safe_extract(archive, dest)
    (root / "current").write_text(release["tag_name"], encoding="utf-8")
    return dest


def installed(kind: str) -> Path | None:
    pointer = home() / "models" / kind / "current"
    if pointer.is_file():
        d = pointer.parent / pointer.read_text(encoding="utf-8").strip()
        return d if d.is_dir() else None
    return None
