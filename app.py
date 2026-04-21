"""
Local Flask backend for the wildlife pipeline UI.

Phase 1: folder-picker + subprocess runner. Runs on http://localhost:5050.
No hosting, no auth UI — uses the same service account as the CLI pipeline.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import FOLDER_ID, MAX_DOWNLOADS, SERVICE_ACCOUNT_FILE

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIMETYPE = "application/vnd.google-apps.folder"

RUNS_DIR = REPO_ROOT / "data" / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = REPO_ROOT / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)


_drive_client = None
_drive_lock = threading.Lock()


def drive():
    global _drive_client
    with _drive_lock:
        if _drive_client is None:
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            _drive_client = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _drive_client


_folder_cache: dict[str, list[dict[str, Any]]] = {}
_folder_meta_cache: dict[str, dict[str, Any]] = {}


def list_children(parent_id: str, folders_only: bool = True) -> list[dict[str, Any]]:
    cache_key = f"{parent_id}|{folders_only}"
    if cache_key in _folder_cache:
        return _folder_cache[cache_key]

    items: list[dict[str, Any]] = []
    page_token = None
    q = f"'{parent_id}' in parents and trashed = false"
    if folders_only:
        q += f" and mimeType = '{FOLDER_MIMETYPE}'"

    while True:
        resp = drive().files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="drive",
            driveId=FOLDER_ID,
            orderBy="name",
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    _folder_cache[cache_key] = items
    return items


def get_folder_meta(folder_id: str) -> dict[str, Any]:
    if folder_id in _folder_meta_cache:
        return _folder_meta_cache[folder_id]
    meta = drive().files().get(
        fileId=folder_id,
        fields="id, name, mimeType, parents",
        supportsAllDrives=True,
    ).execute()
    _folder_meta_cache[folder_id] = meta
    return meta


_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


SNAPSHOT_FILES = [
    "manifest.csv",
    "manifest_new.csv",
    "metadata.csv",
    "ml_outputs.csv",
    "speciesnet_results.csv",
    "speciesnet_results.json",
    "speciesnet_review.csv",
]


def _snapshot_outputs(run_dir: Path) -> dict[str, Any]:
    """Copy key shared outputs into the run dir so each run has its own archive."""
    outputs_dir = REPO_ROOT / "data" / "outputs"
    by_location = outputs_dir / "by_location"

    summary: dict[str, Any] = {"files": [], "by_location": []}

    for name in SNAPSHOT_FILES:
        src = outputs_dir / name
        if src.exists():
            dst = run_dir / name
            try:
                shutil.copy2(src, dst)
                summary["files"].append({"name": name, "size": dst.stat().st_size})
            except Exception as e:
                summary["files"].append({"name": name, "error": repr(e)})

    if by_location.exists():
        run_by_loc = run_dir / "by_location"
        run_by_loc.mkdir(exist_ok=True)
        for src in by_location.glob("*.csv"):
            dst = run_by_loc / src.name
            try:
                shutil.copy2(src, dst)
                summary["by_location"].append({"name": src.name, "size": dst.stat().st_size})
            except Exception as e:
                summary["by_location"].append({"name": src.name, "error": repr(e)})

    return summary


def _job_worker(job_id: str, cmd: list[str], run_dir: Path) -> None:
    job = _jobs[job_id]
    log_path = run_dir / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # for clean cancel via SIGTERM to the group
        )
        job["pid"] = proc.pid
        job["proc"] = proc
        job["status"] = "running"

        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            with _jobs_lock:
                job["log"].append(line.rstrip("\n"))
                for q_ in list(job["subscribers"]):
                    try:
                        q_.put_nowait(line)
                    except queue.Full:
                        pass

        proc.wait()
        job["returncode"] = proc.returncode

        if job.get("cancel_requested"):
            final_status = "cancelled"
        elif proc.returncode == 0:
            final_status = "success"
        else:
            final_status = "failed"
        job["status"] = final_status
        job["ended_at"] = time.time()
        job.pop("proc", None)

        try:
            snap = _snapshot_outputs(run_dir)
        except Exception as e:
            snap = {"error": repr(e)}
        job["snapshot"] = snap

        summary = {
            "job_id": job_id,
            "cmd": cmd,
            "status": final_status,
            "returncode": proc.returncode,
            "started_at": job.get("started_at"),
            "ended_at": job.get("ended_at"),
            "folders": job.get("folders"),
            "max_files": job.get("max_files"),
            "upload": job.get("upload"),
            "overwrite": job.get("overwrite"),
            "snapshot": snap,
        }
        try:
            (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        with _jobs_lock:
            for q_ in list(job["subscribers"]):
                try:
                    q_.put_nowait(None)
                except queue.Full:
                    pass


def _cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job:
        return False
    proc = job.get("proc")
    if not proc:
        return False
    job["cancel_requested"] = True
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            return False
    return True


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "drive_root": FOLDER_ID, "max_downloads_default": MAX_DOWNLOADS})


@app.get("/api/folders")
def api_folders():
    parent = request.args.get("parent") or FOLDER_ID
    try:
        children = list_children(parent, folders_only=True)
    except HttpError as e:
        return jsonify({"error": f"Drive API error: {e}"}), 502

    parent_meta: dict[str, Any] = {}
    try:
        parent_meta = get_folder_meta(parent)
    except HttpError:
        pass

    return jsonify({
        "parent": {
            "id": parent,
            "name": parent_meta.get("name", "Wildlife Camera Photo Database"),
        },
        "folders": [
            {
                "id": c["id"],
                "name": c["name"],
                "modifiedTime": c.get("modifiedTime"),
            }
            for c in children
        ],
    })


@app.get("/api/folders/count")
def api_folders_count():
    """Quick recursive image count so the UI can show 'this will process ~N images.'

    Caps the walk at `limit` files for responsiveness.
    """
    parent = request.args.get("parent")
    if not parent:
        return jsonify({"error": "parent is required"}), 400
    try:
        limit = int(request.args.get("limit", "5000"))
    except ValueError:
        limit = 5000

    stack = [parent]
    seen: set[str] = set()
    image_count = 0
    truncated = False

    try:
        while stack and image_count < limit:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)

            page_token = None
            while True:
                resp = drive().files().list(
                    q=f"'{current}' in parents and trashed = false",
                    fields="nextPageToken, files(id, mimeType)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive",
                    driveId=FOLDER_ID,
                ).execute()
                for item in resp.get("files", []):
                    mime = item.get("mimeType", "")
                    if mime == FOLDER_MIMETYPE:
                        stack.append(item["id"])
                    elif mime.startswith("image/"):
                        image_count += 1
                        if image_count >= limit:
                            truncated = True
                            break
                page_token = resp.get("nextPageToken")
                if not page_token or image_count >= limit:
                    break
    except HttpError as e:
        return jsonify({"error": f"Drive API error: {e}"}), 502

    return jsonify({"image_count": image_count, "truncated": truncated, "limit": limit})


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True, silent=True) or {}
    folders = body.get("folders") or []
    if not isinstance(folders, list) or not folders:
        return jsonify({"error": "folders (list of Drive folder IDs) is required"}), 400

    max_files = body.get("max_files")
    upload = bool(body.get("upload"))
    overwrite = bool(body.get("overwrite"))

    job_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_index = run_dir / "drive_index.csv"

    cmd: list[str] = [
        sys.executable,
        "scripts/pipeline/run_pipeline.py",
        "--start_folders", ",".join(str(f) for f in folders),
        "--out_index", str(out_index),
    ]
    if max_files is not None:
        try:
            mf = int(max_files)
            cmd += ["--max_files", str(mf), "--max_downloads", str(mf)]
        except (TypeError, ValueError):
            pass
    if upload:
        cmd.append("--upload")
    if overwrite:
        cmd.append("--overwrite")

    job = {
        "id": job_id,
        "cmd": cmd,
        "run_dir": str(run_dir),
        "status": "starting",
        "started_at": time.time(),
        "ended_at": None,
        "returncode": None,
        "pid": None,
        "log": [],
        "subscribers": set(),
        "folders": folders,
        "max_files": max_files,
        "upload": upload,
        "overwrite": overwrite,
    }
    with _jobs_lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_job_worker, args=(job_id, cmd, run_dir), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "starting", "cmd": cmd, "run_dir": str(run_dir)})


_PRIVATE_JOB_KEYS = {"log", "subscribers", "proc"}


def _job_public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in _PRIVATE_JOB_KEYS}


@app.get("/api/jobs")
def api_jobs():
    with _jobs_lock:
        return jsonify({"jobs": [_job_public(j) for j in _jobs.values()]})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(_job_public(job))


@app.get("/api/jobs/<job_id>/log")
def api_job_log(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    mode = request.args.get("mode", "snapshot")
    if mode == "snapshot":
        return jsonify({"status": job["status"], "log": job["log"]})

    q_: queue.Queue = queue.Queue(maxsize=10_000)
    with _jobs_lock:
        for line in job["log"]:
            q_.put_nowait(line + "\n")
        if job["status"] in ("running", "starting"):
            job["subscribers"].add(q_)
        else:
            q_.put_nowait(None)

    def stream():
        try:
            while True:
                try:
                    item = q_.get(timeout=15)
                except queue.Empty:
                    # Keepalive keeps proxies + browsers from closing the stream during the
                    # long SpeciesNet step (minutes with no stdout).
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    yield f"event: end\ndata: {json.dumps({'status': _jobs[job_id]['status']})}\n\n"
                    return
                yield f"data: {json.dumps({'line': item.rstrip()})}\n\n"
        finally:
            with _jobs_lock:
                job["subscribers"].discard(q_)

    return Response(stream(), mimetype="text/event-stream")


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    ok = _cancel_job(job_id)
    if not ok:
        return jsonify({"error": "job not running or unknown"}), 404
    return jsonify({"ok": True, "status": "cancelling"})


@app.get("/api/jobs/<job_id>/result")
def api_job_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    run_dir = Path(job["run_dir"])
    snapshot_by_loc = run_dir / "by_location"
    source_by_loc = REPO_ROOT / "data" / "outputs" / "by_location"
    by_loc_dir = snapshot_by_loc if snapshot_by_loc.exists() else source_by_loc

    files: list[dict[str, Any]] = []
    if by_loc_dir.exists():
        for p in sorted(by_loc_dir.glob("*.csv")):
            try:
                rel = str(p.resolve().relative_to(REPO_ROOT.resolve()))
            except ValueError:
                rel = str(p)
            files.append({
                "name": p.name,
                "path": rel,
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })

    return jsonify({
        "status": job["status"],
        "returncode": job.get("returncode"),
        "run_dir": job["run_dir"],
        "snapshot": job.get("snapshot"),
        "by_location": files,
    })


@app.get("/api/download")
def api_download():
    rel = request.args.get("path", "")
    if not rel:
        return jsonify({"error": "path is required"}), 400
    target = (REPO_ROOT / rel).resolve()
    try:
        target.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "invalid path"}), 400
    if not target.exists() or not target.is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(target.parent, target.name, as_attachment=True)


@app.get("/")
def index():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return "<h1>Wildlife Pipeline</h1><p>static/index.html not found.</p>", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
