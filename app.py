import asyncio
import atexit
import json
import os
import queue
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv, set_key
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_OUTPUT_DIR_KEY = "DEFAULT_OUTPUT_DIR"
SUPPORTED_SUFFIXES = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

load_dotenv(ENV_PATH)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

executor = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Cleanup daemon — removes jobs older than 1 hour
# ---------------------------------------------------------------------------

def _cleanup_daemon():
    while True:
        time.sleep(3600)
        cutoff = time.time() - 3600
        with jobs_lock:
            stale = [jid for jid, j in jobs.items() if j.get("finished_at", float("inf")) < cutoff]
            for jid in stale:
                del jobs[jid]


threading.Thread(target=_cleanup_daemon, daemon=True).start()


# ---------------------------------------------------------------------------
# Completion notification
# ---------------------------------------------------------------------------

# terminal-notifier processes still waiting for a click; ended when the app quits
_notifier_procs: list = []
atexit.register(lambda: [p.terminate() for p in list(_notifier_procs)])


def _notify_complete(output_dir: str, filename: str) -> None:
    """Show a macOS notification; 'Open Folder' (or clicking it) opens output_dir."""
    # Apps launched from Finder may not have Homebrew on PATH
    search_path = os.pathsep.join([os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])
    notifier = shutil.which("terminal-notifier", path=search_path)
    if not notifier:
        print("terminal-notifier not found; skipping notification (brew install terminal-notifier)")
        return

    proc = subprocess.Popen(
        [notifier, "-title", "Transcription complete", "-message", filename,
         "-sound", "default", "-action", "Open Folder,Dismiss"],
        stdout=subprocess.PIPE, text=True,
    )
    _notifier_procs.append(proc)
    choice = proc.communicate()[0].strip()  # blocks until the user responds
    _notifier_procs.remove(proc)

    if choice in ("Open Folder", "@ACTIONCLICKED"):
        subprocess.run(["open", output_dir])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_transcription_job(job_id: str, audio_path: str, output_dir: str, quality: int, diarize: bool, save_txt: bool, save_json: bool, original_filename: str, delete_audio: bool):
    from transcriber import TranscriptionCancelled, transcribe_mp3

    job = jobs[job_id]
    job["status"] = "running"

    def progress_callback(step: str, status: str, fraction: Optional[float]) -> None:
        # Stop at the next checkpoint; never once saving has begun, so no partial files
        if job["cancel_requested"] and status != "done" and step != "save":
            raise TranscriptionCancelled()
        job["queue"].put({"type": "step", "step": step, "status": status,
                          "fraction": fraction, "ts": time.time()})

    try:
        text_content, txt_path, json_path = transcribe_mp3(
            file_path=audio_path,
            output_dir=output_dir,
            convert_quality=quality,
            diarize=diarize,
            progress_callback=progress_callback,
            save_txt=save_txt,
            save_json=save_json,
            original_filename=original_filename,
        )

        job["text"] = text_content
        job["txt_path"] = txt_path
        job["json_path"] = json_path
        job["status"] = "done"
        job["queue"].put({"type": "done", "message": "Transcription complete.", "percent": 100})
        threading.Thread(
            target=_notify_complete, args=(output_dir, original_filename), daemon=True
        ).start()

    except TranscriptionCancelled:
        job["status"] = "cancelled"
        job["queue"].put({"type": "cancelled", "ts": time.time()})

    except Exception as e:
        error_msg = str(e)
        # Surface ffmpeg missing error more clearly
        if "ffmpeg" in error_msg.lower() and "No such file" in error_msg:
            error_msg = (
                "ffmpeg is required for diarization of non-WAV files. "
                "Install with: brew install ffmpeg"
            )
        job["status"] = "error"
        job["error"] = error_msg
        job["queue"].put({"type": "error", "message": error_msg, "percent": 0})

    finally:
        job["finished_at"] = time.time()
        if delete_audio:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        job["queue"].put(None)  # sentinel — always last


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    source_path: str = Form(""),
    output_dir: str = Form(""),
    quality: int = Form(2),
    diarize: str = Form("false"),
    save_txt: str = Form("true"),
    save_json: str = Form("true"),
):
    # Either an uploaded file or a local path picked in the native window
    if source_path:
        if not Path(source_path).is_file():
            raise HTTPException(status_code=400, detail=f"File not found: {source_path}")
        filename = Path(source_path).name
    elif file is not None:
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    if source_path:
        audio_path = source_path
    else:
        # Save upload to a temp file using streaming chunks (handles large files)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                tmp.write(chunk)
        finally:
            tmp.close()
        audio_path = tmp.name
    is_temp = not source_path

    # Resolve output directory
    resolved_output_dir = output_dir.strip()
    if not resolved_output_dir:
        load_dotenv(ENV_PATH, override=True)
        resolved_output_dir = os.environ.get(DEFAULT_OUTPUT_DIR_KEY, "").strip()
    if not resolved_output_dir:
        resolved_output_dir = str(Path(audio_path).parent)

    # Pre-validate output directory
    try:
        Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if is_temp:
            os.unlink(audio_path)
        raise HTTPException(status_code=400, detail=f"Cannot create output directory: {e}")

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "queue": q,
            "txt_path": None,
            "json_path": None,
            "text": None,
            "error": None,
            "temp_file": audio_path if is_temp else None,
            "cancel_requested": False,
            "finished_at": None,
        }

    diarize_bool   = diarize.lower()   == "true"
    save_txt_bool  = save_txt.lower()  == "true"
    save_json_bool = save_json.lower() == "true"

    # Must save at least one format
    if not save_txt_bool and not save_json_bool:
        save_txt_bool = True

    executor.submit(_run_transcription_job, job_id, audio_path, resolved_output_dir, quality, diarize_bool, save_txt_bool, save_json_bool, filename, is_temp)

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        q = jobs[job_id]["queue"]
        while True:
            try:
                event = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            if event is None:
                yield "event: close\ndata: {}\n\n"
                break

            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel_requested"] = True
    return JSONResponse({"ok": True})


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("queued", "running"):
        raise HTTPException(status_code=202, detail="Job still running")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    return JSONResponse({
        "text": job["text"],
        "txt_path": job["txt_path"],
        "json_path": job["json_path"],
    })


def _mask_token(token: str) -> str:
    if not token or len(token) < 8:
        return ""
    return token[:4] + "****" + token[-4:]


@app.get("/settings")
async def get_settings():
    env_vals = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    raw_token = env_vals.get("HF_TOKEN", "")
    return JSONResponse({
        "hf_token_masked": _mask_token(raw_token),
        "hf_token_set": bool(raw_token),
        "default_output_dir": env_vals.get(DEFAULT_OUTPUT_DIR_KEY, ""),
    })


@app.post("/settings")
async def save_settings(request: Request):
    body = await request.json()
    hf_token = body.get("hf_token", "").strip()
    default_output_dir = body.get("default_output_dir", "").strip()

    ENV_PATH.touch(exist_ok=True)

    if hf_token:
        set_key(str(ENV_PATH), "HF_TOKEN", hf_token)
    set_key(str(ENV_PATH), DEFAULT_OUTPUT_DIR_KEY, default_output_dir)

    load_dotenv(ENV_PATH, override=True)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _wait_for_server(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)


class _JsApi:
    """Methods exposed to the page as window.pywebview.api.*"""

    def browse_folder(self, start_dir: str = "") -> str:
        import webview
        result = webview.windows[0].create_file_dialog(
            webview.FileDialog.FOLDER, directory=start_dir
        )
        return result[0] if result else ""

    def choose_audio_file(self) -> str:
        import webview
        types = "Audio files (" + ";".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES)) + ")"
        result = webview.windows[0].create_file_dialog(
            webview.FileDialog.OPEN, file_types=(types,)
        )
        return result[0] if result else ""


def _on_drop(window, event: dict) -> None:
    """Set the output directory to the dropped file's source folder."""
    files = event["dataTransfer"].get("files", [])
    path = files[0].get("pywebviewFullPath") if files else None
    if path:
        folder = json.dumps(os.path.dirname(path))
        window.evaluate_js(f"document.getElementById('outputDir').value = {folder}")


def _bind_drop_handler(window) -> None:
    # A Python-side drop listener is what makes pywebview attach full file paths
    window.dom.get_element("#dropZone").events.drop += lambda e: _on_drop(window, e)


def _find_free_port(start: int = 18001, end: int = 18998) -> int:
    candidates = list(range(start, end + 1))
    random.shuffle(candidates)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{end}")


if __name__ == "__main__":
    import uvicorn
    import webview
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Starting server on {url}")
    threading.Thread(
        target=uvicorn.run,
        args=("app:app",),
        kwargs={"host": "127.0.0.1", "port": port, "reload": False},
        daemon=True,
    ).start()
    _wait_for_server(port)
    window = webview.create_window("Transcriber", url, js_api=_JsApi(), width=900, height=900)
    window.events.loaded += _bind_drop_handler
    webview.start()
