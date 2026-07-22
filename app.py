import asyncio
import json
import os
import queue
import random
import socket
import tempfile
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
# Worker
# ---------------------------------------------------------------------------

def _run_transcription_job(job_id: str, audio_path: str, output_dir: str, quality: int, diarize: bool, save_txt: bool, save_json: bool, original_filename: str):
    from transcriber import transcribe_mp3

    job = jobs[job_id]
    job["status"] = "running"

    def progress_callback(message: str, percent: int):
        job["queue"].put({"type": "progress", "message": message, "percent": percent})

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


@app.get("/browse-folder")
async def browse_folder():
    """Open a native macOS folder picker and return the selected path."""
    import subprocess
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["osascript", "-e",
             'tell application "Finder" to set f to choose folder with prompt "Select Output Folder"\n'
             'return POSIX path of f'],
            capture_output=True, text=True,
        )
    )
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        return JSONResponse({"path": ""})
    return JSONResponse({"path": path})


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    output_dir: str = Form(""),
    quality: int = Form(2),
    diarize: str = Form("false"),
    save_txt: str = Form("true"),
    save_json: str = Form("true"),
):
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    # Save upload to a temp file using streaming chunks (handles large files)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            tmp.write(chunk)
    finally:
        tmp.close()

    # Resolve output directory
    resolved_output_dir = output_dir.strip()
    if not resolved_output_dir:
        load_dotenv(ENV_PATH, override=True)
        resolved_output_dir = os.environ.get(DEFAULT_OUTPUT_DIR_KEY, "").strip()
    if not resolved_output_dir:
        resolved_output_dir = str(Path(tmp.name).parent)

    # Pre-validate output directory
    try:
        Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        os.unlink(tmp.name)
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
            "temp_file": tmp.name,
            "finished_at": None,
        }

    diarize_bool   = diarize.lower()   == "true"
    save_txt_bool  = save_txt.lower()  == "true"
    save_json_bool = save_json.lower() == "true"

    # Must save at least one format
    if not save_txt_bool and not save_json_bool:
        save_txt_bool = True

    executor.submit(_run_transcription_job, job_id, tmp.name, resolved_output_dir, quality, diarize_bool, save_txt_bool, save_json_bool, file.filename)

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

def _open_browser_when_ready(url: str, timeout: float = 30.0):
    port = int(url.split(":")[-1])
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    webbrowser.open_new(url)


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
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Starting server on {url}")
    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
