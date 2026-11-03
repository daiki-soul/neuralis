import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

import requests
from flask import Flask, Response, jsonify, send_from_directory, request

app = Flask(__name__)

HOST = "127.0.0.1"
PORT = 8000
APP_URL = f"http://{HOST}:{PORT}/"

OLLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT = (10, 600)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
MEMORY_MAX_ITEMS = 200
MEMORY_MAX_CHARS = 2000
_memory_lock = threading.Lock()


# ---------------------------------------------------------------- Ollama
def start_ollama():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        print("Ollama is already running.")
        return
    except requests.RequestException:
        pass

    print("Starting Ollama...")

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )

    for _ in range(60):
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).raise_for_status()
            print("Ollama is ready.")
            return
        except requests.RequestException:
            time.sleep(1)

    raise RuntimeError("Ollama failed to start within 60 seconds.")


# ---------------------------------------------------------------- Chrome
def find_chrome():
    candidates = []
    if sys.platform.startswith("win"):
        for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(env)
            if root:
                candidates.append(os.path.join(root, "Google", "Chrome", "Application", "chrome.exe"))
    elif sys.platform == "darwin":
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(found)

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def open_in_chrome(url=APP_URL):
    chrome = find_chrome()
    try:
        if chrome:
            webbrowser.register("chrome-local", None, webbrowser.BackgroundBrowser(chrome))
            webbrowser.get("chrome-local").open(url)
        else:
            print("Chrome not found, using the default browser.")
            webbrowser.open(url)
    except Exception as error:
        print(f"Could not open browser: {error}")


# ---------------------------------------------------------------- Memory (memory.json)
def _load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_memory(items):
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MEMORY_FILE)


def _clean(text):
    return " ".join(str(text or "").split())[:MEMORY_MAX_CHARS]


@app.get("/api/memory")
def memory_list():
    with _memory_lock:
        return jsonify({"items": _load_memory()})


@app.post("/api/memory")
def memory_add():
    text = _clean((request.get_json(silent=True) or {}).get("text"))
    if not text:
        return jsonify({"error": "Empty memory."}), 400
    with _memory_lock:
        items = _load_memory()
        added = not any(i.get("text", "").lower() == text.lower() for i in items)
        if added:
            items.append({"id": uuid.uuid4().hex[:12], "text": text, "created": int(time.time() * 1000)})
            items = items[-MEMORY_MAX_ITEMS:]
            _save_memory(items)
        return jsonify({"added": added, "items": items})


@app.put("/api/memory/<item_id>")
def memory_edit(item_id):
    text = _clean((request.get_json(silent=True) or {}).get("text"))
    if not text:
        return jsonify({"error": "Empty memory."}), 400
    with _memory_lock:
        items = _load_memory()
        for item in items:
            if item.get("id") == item_id:
                item["text"] = text
                _save_memory(items)
                return jsonify({"items": items})
    return jsonify({"error": "Memory not found."}), 404


@app.delete("/api/memory/<item_id>")
def memory_delete(item_id):
    with _memory_lock:
        items = [i for i in _load_memory() if i.get("id") != item_id]
        _save_memory(items)
        return jsonify({"items": items})


@app.delete("/api/memory")
def memory_clear():
    with _memory_lock:
        _save_memory([])
        return jsonify({"items": []})


# ---------------------------------------------------------------- Routes
@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/tags")
def models():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )
        return Response(
            response.content,
            status=response.status_code,
            content_type="application/json"
        )
    except requests.RequestException as error:
        return jsonify({"error": str(error)}), 502


@app.post("/api/chat")
def chat():
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=request.get_json(),
            stream=True,
            timeout=TIMEOUT
        )

        # Pass Ollama's own error body through (e.g. "model does not support thinking")
        if not response.ok:
            body = response.content
            status = response.status_code
            response.close()
            return Response(body, status=status, content_type="application/json")

        def generate():
            try:
                for line in response.iter_lines():
                    if line:
                        yield line + b"\n"
            finally:
                response.close()

        return Response(
            generate(),
            content_type="application/x-ndjson"
        )

    except requests.RequestException as error:
        return jsonify({"error": str(error)}), 502


if __name__ == "__main__":
    start_ollama()
    # Open Chrome shortly after the server starts listening
    threading.Timer(1.0, open_in_chrome).start()
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True
    )
