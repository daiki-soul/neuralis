"""
Local AI  -  terminal UI (OpenCode-style) for Ollama.

Run it from the same folder as app.py / index.html so it shares memory.json
with the web version.  Requires:  pip install rich prompt_toolkit requests
"""
import difflib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

# ------------------------------------------------------------------ config
OLLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT = (10, 600)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")      # shared with app.py
RULES_FILE = os.path.join(BASE_DIR, "rules.txt")         # optional: your own rules
CONFIG_FILE = os.path.join(BASE_DIR, "cli_config.json")
HISTORY_FILE = os.path.join(BASE_DIR, ".cli_history")
MEMORY_MAX_ITEMS = 200
MEMORY_MAX_CHARS = 2000
FILE_MAX_BYTES = 200_000

BLUE = "#4f9cf9"
GREEN = "#7fd18b"
MUTED = "#8a8a8a"

DEFAULT_RULES = (
    "Understand what the user actually wants before acting. If a request is missing "
    "something essential (topic, length, language, goal), ask one short question. "
    "If it is already clear, just do it. Never invent requirements."
)

console = Console(highlight=False)


# ------------------------------------------------------------------ ollama
def ollama_up():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except requests.RequestException:
        return False


def start_ollama():
    if ollama_up():
        return
    console.print(Text("Starting Ollama...", style=MUTED))
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        console.print("[red]Could not find the `ollama` command. Install Ollama first.[/red]")
        sys.exit(1)
    for _ in range(60):
        if ollama_up():
            return
        time.sleep(1)
    console.print("[red]Ollama failed to start within 60 seconds.[/red]")
    sys.exit(1)


def list_models():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def fmt_size(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


# ------------------------------------------------------------------ config / memory
def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


_mem_lock = threading.Lock()


def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_memory(items):
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MEMORY_FILE)


def clean(text):
    return " ".join(str(text or "").split())[:MEMORY_MAX_CHARS]


def memory_add(text):
    text = clean(text)
    if not text:
        return False
    with _mem_lock:
        items = load_memory()
        if any(i.get("text", "").lower() == text.lower() for i in items):
            return False
        items.append({"id": uuid.uuid4().hex[:12], "text": text, "created": int(time.time() * 1000)})
        save_memory(items[-MEMORY_MAX_ITEMS:])
    return True


def memory_delete(item_id):
    with _mem_lock:
        save_memory([i for i in load_memory() if i.get("id") != item_id])


def memory_clear():
    with _mem_lock:
        save_memory([])


REMEMBER_RE = re.compile(
    r"^\s*(?:(?:please|pls|hey|ok|okay|and|also)[,\s]+)*"
    r"(?:remember|don'?t forget|do not forget|keep in mind|make a note(?: that)?|note that|memori[sz]e)"
    r"\b[\s:,\-]*(?:that\b[\s:,\-]*)?([\s\S]+)$",
    re.I,
)


def system_messages():
    msgs = []
    rules = DEFAULT_RULES
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            rules = f.read().strip()
    except OSError:
        pass
    if rules:
        msgs.append({"role": "system", "content": "RULES (always apply):\n" + rules})
    if S.agent:
        msgs.append({"role": "system", "content": agent_prompt()})
    else:
        msgs.append({"role": "system", "content": (
            "You are a text-only chat assistant in a terminal. You have NO access to the user's computer: you "
            "cannot create, read, edit or delete files and you cannot run commands. Never claim you did. If the "
            "user asks for something like that, say so and tell them to type /agent (after /cd <their folder>) "
            "to enable agent mode. You may still show code or commands for them to run themselves.")})
    mem = load_memory()
    if mem:
        lines = ["- " + str(m.get("text", "")) for m in mem]
        while len(lines) > 1 and len("\n".join(lines)) > 5000:
            lines.pop(0)
        msgs.append({
            "role": "system",
            "content": (
                "SAVED MEMORY (what the user told you to remember):\n" + "\n".join(lines) + "\n"
                "Apply every entry naturally and without announcing it. Entries about \"the assistant\" "
                "are rules for you. Do not quote this list or mention 'memory' unless the user asks."
            ),
        })
    return msgs


# ------------------------------------------------------------------ streaming
def split_inline_think(answer):
    """Handle <think>...</think> tags that some models put inside the content."""
    think = ""
    while True:
        o = answer.find("<think>")
        if o < 0:
            break
        c = answer.find("</think>", o)
        inner = answer[o + 7: c if c >= 0 else len(answer)]
        think = (think + "\n" + inner).strip()
        answer = answer[:o] + (answer[c + 8:] if c >= 0 else "")
    return think, answer.strip()


def open_stream(body):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, stream=True, timeout=TIMEOUT)
    if not r.ok:
        try:
            err = r.json().get("error", r.text)
        except ValueError:
            err = r.text
        r.close()
        raise RuntimeError(err)
    return r


# ------------------------------------------------------------------ state
class State:
    def __init__(self):
        cfg = load_config()
        self.model = cfg.get("model", "")
        self.think = cfg.get("think", True)
        self.num_ctx = int(cfg.get("num_ctx", 16384))
        self.history = []          # chat messages (user / assistant only)
        self.pending_files = []    # (name, text) attached to the next message
        self.last_tokens = 0
        self.no_think_models = set()
        self.agent = False         # agent mode always starts off

    def save(self):
        save_config({"model": self.model, "think": self.think, "num_ctx": self.num_ctx})


S = State()


# ------------------------------------------------------------------ rendering
def render_reply(think_txt, answer, thinking_live, think_secs, show_cursor=False):
    parts = []
    if think_txt:
        if thinking_live:
            tail = "\n".join(think_txt.strip().splitlines()[-6:])
            parts.append(Text("▸ Thinking…", style=f"italic {MUTED}"))
            parts.append(Text(tail, style=f"dim {MUTED}"))
        else:
            parts.append(Text(f"▸ Thought for {think_secs:.0f}s", style=f"italic {MUTED}"))
        parts.append(Text(""))
    if answer:
        parts.append(Markdown(answer))
    elif not think_txt and show_cursor:
        parts.append(Text("…", style=MUTED))
    return Padding(Group(*parts), (0, 0, 0, 3))


def footer(model, secs, tps):
    t = Text()
    t.append("Build", style=f"bold {BLUE}")
    t.append(" · ", style=MUTED)
    t.append(model, style="bold")
    t.append(f" · {secs:.1f}s", style=MUTED)
    if tps:
        t.append(f" · {tps:.1f} tok/s", style=MUTED)
    return Padding(t, (1, 0, 1, 3))


def do_chat(user_text):
    """Send one user message, stream the reply, return True on success."""
    content = user_text
    for name, text in S.pending_files:
        content += f"\n\n[File: {name}]\n```\n{text}\n```"
    S.pending_files.clear()
    S.history.append({"role": "user", "content": content})

    msgs = system_messages() + S.history
    body = {
        "model": S.model,
        "messages": msgs,
        "stream": True,
        "keep_alive": "30m",
        "options": {"num_ctx": S.num_ctx, "num_predict": -1},
    }
    if S.model not in S.no_think_models:
        body["think"] = bool(S.think)

    start = time.time()
    try:
        try:
            resp = open_stream(body)
        except RuntimeError as e:
            if "think" in str(e).lower() and "think" in body:
                S.no_think_models.add(S.model)
                body.pop("think")
                resp = open_stream(body)
            else:
                raise
    except (requests.RequestException, RuntimeError) as e:
        S.history.pop()
        console.print(Padding(Text(f"Error: {e}", style="red"), (1, 0, 1, 3)))
        return False

    think_raw, content_raw = "", ""
    think_start = think_end = None
    final = {}
    interrupted = False

    console.print()
    live = Live(render_reply("", "", False, 0, True), console=console,
                refresh_per_second=12, vertical_overflow="visible")
    live.start()
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            if msg.get("thinking"):
                if think_start is None:
                    think_start = time.time()
                think_raw += msg["thinking"]
            if msg.get("content"):
                content_raw += msg["content"]
            if chunk.get("done"):
                final = chunk

            inline_think, answer = split_inline_think(content_raw)
            think_txt = (think_raw + "\n" + inline_think).strip() if inline_think else think_raw
            if think_txt and think_start is None:
                think_start = time.time()
            live_thinking = bool(think_txt) and not answer and not chunk.get("done")
            if think_txt and not live_thinking and think_end is None:
                think_end = time.time()
            secs = ((think_end or time.time()) - think_start) if think_start else 0
            live.update(render_reply(think_txt, answer, live_thinking, secs, True))
    except KeyboardInterrupt:
        interrupted = True
    except (requests.RequestException, RuntimeError) as e:
        live.stop()
        resp.close()
        S.history.pop()
        console.print(Padding(Text(f"Error: {e}", style="red"), (1, 0, 1, 3)))
        return False
    finally:
        resp.close()

    inline_think, answer = split_inline_think(content_raw)
    think_txt = (think_raw + "\n" + inline_think).strip() if inline_think else think_raw
    secs = ((think_end or time.time()) - think_start) if think_start else 0
    live.update(render_reply(think_txt, answer, False, secs))
    live.stop()

    if interrupted:
        console.print(Padding(Text("■ stopped", style=MUTED), (0, 0, 0, 3)))

    elapsed = time.time() - start
    ev, ed = final.get("eval_count"), final.get("eval_duration")
    tps = (ev / (ed / 1e9)) if ev and ed else (len(content_raw) / 4 / elapsed if elapsed else 0)
    S.last_tokens = (final.get("prompt_eval_count") or 0) + (ev or 0) or S.last_tokens
    console.print(footer(S.model, elapsed, tps))

    if answer or interrupted:
        S.history.append({"role": "assistant", "content": answer or "(stopped)"})
    else:
        S.history.pop()
    return True


# ------------------------------------------------------------------ agent mode
MAX_STEPS = 8
MAX_READ = 20_000
MAX_WRITE = 500_000
OUT_TO_MODEL = 4_000
CMD_TIMEOUT = 60
DANGER_RE = re.compile(
    r"\b(rm\s+-\w*r|rmdir|rd\s+/s|del\s|erase\s|format\s|shutdown|mkfs|diskpart|reg\s+delete|Remove-Item)\b", re.I)
OPEN_RE = re.compile(r"^\s*(`{3,})(file|run|read|ls)\b[ \t]*(.*?)\s*$")


def agent_prompt():
    return (
        "AGENT MODE is on. You can work on the user's project folder: " + os.getcwd() + "\n"
        "The computer runs " + ("WINDOWS and commands run in cmd.exe (use dir, mkdir, type, copy; NOT Linux or WSL "
                                 "paths like /mnt/c)" if os.name == "nt" else "Linux/macOS") + ". "
        "To create or change files, ALWAYS use a file block, not shell commands like touch or echo.\n"
        "You can only touch files inside that folder. If the user wants something somewhere else, tell them to "
        "run /cd <folder> first.\n"
        "You act ONLY by writing special fenced blocks in your reply. The program runs them and sends you the "
        "results in the next message. The user must approve every file write and every command.\n\n"
        "Blocks (paths are relative to the project folder):\n"
        "```ls <folder>```      list a folder (use . for the project root)\n"
        "```read <path>```      read a file\n"
        "```file <path>```      write a file: put the COMPLETE new file content on the lines after it\n"
        "```run```              run ONE shell command: put the command on the line after it\n"
        "Each block starts with a line like ```file src/app.py and ends with a line containing only ```.\n"
        "If the content itself contains triple backticks, open and close the block with four backticks.\n\n"
        "Rules: use ls/read to look at existing files before changing them. Always write the whole file, never "
        "a fragment. Do the next step only, then end your reply right after the block(s) and wait for the "
        "results. Never claim something was written or run until the results confirm it. Ordinary code in "
        "normal ```python style blocks is just for showing the user and does nothing."
    )


def clean_path(arg):
    arg = arg.strip()
    for prefix in ("path=", "path:", "file=", "file:"):
        if arg.lower().startswith(prefix):
            arg = arg[len(prefix):]
    return arg.strip().strip("\"'`").strip()


def parse_actions(text):
    actions, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, kind, arg = m.groups()
        body, j, closed = [], i + 1, False
        while j < len(lines):
            t = lines[j].strip()
            if t and set(t) == {"`"} and len(t) >= len(fence):
                closed = True
                break
            body.append(lines[j])
            j += 1
        i = j + 1
        if not closed:
            continue                      # a cut-off block is never run
        if kind == "run":
            cmd = "\n".join(body).strip() or arg
            if cmd:
                actions.append({"kind": "run", "cmd": cmd})
        elif kind == "file":
            path = clean_path(arg)
            if not path and body:
                path = clean_path(body.pop(0))
            if path:
                actions.append({"kind": "file", "path": path, "content": "\n".join(body) + "\n"})
        else:
            path = clean_path(arg) or (clean_path(body[0]) if body else "") or "."
            actions.append({"kind": kind, "path": path})
    return actions


def resolve_in_workspace(path):
    ws = os.path.realpath(os.getcwd())
    full = os.path.realpath(os.path.join(ws, path))
    try:
        common = os.path.commonpath([ws, full])
    except ValueError:
        return None
    return full if os.path.normcase(common) == os.path.normcase(ws) else None


def workspace_problem(path):
    p = os.path.realpath(path)
    if os.path.dirname(p) == p:
        return "that is a drive root"
    win = os.environ.get("SystemRoot")
    if win and os.path.normcase(p).startswith(os.path.normcase(os.path.realpath(win))):
        return "that is inside the Windows folder"
    if os.path.normcase(p) == os.path.normcase(os.path.realpath(os.path.expanduser("~"))):
        return "that is your whole home folder"
    return None


def confirm(question):
    try:
        ans = pt_prompt(FormattedText([("#4f9cf9 bold", "   ? "), ("", question + " [y/N] ")]))
    except (KeyboardInterrupt, EOFError):
        return False
    return ans.strip().lower() in ("y", "yes")


def action_header(label, detail, style=BLUE):
    t = Text("   ◆ ", style=style)
    t.append(label + " ", style="bold")
    t.append(detail, style=MUTED)
    console.print(t)


def show_text_block(text, lexer="text", max_lines=30):
    lines = text.rstrip("\n").splitlines()
    shown = "\n".join(lines[:max_lines])
    console.print(Padding(Syntax(shown, lexer, theme="monokai", background_color="default"), (0, 0, 0, 5)))
    if len(lines) > max_lines:
        console.print(Text(f"     … {len(lines) - max_lines} more lines", style=MUTED))


def do_ls(a):
    full = resolve_in_workspace(a["path"])
    action_header("list", a["path"])
    if not full or not os.path.isdir(full):
        return f"[ls {a['path']}] not found or outside the project folder."
    names = []
    for n in sorted(os.listdir(full), key=str.lower)[:200]:
        names.append(n + ("/" if os.path.isdir(os.path.join(full, n)) else ""))
    return f"[ls {a['path']}]\n" + "\n".join(names)


def do_read(a):
    full = resolve_in_workspace(a["path"])
    action_header("read", a["path"])
    if not full or not os.path.isfile(full):
        return f"[read {a['path']}] file not found or outside the project folder."
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(MAX_READ + 1)
    except OSError as e:
        return f"[read {a['path']}] error: {e}"
    note = "\n[truncated]" if len(data) > MAX_READ else ""
    return f"[read {a['path']}]\n```\n{data[:MAX_READ]}{note}\n```"


def do_write(a):
    path, new = a["path"], a["content"]
    full = resolve_in_workspace(path)
    if not full:
        action_header("write", path + "  (outside the project folder, blocked)", "red")
        return f"[file {path}] blocked: path is outside the project folder."
    if len(new) > MAX_WRITE:
        return f"[file {path}] blocked: file too large."
    old = None
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                old = f.read()
        except OSError:
            old = None
    lexer = os.path.splitext(path)[1].lstrip(".") or "text"
    if old is None:
        action_header("create", f"{path}  ({len(new.splitlines())} lines)")
        show_text_block(new, lexer)
    else:
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                            "a/" + path, "b/" + path, n=2))
        if not diff:
            action_header("write", path + "  (no changes)")
            return f"[file {path}] no changes: the file already has this content."
        action_header("edit", path)
        show_text_block(diff, "diff", 60)
    if not confirm("Apply this change?"):
        console.print(Text("     declined", style=MUTED))
        return f"[file {path}] the user DECLINED this change. Nothing was written."
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    except OSError as e:
        console.print(Text(f"     error: {e}", style="red"))
        return f"[file {path}] error: {e}"
    console.print(Text("     ✓ written", style=GREEN))
    return f"[file {path}] written ({len(new.splitlines())} lines)."


def do_run(a):
    cmd = a["cmd"]
    danger = bool(DANGER_RE.search(cmd))
    action_header("run", "", "red" if danger else BLUE)
    show_text_block(cmd, "bat" if os.name == "nt" else "bash", 10)
    if danger:
        console.print(Text("     ⚠ this command looks destructive, read it carefully", style="red"))
    if not confirm("Run this command?"):
        console.print(Text("     declined", style=MUTED))
        return f"[run] the user DECLINED to run: {cmd}"
    try:
        r = subprocess.run(cmd, shell=True, cwd=os.getcwd(), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=CMD_TIMEOUT,
                           stdin=subprocess.DEVNULL)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = f"(timed out after {CMD_TIMEOUT}s)", -1
    except OSError as e:
        out, code = str(e), -1
    if out:
        show_text_block(out, "text", 20)
    console.print(Text(f"     exit code {code}", style=GREEN if code == 0 else "red"))
    if len(out) > OUT_TO_MODEL:
        out = out[:OUT_TO_MODEL] + "\n[output truncated]"
    return f"[run] {cmd}\nexit code {code}\n{out}"


def execute_action(a):
    return {"ls": do_ls, "read": do_read, "file": do_write, "run": do_run}[a["kind"]](a)


def run_turn(user_text):
    ok = do_chat(user_text)
    steps = 0
    while ok and S.agent and steps < MAX_STEPS:
        last = S.history[-1] if S.history else {}
        actions = parse_actions(last.get("content", "")) if last.get("role") == "assistant" else []
        if not actions:
            return
        steps += 1
        console.print()
        results = [execute_action(a) for a in actions]
        console.print()
        ok = do_chat("[Tool results]\n\n" + "\n\n".join(results))
    if ok and S.agent and steps >= MAX_STEPS:
        console.print(Text(f"   Stopped after {MAX_STEPS} steps. Send a message to continue.", style=MUTED))


# ------------------------------------------------------------------ dialogs / commands
DIALOG_STYLE = Style.from_dict({
    "dialog": "bg:#0c0c0c",
    "dialog frame.label": "bg:#0c0c0c #4f9cf9 bold",
    "dialog.body": "bg:#141414 #e5e5e5",
    "dialog shadow": "bg:#000000",
    "button": "bg:#1e1e1e #e5e5e5",
    "button.focused": "bg:#4f9cf9 #000000",
    "radio-selected": "#4f9cf9 bold",
    "radio-checked": "#4f9cf9 bold",
})


def pick(title, values):
    if not values:
        return None
    return radiolist_dialog(title=title, text="↑/↓ to move · Enter to select · Esc to cancel",
                            values=values, style=DIALOG_STYLE).run()


def choose_model():
    try:
        models = list_models()
    except requests.RequestException as e:
        console.print(f"[red]Could not list models: {e}[/red]")
        return
    if not models:
        console.print(Text("   No models installed. Run: ollama pull <model>", style=MUTED))
        return
    vals = [(m["name"], f"{m['name']}   ({fmt_size(m.get('size', 0))})") for m in models]
    choice = pick("Select model", vals)
    if choice:
        S.model = choice
        S.save()
        console.print(Text(f"   Model: {S.model}", style=MUTED))


def ensure_model():
    if S.model:
        try:
            if any(m["name"] == S.model for m in list_models()):
                return
        except requests.RequestException:
            return
    try:
        models = list_models()
    except requests.RequestException:
        models = []
    if not models:
        console.print("[red]No Ollama models found. Run `ollama pull <model>` first.[/red]")
        sys.exit(1)
    if len(models) == 1:
        S.model = models[0]["name"]
        S.save()
    else:
        choose_model()
        if not S.model:
            S.model = models[0]["name"]
            S.save()


def show_memory():
    items = load_memory()
    if not items:
        console.print(Text("   No saved memory yet. Try: /remember I study IT", style=MUTED))
        return
    for i, m in enumerate(items, 1):
        console.print(Text(f"   {i:>3}. {m.get('text', '')}", style=MUTED))


def forget_memory():
    items = load_memory()
    if not items:
        console.print(Text("   Memory is empty.", style=MUTED))
        return
    vals = [(m["id"], (m["text"][:90] + "…") if len(m["text"]) > 90 else m["text"]) for m in items]
    choice = pick("Forget which memory?", vals)
    if choice:
        memory_delete(choice)
        console.print(Text("   Memory removed.", style=MUTED))


def attach_file(path):
    path = path.strip().strip('"')
    if not path:
        console.print(Text("   Usage: /file <path>", style=MUTED))
        return
    path = os.path.abspath(os.path.expanduser(path))
    try:
        if os.path.getsize(path) > FILE_MAX_BYTES:
            console.print(Text("   File too large (max 200 KB).", style="red"))
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        console.print(Text(f"   {e}", style="red"))
        return
    S.pending_files.append((os.path.basename(path), text))
    console.print(Text(f"   Attached {os.path.basename(path)} — it will go with your next message.", style=MUTED))


def new_chat():
    S.history.clear()
    S.pending_files.clear()
    S.last_tokens = 0
    console.clear()
    banner()


COMMANDS = [
    ("/models", "Switch model"),
    ("/think", "Toggle thinking on/off"),
    ("/agent", "Agent mode: edit files / run commands (asks first)"),
    ("/cd", "Change the project folder"),
    ("/memory", "Show saved memory"),
    ("/remember", "Save something to memory"),
    ("/forget", "Remove a memory"),
    ("/file", "Attach a text file to the next message"),
    ("/ctx", "Set context size (num_ctx)"),
    ("/new", "Start a new chat"),
    ("/help", "Show commands"),
    ("/exit", "Quit"),
]


def show_help():
    for cmd, desc in COMMANDS:
        t = Text("   ")
        t.append(f"{cmd:<11}", style=BLUE)
        t.append(desc, style=MUTED)
        console.print(t)
    console.print(Text("   Enter = send · Alt+Enter = new line · Ctrl+C = stop reply · Ctrl+P = command palette",
                       style=MUTED))


def run_command(line):
    """Returns False if the app should quit."""
    cmd, _, arg = line.strip().partition(" ")
    cmd = cmd.lower()
    arg = arg.strip()
    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd == "/palette":
        choice = pick("Commands", [(c, f"{c:<11} {d}") for c, d in COMMANDS])
        return run_command(choice) if choice else True
    if cmd in ("/models", "/model"):
        choose_model()
    elif cmd == "/think":
        S.think = not S.think
        S.save()
        console.print(Text(f"   Thinking: {'on' if S.think else 'off'}", style=MUTED))
    elif cmd == "/agent":
        want = (arg.lower() == "on") if arg.lower() in ("on", "off") else not S.agent
        if want:
            problem = workspace_problem(os.getcwd())
            if problem:
                console.print(Text(f"   Not enabling agent mode: {problem}. Use /cd <project folder> first.",
                                   style="red"))
            else:
                S.agent = True
                console.print(Text(f"   Agent mode ON. Project folder: {os.getcwd()}", style=GREEN))
                console.print(Text("   The model can list/read files here. Every write and command asks first.",
                                   style=MUTED))
        else:
            S.agent = False
            console.print(Text("   Agent mode off.", style=MUTED))
    elif cmd == "/cd":
        if not arg:
            console.print(Text(f"   {os.getcwd()}", style=MUTED))
        else:
            target = os.path.abspath(os.path.expanduser(arg.strip('"')))
            if not os.path.isdir(target):
                console.print(Text("   Folder not found.", style="red"))
            else:
                os.chdir(target)
                console.print(Text(f"   Project folder: {os.getcwd()}", style=MUTED))
                if S.agent and workspace_problem(target):
                    S.agent = False
                    console.print(Text("   Agent mode turned off: that folder is not a safe project folder.",
                                       style="red"))
    elif cmd == "/memory":
        show_memory()
    elif cmd == "/remember":
        if arg and memory_add(arg):
            console.print(Text("   Saved to memory.", style=MUTED))
        elif not arg:
            console.print(Text("   Usage: /remember <something>", style=MUTED))
        else:
            console.print(Text("   Already in memory.", style=MUTED))
    elif cmd == "/forget":
        if arg.lower() == "all":
            memory_clear()
            console.print(Text("   Memory cleared.", style=MUTED))
        else:
            forget_memory()
    elif cmd == "/file":
        attach_file(arg)
    elif cmd == "/ctx":
        if arg.isdigit() and 2048 <= int(arg) <= 131072:
            S.num_ctx = int(arg)
            S.save()
            console.print(Text(f"   Context size: {S.num_ctx}", style=MUTED))
        else:
            console.print(Text(f"   Current: {S.num_ctx}. Usage: /ctx 16384", style=MUTED))
    elif cmd in ("/new", "/clear"):
        new_chat()
    elif cmd in ("/help", "/?"):
        show_help()
    else:
        console.print(Text(f"   Unknown command: {cmd}  (try /help)", style=MUTED))
    return True


# ------------------------------------------------------------------ prompt UI
PT_STYLE = Style.from_dict({
    "bar": BLUE,
    "": "bg:#141414 #e5e5e5",
    "bottom-toolbar": "bg:#000000 #8a8a8a noreverse",
    "bottom-toolbar.text": "bg:#000000 #8a8a8a noreverse",
    "tb.mode": f"bg:#000000 {BLUE} bold",
    "tb.model": "bg:#000000 #e5e5e5 bold",
    "tb.key": "bg:#000000 #e5e5e5 bold",
    "tb.ok": f"bg:#000000 {GREEN}",
    "completion-menu": "bg:#141414 #cfcfcf",
    "completion-menu.completion.current": "bg:#4f9cf9 #000000",
})


def toolbar():
    width = console.size.width
    cwd = os.getcwd()
    pct = int(S.last_tokens / S.num_ctx * 100) if S.num_ctx else 0
    if S.last_tokens >= 1000:
        tok = f"{S.last_tokens / 1000:.1f}K"
    else:
        tok = str(S.last_tokens)
    right_txt = f"{tok} ({pct}%)  ctrl+p commands"
    top = [("class:tb.mode", " Build"), ("class:bottom-toolbar", " · "),
           ("class:tb.model", S.model or "no model"),
           ("class:bottom-toolbar", "  " + ("think on" if S.think else "think off"))]
    if S.agent:
        top.append(("class:tb.ok", "  agent on"))
    if S.pending_files:
        top.append(("class:tb.ok", f"  📎 {len(S.pending_files)}"))
    pad = max(1, width - len(cwd) - len(right_txt) - 2)
    bottom = [("class:bottom-toolbar", " " + cwd + " " * pad),
              ("class:bottom-toolbar", f"{tok} ({pct}%)  "),
              ("class:tb.key", "ctrl+p"), ("class:bottom-toolbar", " commands")]
    return FormattedText(top + [("", "\n")] + bottom)


def make_session():
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-p")
    def _palette(event):
        event.app.exit(result="/palette")

    @kb.add("c-l")
    def _clear(event):
        event.app.renderer.clear()

    return PromptSession(
        message=FormattedText([("class:bar", "┃ ")]),
        multiline=True,
        prompt_continuation=lambda w, n, s: FormattedText([("class:bar", "┃ ")]),
        key_bindings=kb,
        style=PT_STYLE,
        bottom_toolbar=toolbar,
        history=FileHistory(HISTORY_FILE),
        completer=WordCompleter([c for c, _ in COMMANDS], sentence=True),
        complete_while_typing=False,
        reserve_space_for_menu=0,
    )


def banner():
    console.print()
    t = Text("   Local AI", style=f"bold {BLUE}")
    t.append("  ·  terminal", style=MUTED)
    console.print(t)
    console.print(Text("   Type a message, or /help for commands.  /agent lets it edit files in this folder.", style=MUTED))
    console.print()


# ------------------------------------------------------------------ main
def main():
    if os.name == "nt":
        os.system("")          # enable ANSI colours in classic cmd.exe
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("OC | Local AI")
        except Exception:
            pass
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])            # e.g. drag a project folder onto run_cli.bat
    start_ollama()
    ensure_model()
    console.clear()
    banner()
    session = make_session()

    while True:
        try:
            line = session.prompt()
        except KeyboardInterrupt:
            console.print(Text("   Ctrl+C — type /exit or press Ctrl+D to quit.", style=MUTED))
            continue
        except EOFError:
            break

        line = (line or "").strip()
        if not line:
            continue
        if line.startswith("/"):
            if not run_command(line):
                break
            continue

        m = REMEMBER_RE.match(line)
        if m and memory_add(m.group(1)):
            console.print(Text("   ✓ saved to memory", style=MUTED))

        try:
            run_turn(line)
        except KeyboardInterrupt:
            console.print(Text("   ■ stopped", style=MUTED))

    console.print(Text("\n   Bye!", style=MUTED))


if __name__ == "__main__":
    main()
