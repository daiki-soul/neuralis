# Neuralis

A local, private AI chat for [Ollama](https://ollama.com): a browser UI and a terminal CLI, both running entirely on your machine.

## Features

- **Web UI**: Flask server with streaming chat, model picker, thinking display, and settings
- **CLI**: rich terminal chat with slash commands, file attachments, and an optional agent mode that edits files and runs commands (asks first)
- **Persistent memory**: saved to `memory.json`, editable from the UI or via `/remember`, `/memory`, `/forget`
- **Auto-start**: launches `ollama serve` if it isn't running and opens Chrome for you

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with at least one model pulled (e.g. `ollama pull qwen3:8b`)

## Setup

```bash
pip install flask requests rich prompt_toolkit
```

## Usage

**Web UI**

```bash
python app.py
```

Opens at http://127.0.0.1:8000/

**CLI** (Windows)

```bash
run_cli.bat
```

Or run `python ollama_cli.py` directly.

### CLI commands

| Command | Description |
| --- | --- |
| `/models` | Switch model |
| `/think` | Toggle thinking |
| `/agent` | Agent mode (edit files, run commands) |
| `/cd` | Change project folder |
| `/memory`, `/remember`, `/forget` | Manage memory |
| `/file` | Attach a text file |
| `/ctx` | Set context size |
| `/new` | New chat |
| `/help`, `/exit` | Help, quit |

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Web server and Ollama proxy |
| `index.html` | Web UI |
| `ollama_cli.py` | Terminal client |
| `cli_config.json` | CLI model, thinking, and context settings |
| `memory.json` | Saved memories |
