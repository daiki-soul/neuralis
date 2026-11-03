@echo off
title OC ^| Local AI
cd /d "%~dp0"
python -c "import rich, prompt_toolkit, requests" 2>nul || pip install rich prompt_toolkit requests
python ollama_cli.py %*
