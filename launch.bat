@echo off
cd /d "%~dp0"
python -c "import sys; sys.path.insert(0, 'src'); from artifice.ui.window import run; run()"
