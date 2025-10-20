@echo off
echo ====================================
echo Building SnipLM Executable...
echo ====================================
echo.

REM Check if pyinstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install --user pyinstaller
    echo.
)

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "SnipLM.spec" del SnipLM.spec
echo.

REM Build the executable
echo Building executable...
pyinstaller --onefile --windowed --name SnipLM snip_tool.py --add-data "config.json;." --add-data "memory.json;." --hidden-import=PIL._tkinter_finder

echo.
echo ====================================
echo Build complete!
echo ====================================
echo.
echo Your executable is located at: dist\SnipLM.exe
echo.
echo To create a desktop shortcut:
echo 1. Right-click on dist\SnipLM.exe
echo 2. Click "Send to" -^> "Desktop (create shortcut)"
echo.
echo Or run: create_shortcut.bat
echo.
pause
