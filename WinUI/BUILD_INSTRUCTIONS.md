# Building SnipLM Executable

## Quick Build

Simply double-click `build_exe.bat` to create the executable!

## Step-by-Step Instructions

### 1. Build the Executable

```powershell
.\build_exe.bat
```

This will:
- Install PyInstaller (if needed)
- Clean previous builds
- Create a standalone `SnipLM.exe` in the `dist` folder

The build takes about 30-60 seconds.

### 2. Create Desktop Shortcut

**Option A: Automatic (Recommended)**
```powershell
.\create_shortcut.bat
```

**Option B: Manual**
1. Navigate to the `dist` folder
2. Right-click on `SnipLM.exe`
3. Click "Send to" → "Desktop (create shortcut)"

### 3. Launch SnipLM

Double-click the shortcut on your desktop!

## Additional Options

### Keyboard Shortcut (Optional)

To add a global keyboard shortcut:

1. Right-click the desktop shortcut
2. Click "Properties"
3. Click in the "Shortcut key" field
4. Press your desired key combo (e.g., `Ctrl+Alt+S`)
5. Click "OK"

Now you can launch SnipLM from anywhere!

### Pin to Taskbar

For even faster access:
1. Navigate to `dist\SnipLM.exe`
2. Right-click and select "Pin to taskbar"

## Distributing

The executable in `dist\SnipLM.exe` is completely standalone! You can:
- Copy it to any Windows PC (no Python required)
- Share it with colleagues
- Run it from a USB drive

**Note**: Config files (`config.json`, `memory.json`) and snip history will be created in the same directory as the executable.

## Troubleshooting

### "PyInstaller not found"
The build script automatically installs PyInstaller. If it fails:
```powershell
pip install --user pyinstaller
```

### Build fails
Make sure all dependencies are installed:
```powershell
pip install --user -r requirements.txt
```

### Executable won't run
- Check Windows Defender/antivirus isn't blocking it
- Try running as administrator
- Ensure OpenAI API key is configured

### Shortcut creation blocked
If PowerShell is blocked, manually create the shortcut:
1. Right-click `dist\SnipLM.exe`
2. Select "Send to" → "Desktop (create shortcut)"

## File Sizes

- Executable: ~15-20 MB (includes Python runtime and all dependencies)
- First run creates `config.json` (~200 bytes) and `memory.json` (~100 bytes)
- Each snip: ~50-500 KB depending on screenshot size

Enjoy your portable SnipLM! 🚀

