# PowerShell script to create desktop shortcut for SnipLM

$exePath = Join-Path $PSScriptRoot "dist\SnipLM.exe"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "SnipLM.lnk"

if (-not (Test-Path $exePath)) {
    Write-Host "Error: SnipLM.exe not found at $exePath" -ForegroundColor Red
    Write-Host "Please run build_exe.bat first to create the executable." -ForegroundColor Yellow
    pause
    exit
}

Write-Host "Creating desktop shortcut..." -ForegroundColor Green

$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $exePath
$Shortcut.WorkingDirectory = Split-Path $exePath
$Shortcut.Description = "AI-Powered Snipping Tool"
$Shortcut.Save()

Write-Host "✓ Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "Location: $shortcutPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now launch SnipLM from your desktop!" -ForegroundColor Yellow
pause

