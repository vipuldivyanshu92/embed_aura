# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies

Open PowerShell in this directory and run:
```powershell
pip install -r requirements.txt
```

### 2. Get Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy your API key

### 3. Run the App

```powershell
python snip_tool.py
```

Then click the "⚙️ Settings" button and paste your API key!

## Instant Start

When you open the app, it **automatically captures your screen**! The AI detects what's in the image and suggests specific questions:

**Examples:**
- Error message? → "What's causing this error?", "How do I fix this?"
- Code snippet? → "Explain this function", "Suggest optimizations"
- UI screenshot? → "What UX issues exist?", "Improve this interface"
- Data/charts? → "Summarize insights", "Extract key metrics"

Click any suggestion for instant answers!

## Taking New Snips

**Optional: Set a Delay**
- Use the "Delay" dropdown in the toolbar if you need time to:
  - Open a menu or tooltip
  - Position windows
  - Setup your screen
- Options: No delay, 3s, 5s, 10s

**Option 1: Selective Capture**
1. Click "📸 Take Snip"
2. (If delay set) Wait for countdown, press ESC to cancel
3. Click and drag to select an area

**Option 2: Full Screen Capture**
1. Click "🖥️ Capture Screen"
2. (If delay set) Wait for countdown, press ESC to cancel
3. Screen captured!

**Then:**
3. Your snip appears in the center and is automatically saved to history!
4. Wait a moment while AI analyzes your snip
5. Click any suggested question for an **instant answer** (already pre-generated!)
6. Use "📋 Copy Image" to copy to clipboard or "💾 Save Image" to save to file
7. Or type your own questions in the chat!

## Using History

- All your snips are saved automatically in the **left sidebar**
- **AI automatically names** each snip based on what's in it (e.g., "Python Error Fix", "Login Form Design")
- **Click** any snip to reload it
- **Double-click** any snip name to rename it manually
- Click "🗑️ Clear All" to delete all history

## Building an Executable

Want a standalone .exe file? Just run:
```powershell
.\build_exe.bat
```

Your executable will be in the `dist` folder!

## Tips

- **Auto-capture on startup** - The app is instantly ready with a screen capture!
- **Capture delay** - Set a delay timer to setup your screen before capture
- **Floating countdown** - Small timer in top-right corner (doesn't block interactions!)
- **Switch tabs freely** - Open menus, hover tooltips while countdown runs
- **Press ESC** - Cancel the countdown at any time
- **Loading animation** - Watch the dots (. .. ...) animate while waiting for AI responses
- **Drag dividers** - Resize the history, preview, and chat panes by dragging the dividers between them
- **ESC key** - Cancel snip selection
- **Enter key** - Send chat message
- **Ctrl+C** - Copy selected text in chat
- **Right-click** in chat - Quick copy menu
- **Model dropdown** - Switch between GPT-4o-mini (faster/cheaper) and GPT-4o (more capable)
- **Smart naming** - AI automatically generates meaningful names for your snips based on content
- The app remembers your common questions and suggests them
- Click "📋 Copy Last Response" to quickly copy AI's answer
- History saves your last 100 snips automatically
- You can go back to any previous snip anytime!

## Troubleshooting

**"Please set your OpenAI API key"**
- Click Settings and enter your API key from OpenAI

**"Error: ..."**
- Check that you have credits in your OpenAI account
- Verify your API key is correct
- Make sure you have internet connection

**Can't capture screen**
- Try running as administrator if on Windows 11

## What Can You Do?

The AI auto-detects your content and suggests relevant questions! But you can also ask:
- "What does this error mean and how do I fix it?"
- "Extract all the text from this image"
- "Summarize this document"
- "Explain what this code does and suggest improvements"
- "What UX issues exist in this interface?"
- "Translate this text to Spanish"

**Pro Tip:** The suggestions adapt to what's in your screenshot - error messages get debugging suggestions, code gets explanation suggestions, UI gets UX suggestions!

Enjoy! 🚀


