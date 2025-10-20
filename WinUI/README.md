# SnipLM - AI-Powered Snipping Tool

An intelligent Windows snipping tool that integrates with OpenAI's vision models to help you interact with your screenshots.

## Features

- 📸 Easy screen capture with click-and-drag selection
- ⏱️ Optional capture delay (3s, 5s, 10s) with visual countdown
- 🖥️ Instant full-screen capture with one click
- 🚀 Auto-capture on startup - immediately chat with your current screen
- 🤖 AI-powered image analysis using OpenAI's vision models
- 💬 Chat with your screenshots with animated loading indicators
- 💡 Smart pattern-detection suggestions with instant pre-generated answers
- 🔍 Auto-detects content type: errors, code, UI, data, docs, diagrams
- ⚡ One-click answers - specific to what's in your screenshot (not generic!)
- 💾 Cached suggestions - reload old snips with their original suggestions instantly
- 🧠 Memory system that learns your common questions
- ⚙️ Configurable AI models (GPT-4o-mini, GPT-4o, GPT-4-turbo)
- 📋 Copy snips to clipboard
- 💾 Save snips to file (PNG, JPG, BMP)
- 📝 Copy individual chat responses or selected text
- ⌨️ Right-click context menu in chat for quick copying
- 📚 History sidebar - browse and reload previous snips
- 🏷️ Smart contextual names - AI automatically names snips based on content
- ✏️ Rename snips by double-clicking in history
- 🕒 Auto-saved snip history (up to 100 snips)
- 🔧 Resizable panes - drag dividers to customize your layout

## Installation

1. Clone or download this repository

2. Install Python 3.8 or higher

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python snip_tool.py
```

## Setup

1. Get your OpenAI API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

2. Click the "⚙️ Settings" button in the app

3. Enter your API key and save

## Usage

**On Startup:**
- The app automatically captures your screen and is ready to answer questions immediately!

1. **Take a Snip**: 
   - Set an optional delay (3s, 5s, 10s) in the toolbar if you need time to setup your screen
   - Click "📸 Take Snip" to select a region of your screen
   - OR click "🖥️ Capture Screen" to capture your entire screen instantly
   - A floating countdown timer appears in top-right if delay is set (press ESC to cancel)

2. **Browse History**: Click any previous snip in the left sidebar to load it

3. **Rename Snips**: Double-click any snip name in the history to rename it

4. **Copy or Save**: Use the "📋 Copy Image" button to copy to clipboard or "💾 Save Image" to save to a file

5. **Smart Suggestions**: After snipping, AI analyzes the image and generates relevant questions - click any suggestion for an instant answer!

6. **Chat**: Ask your own questions about the snip

7. **Copy Responses**: Click "📋 Copy Last Response" or right-click in chat to copy any text

8. **Extract Information**: Ask the AI to extract text, explain concepts, or provide insights

## Building Executable

To create a standalone Windows executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name SnipLM --icon=icon.ico snip_tool.py
```

The executable will be in the `dist` folder.

## Configuration Files

- `config.json`: Stores your API key and model preferences
- `memory.json`: Stores your snip metadata and common questions
- `snip_history/`: Directory where snip images are stored (auto-created)

## Tips

- The app captures your screen automatically when you open it - ready to chat immediately!
- Set a capture delay if you need time to setup your screen (menus, tooltips, etc.)
- A floating countdown timer appears in the top-right corner (doesn't block interactions!)
- Switch tabs, open menus, hover tooltips - the countdown stays visible
- **Drag the dividers** between panes to resize them to your preference
- Watch the animated dots (. .. ...) while waiting for AI responses
- The AI detects content patterns and generates highly specific suggestions:
  - **Errors** → "What's causing this error?", "How to fix?"
  - **Code** → "Explain this function", "Suggest improvements"
  - **UI** → "What UX issues exist?", "Suggest improvements"
  - **Data** → "Summarize insights", "Extract metrics"
- Suggested questions are contextual and come with pre-generated answers
- Click any suggestion for an instant answer (no API call needed!)
- Suggestions are saved with snips - reload old snips with their original suggestions
- **AI automatically names your snips** based on content: "Python Error Fix", "Login Form", etc.
- The AI remembers your chat history for each snip session
- Press ESC during screen selection to cancel
- You can switch between different GPT models in the toolbar
- **Right-click** in the chat area for quick copy options
- **Ctrl+C** works to copy selected text in chat
- Use "📋 Copy Last Response" button for quick copying of AI responses
- **Double-click** any snip in the history sidebar to rename it
- History automatically saves your last 100 snips
- Click "🗑️ Clear All" in the history panel to delete all saved snips

## Requirements

- Windows 10 or higher
- Python 3.8+
- OpenAI API key

## Privacy

- All data is stored locally on your machine
- Your API key is stored in `config.json` on your computer
- Screenshots are sent to OpenAI's API for processing according to their privacy policy


