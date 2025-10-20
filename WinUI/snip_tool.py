import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
import mss
import json
import os
from PIL import Image, ImageTk, ImageGrab
import io
import base64
from datetime import datetime
import pyperclip
from openai import OpenAI
import threading

class SnipLM:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SnipLM - AI-Powered Snipping Tool")
        self.root.geometry("1200x700")
        
        self.config = self.load_config()
        self.memory = self.load_memory()
        self.current_snip = None
        self.current_snip_base64 = None
        self.current_snip_id = None
        self.chat_history = []
        self.suggestion_cache = {}  # Cache for pre-generated answers
        self.loading_animation_id = None  # Track loading animation
        self.loading_dots = 0  # Track dot count for animation
        self.loading_mark = None  # Track where loading text is
        
        # Create history directory if it doesn't exist
        self.history_dir = os.path.join(os.getcwd(), 'snip_history')
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        
        self.setup_ui()
        
        # Load existing history
        self.refresh_history_list()
        
        # Capture screen on startup after a delay
        self.root.after(500, self.initial_screen_capture)
        
    def load_config(self):
        """Load configuration from config.json"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                # Ensure capture_delay exists
                if 'capture_delay' not in config:
                    config['capture_delay'] = 'No delay'
                return config
        except FileNotFoundError:
            return {
                "openai_api_key": "",
                "model": "gpt-4o-mini",
                "available_models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                "capture_delay": "No delay"
            }
    
    def save_config(self):
        """Save configuration to config.json"""
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_memory(self):
        """Load memory from memory.json"""
        try:
            with open('memory.json', 'r') as f:
                data = json.load(f)
                # Ensure we have the snips key (new format)
                if 'snips' not in data:
                    data['snips'] = []
                return data
        except FileNotFoundError:
            return {"snip_history": [], "common_questions": [], "snips": []}
    
    def save_memory(self):
        """Save memory to memory.json"""
        with open('memory.json', 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def setup_ui(self):
        """Setup the main UI"""
        # Top toolbar
        toolbar = tk.Frame(self.root, bg='#f0f0f0', height=60)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Snip button
        self.snip_btn = tk.Button(
            toolbar, 
            text="📸 Take Snip", 
            command=self.take_snip,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.snip_btn.pack(side=tk.LEFT, padx=5)
        
        # Full screen capture button
        self.fullscreen_btn = tk.Button(
            toolbar,
            text="🖥️ Capture Screen",
            command=self.capture_full_screen,
            bg='#2196F3',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        settings_btn = tk.Button(
            toolbar,
            text="⚙️ Settings",
            command=self.open_settings,
            font=('Arial', 10),
            padx=10,
            pady=8
        )
        settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Model selector
        tk.Label(toolbar, text="Model:", bg='#f0f0f0', font=('Arial', 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.model_var = tk.StringVar(value=self.config.get('model', 'gpt-4o-mini'))
        model_dropdown = ttk.Combobox(
            toolbar,
            textvariable=self.model_var,
            values=self.config.get('available_models', ['gpt-4o-mini']),
            state='readonly',
            width=15
        )
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Delay timer selector
        tk.Label(toolbar, text="Delay:", bg='#f0f0f0', font=('Arial', 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.delay_var = tk.StringVar(value=self.config.get('capture_delay', 'No delay'))
        delay_options = ['No delay', '3 seconds', '5 seconds', '10 seconds']
        delay_dropdown = ttk.Combobox(
            toolbar,
            textvariable=self.delay_var,
            values=delay_options,
            state='readonly',
            width=12
        )
        delay_dropdown.pack(side=tk.LEFT, padx=5)
        delay_dropdown.bind('<<ComboboxSelected>>', self.on_delay_change)
        
        # Main container with PanedWindow for resizable panes
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # History sidebar - Far left
        history_frame = tk.Frame(main_container, bg='white', relief=tk.SUNKEN, bd=2, width=200)
        main_container.add(history_frame, minsize=150)
        
        tk.Label(history_frame, text="📚 History", font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        # History list with scrollbar
        history_list_frame = tk.Frame(history_frame, bg='white')
        history_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        history_scrollbar = tk.Scrollbar(history_list_frame)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(
            history_list_frame,
            yscrollcommand=history_scrollbar.set,
            font=('Arial', 9),
            bg='#f9f9f9',
            selectmode=tk.SINGLE
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.config(command=self.history_listbox.yview)
        
        # Bind click event
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        self.history_listbox.bind('<Double-Button-1>', self.on_history_rename)
        
        # Clear history button
        clear_history_btn = tk.Button(
            history_frame,
            text="🗑️ Clear All",
            command=self.clear_history,
            font=('Arial', 8),
            bg='#ffebee',
            fg='#c62828',
            padx=5,
            pady=3
        )
        clear_history_btn.pack(pady=5)
        
        # Middle - Image preview
        left_frame = tk.Frame(main_container, bg='white', relief=tk.SUNKEN, bd=2)
        main_container.add(left_frame, minsize=300)
        
        tk.Label(left_frame, text="Snip Preview", font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        self.image_frame = tk.Label(left_frame, bg='#f9f9f9', text="No snip taken yet\n\nClick 'Take Snip' to capture your screen", 
                                    font=('Arial', 11), fg='#666')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image action buttons
        image_actions = tk.Frame(left_frame, bg='white')
        image_actions.pack(fill=tk.X, padx=5, pady=5)
        
        self.copy_btn = tk.Button(
            image_actions,
            text="📋 Copy Image",
            command=self.copy_to_clipboard,
            bg='#2196F3',
            fg='white',
            font=('Arial', 9, 'bold'),
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.copy_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.save_btn = tk.Button(
            image_actions,
            text="💾 Save Image",
            command=self.save_to_file,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 9, 'bold'),
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Right side - Chat interface
        right_frame = tk.Frame(main_container, bg='white', relief=tk.SUNKEN, bd=2)
        main_container.add(right_frame, minsize=400)
        
        tk.Label(right_frame, text="Chat with your Snip", font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        # Suggested questions
        self.suggestions_frame = tk.Frame(right_frame, bg='white')
        self.suggestions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='#f9f9f9',
            state=tk.DISABLED,
            height=20,
            exportselection=True  # Allow copying selected text
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Enable text selection in chat (bind events for copy functionality)
        self.chat_display.bind('<Button-3>', self.show_chat_context_menu)
        self.chat_display.bind('<Control-c>', self.copy_selected_chat_kbd)
        
        # Create context menu for chat
        self.chat_context_menu = tk.Menu(self.chat_display, tearoff=0)
        self.chat_context_menu.add_command(label="Copy Selected Text", command=self.copy_selected_chat)
        self.chat_context_menu.add_command(label="Copy Last Response", command=self.copy_last_response)
        self.chat_context_menu.add_separator()
        self.chat_context_menu.add_command(label="Select All", command=self.select_all_chat)
        
        # Chat actions (copy button)
        chat_actions = tk.Frame(right_frame, bg='white')
        chat_actions.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.copy_response_btn = tk.Button(
            chat_actions,
            text="📋 Copy Last Response",
            command=self.copy_last_response,
            bg='#e3f2fd',
            font=('Arial', 9),
            padx=10,
            pady=3,
            state=tk.DISABLED
        )
        self.copy_response_btn.pack(side=tk.LEFT, padx=2)
        
        # Chat input
        input_frame = tk.Frame(right_frame, bg='white')
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.chat_input = tk.Entry(input_frame, font=('Arial', 11))
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.chat_input.bind('<Return>', lambda e: self.send_message())
        
        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15
        )
        send_btn.pack(side=tk.RIGHT)
        
    def get_delay_seconds(self):
        """Get the delay in seconds from the dropdown"""
        delay_str = self.delay_var.get()
        if delay_str == 'No delay':
            return 0
        elif delay_str == '3 seconds':
            return 3
        elif delay_str == '5 seconds':
            return 5
        elif delay_str == '10 seconds':
            return 10
        return 0
    
    def take_snip(self):
        """Initiate screen capture with optional delay"""
        delay = self.get_delay_seconds()
        
        if delay > 0:
            # Show countdown then capture
            self.show_countdown(delay, self._delayed_capture_screen)
        else:
            # Immediate capture
            self.root.iconify()
            self.root.after(200, self.capture_screen)
    
    def capture_screen(self):
        """Capture screen region"""
        # Create selection window
        selector = ScreenSelector(self.root, self.on_snip_complete)
    
    def initial_screen_capture(self):
        """Capture screen on app startup"""
        # Minimize window briefly
        self.root.iconify()
        
        # Wait a moment for window to minimize, then capture
        self.root.after(300, self._do_initial_capture)
    
    def _do_initial_capture(self):
        """Execute the initial screen capture"""
        try:
            # Capture entire screen
            screenshot = ImageGrab.grab()
            
            # Restore window
            self.root.deiconify()
            
            # Process the screenshot silently
            self.on_snip_complete(screenshot)
        except Exception as e:
            # Silently fail - don't show error on startup
            self.root.deiconify()
            print(f"Initial capture failed: {e}")
    
    def capture_full_screen(self):
        """Capture the entire screen with optional delay"""
        delay = self.get_delay_seconds()
        
        if delay > 0:
            # Show countdown then capture
            self.show_countdown(delay, self._delayed_full_screen_capture)
        else:
            # Immediate capture
            self.root.iconify()
            self.root.after(200, self._do_full_screen_capture)
    
    def _delayed_capture_screen(self):
        """Execute delayed snip capture"""
        self.root.iconify()
        self.root.after(200, self.capture_screen)
    
    def _delayed_full_screen_capture(self):
        """Execute delayed full screen capture"""
        self.root.iconify()
        self.root.after(200, self._do_full_screen_capture)
    
    def _do_full_screen_capture(self):
        """Execute the full screen capture"""
        try:
            # Capture entire screen
            screenshot = ImageGrab.grab()
            
            # Restore window
            self.root.deiconify()
            
            # Process the screenshot
            self.on_snip_complete(screenshot)
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("Error", f"Could not capture screen: {str(e)}")
    
    def on_snip_complete(self, image):
        """Handle completed snip"""
        self.root.deiconify()  # Restore main window
        
        if image is None:
            return
        
        self.current_snip = image
        
        # Convert to base64 for API
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        self.current_snip_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Display image
        self.display_image(image)
        
        # Enable copy/save buttons
        self.copy_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        
        # Save to memory
        self.add_to_memory(image)
        
        # Clear chat
        self.chat_history = []
        self.update_chat_display()
        
        # Generate suggested questions
        self.generate_suggestions()
    
    def display_image(self, image):
        """Display the snipped image"""
        # Make a copy so we don't modify the original
        display_image = image.copy()
        
        # Get the available space in the image frame
        # Use reasonable default dimensions
        max_width = 600
        max_height = 500
        
        # Calculate scaling to fit within available space while maintaining aspect ratio
        img_width, img_height = display_image.size
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_image)
        self.image_frame.config(image=photo, text="")
        self.image_frame.image = photo  # Keep reference
        self.image_frame.display_copy = display_image  # Keep the resized copy
    
    def add_to_memory(self, image):
        """Add snip to memory and save to disk"""
        # Generate unique ID
        timestamp = datetime.now()
        snip_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save image to disk
        image_path = os.path.join(self.history_dir, f"{snip_id}.png")
        image.save(image_path)
        
        # Create snip entry with temporary name
        snip_entry = {
            'id': snip_id,
            'name': f"📸 {timestamp.strftime('%I:%M %p')}",  # Temporary name, will be updated by AI
            'timestamp': timestamp.isoformat(),
            'path': image_path,
            'size': list(image.size),
            'suggestions_path': os.path.join(self.history_dir, f"{snip_id}_suggestions.json")
        }
        
        # Add to memory
        self.memory['snips'].insert(0, snip_entry)  # Add to front
        
        # Keep only last 100
        self.memory['snips'] = self.memory['snips'][:100]
        
        # Remove old image and suggestion files if needed
        if len(self.memory['snips']) >= 100:
            for old_snip in self.memory['snips'][100:]:
                old_path = old_snip.get('path')
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except:
                        pass
                old_sugg_path = old_snip.get('suggestions_path')
                if old_sugg_path and os.path.exists(old_sugg_path):
                    try:
                        os.remove(old_sugg_path)
                    except:
                        pass
        
        self.save_memory()
        self.current_snip_id = snip_id
        
        # Refresh history list
        self.refresh_history_list()
    
    def save_suggestions_to_history(self, suggestions_data):
        """Save generated suggestions to disk for current snip"""
        if not self.current_snip_id:
            return
        
        # Find current snip in memory
        for snip in self.memory.get('snips', []):
            if snip.get('id') == self.current_snip_id:
                suggestions_path = snip.get('suggestions_path')
                if suggestions_path:
                    try:
                        with open(suggestions_path, 'w') as f:
                            json.dump(suggestions_data, f, indent=2)
                    except Exception as e:
                        print(f"Error saving suggestions: {e}")
                break
    
    def generate_suggestions(self):
        """Generate smart suggestions based on the snip using LLM"""
        # Clear previous suggestions and cache
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        self.suggestion_cache = {}
        
        if not self.config.get('openai_api_key'):
            tk.Label(
                self.suggestions_frame,
                text="⚠️ Please set your OpenAI API key in Settings",
                bg='#fff3cd',
                fg='#856404',
                font=('Arial', 9)
            ).pack(pady=5)
            return
        
        # Show loading indicator
        loading_label = tk.Label(
            self.suggestions_frame,
            text="🔄 Analyzing image...",
            bg='white',
            font=('Arial', 9, 'italic'),
            fg='#666'
        )
        loading_label.pack(anchor='w', pady=5)
        
        # Call LLM in background to generate smart suggestions
        thread = threading.Thread(target=self.generate_smart_suggestions)
        thread.daemon = True
        thread.start()
    
    def generate_smart_suggestions(self):
        """Call LLM to analyze image and generate relevant questions with answers"""
        try:
            client = OpenAI(api_key=self.config['openai_api_key'])
            
            # Ask LLM to analyze the image and suggest questions
            response = client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing screenshots and detecting user intent.

STEP 1: Generate a SHORT, DESCRIPTIVE NAME (2-4 words) for this screenshot based on its content.
Examples: "Python IndexError Fix", "Login Form Design", "Sales Dashboard", "Git Merge Conflict"

STEP 2: Detect the content type. Look for these patterns:

🔴 ERROR/EXCEPTION:
- Keywords: "Error", "Exception", "Traceback", "Failed", "Warning", "at line"
- Suggest: "What does this error mean?", "What's causing this error?", "How do I fix this?"

💻 CODE:
- Patterns: function definitions, imports, brackets, syntax highlighting
- Suggest: "What does this code do?", "Explain this function", "Suggest improvements", "Add comments to this code"

📊 DATA/CHARTS:
- Patterns: graphs, tables, numbers, statistics
- Suggest: "Summarize these insights", "What trends are shown?", "Extract key metrics"

🎨 UI/DESIGN:
- Patterns: buttons, forms, navigation, layouts
- Suggest: "What UX issues exist?", "Suggest UI improvements", "Describe this workflow"

📄 DOCUMENT/TEXT:
- Patterns: paragraphs, articles, documentation
- Suggest: "Summarize this content", "Extract key points", "Translate this text"

🖼️ DIAGRAM/FLOWCHART:
- Patterns: boxes with arrows, process flows, architecture diagrams
- Suggest: "Explain this diagram", "What are the main components?", "Describe this workflow"

⚙️ SETTINGS/CONFIG:
- Patterns: configuration options, settings menus
- Suggest: "What do these settings do?", "Recommend configuration", "Explain these options"

STEP 3: Generate 3-4 HIGHLY SPECIFIC questions based on what you detected. Make questions actionable and directly related to the content.

STEP 4: ALWAYS include text extraction if any readable text is visible.

Format as valid JSON: {"name": "Short Name Here", "questions": [{"question": "...", "answer": "..."}]}

Be specific! Don't say "What is this?" - say "What's causing this IndexError?" or "How does this authentication function work?"."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this screenshot. Generate a short descriptive name, identify the content type, then generate specific, actionable questions with complete answers."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self.current_snip_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
                
                # Extract name and update memory
                snip_name = response_data.get('name', 'Unnamed Snip')
                if self.current_snip_id:
                    # Update the name in memory
                    for snip in self.memory['snips']:
                        if snip['id'] == self.current_snip_id:
                            snip['name'] = snip_name
                            break
                    self.save_memory()
                    # Refresh history list on main thread
                    self.root.after(0, self.refresh_history_list)
                
                # Extract questions
                suggestions_data = response_data.get('questions', [])
                
                # Cache the answers
                for item in suggestions_data:
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    if question and answer:
                        self.suggestion_cache[question] = answer
                
                # Save suggestions to disk for this snip
                self.save_suggestions_to_history(suggestions_data)
                
                # Update UI on main thread
                self.root.after(0, lambda: self.display_smart_suggestions(suggestions_data))
            else:
                # Fallback to default if parsing fails
                self.root.after(0, self.display_default_suggestions)
                
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            # Fallback to default suggestions
            self.root.after(0, self.display_default_suggestions)
    
    def display_smart_suggestions(self, suggestions_data):
        """Display the generated suggestions in UI"""
        # Clear loading
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.suggestions_frame, text="Suggested questions:", 
                bg='white', font=('Arial', 9, 'italic')).pack(anchor='w')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in suggestions_data:
            question = item.get('question', '')
            if question and question not in seen:
                seen.add(question)
                unique_suggestions.append(question)
        
        for question in unique_suggestions[:4]:  # Limit to 4
            btn = tk.Button(
                self.suggestions_frame,
                text=f"💡 {question}",
                command=lambda q=question: self.use_cached_suggestion(q),
                bg='#e3f2fd',
                font=('Arial', 9),
                anchor='w',
                relief=tk.FLAT,
                cursor='hand2'
            )
            btn.pack(fill=tk.X, pady=2)
    
    def display_default_suggestions(self):
        """Display default suggestions as fallback"""
        # Clear loading
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.suggestions_frame, text="Suggested actions:", 
                bg='white', font=('Arial', 9, 'italic')).pack(anchor='w')
        
        default_suggestions = [
            "What is this image showing?",
            "Extract all text from this image"
        ]
        
        for suggestion in default_suggestions:
            btn = tk.Button(
                self.suggestions_frame,
                text=f"💡 {suggestion}",
                command=lambda s=suggestion: self.use_suggestion(s),
                bg='#e3f2fd',
                font=('Arial', 9),
                anchor='w',
                relief=tk.FLAT,
                cursor='hand2'
            )
            btn.pack(fill=tk.X, pady=2)
    
    def use_cached_suggestion(self, question):
        """Use a suggestion with pre-generated answer"""
        # Add user message
        self.add_message("You", question)
        
        # Get cached answer
        answer = self.suggestion_cache.get(question)
        
        if answer:
            # Show cached answer immediately
            self.add_message("Assistant", answer)
        else:
            # Fallback: call API
            self.chat_input.delete(0, tk.END)
            self.chat_input.insert(0, question)
            self.send_message()
    
    def use_suggestion(self, suggestion):
        """Use a suggested question"""
        self.chat_input.delete(0, tk.END)
        self.chat_input.insert(0, suggestion)
        self.send_message()
    
    def send_message(self):
        """Send message to LLM"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        if not self.current_snip:
            messagebox.showwarning("No Snip", "Please take a snip first!")
            return
        
        if not self.config.get('openai_api_key'):
            messagebox.showerror("API Key Missing", "Please set your OpenAI API key in Settings!")
            return
        
        # Clear input
        self.chat_input.delete(0, tk.END)
        
        # Add user message to chat
        self.add_message("You", message)
        
        # Show loading indicator
        self.show_loading_indicator()
        
        # Track common questions
        self.track_question(message)
        
        # Call API in background
        thread = threading.Thread(target=self.call_llm, args=(message,))
        thread.daemon = True
        thread.start()
    
    def call_llm(self, message):
        """Call OpenAI API with vision"""
        try:
            client = OpenAI(api_key=self.config['openai_api_key'])
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes images and helps users with their snipped screenshots. Be concise and helpful."
                }
            ]
            
            # Add chat history
            for chat in self.chat_history[:-1]:  # Exclude the last one we just added
                if chat['role'] == 'user':
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": chat['content']}
                        ]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": chat['content']
                    })
            
            # Add current message with image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.current_snip_base64}"
                        }
                    }
                ]
            })
            
            response = client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Hide loading indicator and add response to chat
            self.root.after(0, self.hide_loading_indicator)
            self.root.after(0, lambda: self.add_message("Assistant", assistant_message))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self.hide_loading_indicator)
            self.root.after(0, lambda msg=error_msg: self.add_message("System", f"Error: {msg}", is_error=True))
    
    def show_loading_indicator(self):
        """Show animated loading indicator"""
        self.loading_dots = 0
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n🤖 Assistant: ", "assistant")
        self.chat_display.tag_config("assistant", foreground="#388E3C", font=('Arial', 10, 'bold'))
        
        # Mark where loading text starts
        self.loading_mark = self.chat_display.index("end-1c")
        self.chat_display.insert(tk.END, ".", "loading")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Start animation
        self.animate_loading()
    
    def animate_loading(self):
        """Animate the loading dots"""
        if self.loading_mark is None:
            return
        
        try:
            self.loading_dots = (self.loading_dots + 1) % 4
            dots = "." * max(1, self.loading_dots)
            
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(self.loading_mark, tk.END)
            self.chat_display.insert(tk.END, dots, "loading")
            self.chat_display.tag_config("loading", foreground="#999", font=('Arial', 10))
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
            
            # Schedule next animation frame
            self.loading_animation_id = self.root.after(500, self.animate_loading)
        except:
            pass
    
    def hide_loading_indicator(self):
        """Hide the loading indicator"""
        # Cancel animation
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        
        # Remove loading text
        if hasattr(self, 'loading_mark') and self.loading_mark:
            try:
                self.chat_display.config(state=tk.NORMAL)
                # Delete the "Assistant: ..." line
                line_start = self.chat_display.index(f"{self.loading_mark} linestart")
                self.chat_display.delete(line_start, tk.END)
                self.chat_display.config(state=tk.DISABLED)
            except:
                pass
            self.loading_mark = None
    
    def add_message(self, sender, content, is_error=False):
        """Add message to chat display"""
        # Add to history
        role = "user" if sender == "You" else "assistant"
        if not is_error:
            self.chat_history.append({"role": role, "content": content})
        
        # Display
        self.chat_display.config(state=tk.NORMAL)
        
        if is_error:
            self.chat_display.insert(tk.END, f"\n❌ {sender}: ", "error")
        else:
            tag = "user" if sender == "You" else "assistant"
            prefix = "👤" if sender == "You" else "🤖"
            self.chat_display.insert(tk.END, f"\n{prefix} {sender}: ", tag)
        
        self.chat_display.insert(tk.END, f"{content}\n")
        
        # Configure tags
        self.chat_display.tag_config("user", foreground="#1976D2", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("assistant", foreground="#388E3C", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("error", foreground="#D32F2F", font=('Arial', 10, 'bold'))
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Enable copy button if we have assistant messages
        if sender == "Assistant" and not is_error:
            self.copy_response_btn.config(state=tk.NORMAL)
    
    def update_chat_display(self):
        """Update chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.insert(1.0, "Chat with your snip below...\n")
        self.chat_display.config(state=tk.DISABLED)
        
        # Disable copy button when chat is cleared
        self.copy_response_btn.config(state=tk.DISABLED)
    
    def track_question(self, question):
        """Track common questions"""
        common = self.memory.get('common_questions', [])
        
        # Simple tracking - add if not exists
        if question not in common:
            common.append(question)
            common = common[-10:]  # Keep last 10
            self.memory['common_questions'] = common
            self.save_memory()
    
    def refresh_history_list(self):
        """Refresh the history listbox"""
        self.history_listbox.delete(0, tk.END)
        
        for snip in self.memory.get('snips', []):
            display_name = snip.get('name', 'Unnamed Snip')
            self.history_listbox.insert(tk.END, display_name)
    
    def on_history_select(self, event):
        """Handle history item selection"""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        snips = self.memory.get('snips', [])
        
        if index < len(snips):
            snip = snips[index]
            self.load_snip_from_history(snip)
    
    def load_snip_from_history(self, snip):
        """Load a snip from history"""
        image_path = snip.get('path')
        
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Snip image not found!")
            return
        
        try:
            # Load image
            image = Image.open(image_path)
            self.current_snip = image.copy()
            self.current_snip_id = snip.get('id')
            
            # Convert to base64 for API
            buffered = io.BytesIO()
            self.current_snip.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            self.current_snip_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Display image
            self.display_image(self.current_snip.copy())
            
            # Enable copy/save buttons
            self.copy_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            
            # Clear chat for new session
            self.chat_history = []
            self.update_chat_display()
            
            # Load cached suggestions if available
            suggestions_path = snip.get('suggestions_path')
            if suggestions_path and os.path.exists(suggestions_path):
                try:
                    with open(suggestions_path, 'r') as f:
                        suggestions_data = json.load(f)
                    
                    # Rebuild cache
                    self.suggestion_cache = {}
                    for item in suggestions_data:
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        if question and answer:
                            self.suggestion_cache[question] = answer
                    
                    # Display cached suggestions
                    self.display_smart_suggestions(suggestions_data)
                except Exception as e:
                    print(f"Error loading cached suggestions: {e}")
                    # Fallback to generating new ones
                    self.generate_suggestions()
            else:
                # No cached suggestions, generate new ones
                self.generate_suggestions()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load snip: {str(e)}")
    
    def on_history_rename(self, event):
        """Handle double-click to rename"""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        snips = self.memory.get('snips', [])
        
        if index < len(snips):
            snip = snips[index]
            current_name = snip.get('name', 'Unnamed Snip')
            
            # Simple dialog for rename
            new_name = tk.simpledialog.askstring(
                "Rename Snip",
                "Enter new name:",
                initialvalue=current_name,
                parent=self.root
            )
            
            if new_name and new_name.strip():
                snip['name'] = new_name.strip()
                self.save_memory()
                self.refresh_history_list()
                # Reselect the renamed item
                self.history_listbox.selection_set(index)
    
    def clear_history(self):
        """Clear all history"""
        if not messagebox.askyesno("Clear History", "Are you sure you want to clear all snip history?"):
            return
        
        # Delete all image and suggestion files
        for snip in self.memory.get('snips', []):
            image_path = snip.get('path')
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            suggestions_path = snip.get('suggestions_path')
            if suggestions_path and os.path.exists(suggestions_path):
                try:
                    os.remove(suggestions_path)
                except:
                    pass
        
        # Clear memory
        self.memory['snips'] = []
        self.save_memory()
        self.refresh_history_list()
        
        messagebox.showinfo("Success", "History cleared!")
    
    def show_chat_context_menu(self, event):
        """Show context menu on right-click in chat"""
        try:
            self.chat_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.chat_context_menu.grab_release()
    
    def copy_selected_chat(self):
        """Copy selected text from chat"""
        try:
            # Get selected text
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected_text:
                pyperclip.copy(selected_text)
                messagebox.showinfo("Success", "Selected text copied to clipboard!")
        except tk.TclError:
            messagebox.showwarning("No Selection", "Please select some text first!")
    
    def copy_selected_chat_kbd(self, event=None):
        """Copy selected text from chat (keyboard shortcut)"""
        try:
            # Get selected text
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected_text:
                pyperclip.copy(selected_text)
        except tk.TclError:
            pass  # No selection, ignore
    
    def copy_last_response(self):
        """Copy the last assistant response"""
        # Find the last assistant message
        for msg in reversed(self.chat_history):
            if msg['role'] == 'assistant':
                pyperclip.copy(msg['content'])
                messagebox.showinfo("Success", "Last response copied to clipboard!")
                return
        
        messagebox.showwarning("No Response", "No assistant response to copy!")
    
    def select_all_chat(self):
        """Select all text in chat"""
        self.chat_display.tag_add(tk.SEL, "1.0", tk.END)
        self.chat_display.mark_set(tk.INSERT, "1.0")
        self.chat_display.see(tk.INSERT)
    
    def open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x300")
        settings_window.resizable(False, False)
        
        # API Key
        tk.Label(settings_window, text="OpenAI API Key:", font=('Arial', 11, 'bold')).pack(pady=(20, 5), padx=20, anchor='w')
        
        api_key_frame = tk.Frame(settings_window)
        api_key_frame.pack(fill=tk.X, padx=20, pady=5)
        
        api_key_var = tk.StringVar(value=self.config.get('openai_api_key', ''))
        api_key_entry = tk.Entry(api_key_frame, textvariable=api_key_var, show='*', font=('Arial', 10))
        api_key_entry.pack(fill=tk.X)
        
        tk.Label(settings_window, text="Get your API key from: https://platform.openai.com/api-keys", 
                font=('Arial', 8), fg='#666').pack(padx=20, anchor='w')
        
        # Save button
        def save_settings():
            self.config['openai_api_key'] = api_key_var.get()
            self.config['model'] = self.model_var.get()
            self.save_config()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            settings_window.destroy()
        
        save_btn = tk.Button(
            settings_window,
            text="💾 Save Settings",
            command=save_settings,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10
        )
        save_btn.pack(pady=20)
        
        # Memory management
        tk.Label(settings_window, text="Memory Management:", font=('Arial', 11, 'bold')).pack(pady=(10, 5), padx=20, anchor='w')
        
        def clear_memory():
            if messagebox.askyesno("Clear Memory", "Are you sure you want to clear all memory?"):
                self.memory = {"snip_history": [], "common_questions": []}
                self.save_memory()
                messagebox.showinfo("Memory", "Memory cleared!")
        
        clear_btn = tk.Button(
            settings_window,
            text="🗑️ Clear Memory",
            command=clear_memory,
            font=('Arial', 10),
            padx=15,
            pady=5
        )
        clear_btn.pack(padx=20, anchor='w')
    
    def on_model_change(self, event):
        """Handle model change"""
        self.config['model'] = self.model_var.get()
        self.save_config()
    
    def on_delay_change(self, event):
        """Handle delay change"""
        self.config['capture_delay'] = self.delay_var.get()
        self.save_config()
    
    def show_countdown(self, seconds, callback):
        """Show countdown overlay before capture"""
        # Minimize main window
        self.root.iconify()
        
        # Create countdown window - small floating widget
        countdown_window = tk.Toplevel(self.root)
        countdown_window.overrideredirect(True)  # Remove window decorations
        countdown_window.attributes('-topmost', True)
        countdown_window.attributes('-alpha', 0.9)
        countdown_window.configure(bg='#2c3e50')
        
        # Position in top-right corner
        window_width = 300
        window_height = 200
        screen_width = countdown_window.winfo_screenwidth()
        x_position = screen_width - window_width - 20
        y_position = 20
        countdown_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Add border effect
        border_frame = tk.Frame(countdown_window, bg='#3498db', padx=3, pady=3)
        border_frame.pack(fill=tk.BOTH, expand=True)
        
        content_frame = tk.Frame(border_frame, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text="⏱️ Capture in...",
            font=('Arial', 14, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=(10, 5))
        
        # Countdown label
        countdown_label = tk.Label(
            content_frame,
            text=str(seconds),
            font=('Arial', 60, 'bold'),
            fg='#3498db',
            bg='#2c3e50'
        )
        countdown_label.pack(pady=10)
        
        # Info label
        info_label = tk.Label(
            content_frame,
            text="Press ESC to cancel",
            font=('Arial', 10),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        info_label.pack(pady=(5, 10))
        
        # Bind escape key to cancel
        def cancel_countdown(event=None):
            countdown_window.destroy()
            self.root.deiconify()
        
        countdown_window.bind('<Escape>', cancel_countdown)
        self.root.bind('<Escape>', cancel_countdown)  # Also bind to main window
        
        # Countdown animation
        def update_countdown(remaining):
            if remaining > 0:
                countdown_label.config(text=str(remaining))
                # Change color as countdown progresses
                if remaining <= 1:
                    countdown_label.config(fg='#e74c3c')  # Red
                elif remaining <= 2:
                    countdown_label.config(fg='#f39c12')  # Orange
                countdown_window.after(1000, lambda: update_countdown(remaining - 1))
            else:
                countdown_window.destroy()
                # Execute the callback
                callback()
        
        # Start countdown
        update_countdown(seconds)
    
    def copy_to_clipboard(self):
        """Copy current snip to clipboard"""
        if not self.current_snip:
            messagebox.showwarning("No Snip", "Please take a snip first!")
            return
        
        try:
            # Convert to Windows clipboard format
            output = io.BytesIO()
            self.current_snip.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove BMP header
            output.close()
            
            # Copy to clipboard using win32clipboard
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            
            messagebox.showinfo("Success", "Image copied to clipboard!")
        except ImportError:
            # Fallback: save to temp file and use pyperclip
            try:
                import tempfile
                temp_path = os.path.join(tempfile.gettempdir(), 'sniplm_temp.png')
                self.current_snip.save(temp_path)
                
                # Use PowerShell to copy image to clipboard
                import subprocess
                ps_command = f'Set-Clipboard -Path "{temp_path}"'
                subprocess.run(['powershell', '-command', ps_command], check=True)
                
                messagebox.showinfo("Success", "Image copied to clipboard!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not copy to clipboard: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy to clipboard: {str(e)}")
    
    def save_to_file(self):
        """Save current snip to file"""
        if not self.current_snip:
            messagebox.showwarning("No Snip", "Please take a snip first!")
            return
        
        # Get timestamp for default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"snip_{timestamp}.png"
        
        # Open save dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_snip.save(file_path)
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


class ScreenSelector:
    """Screen selection tool for capturing regions"""
    def __init__(self, parent, callback):
        self.callback = callback
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        
        # Capture screenshot FIRST to get actual dimensions
        self.screenshot = ImageGrab.grab()
        
        # Create fullscreen window
        self.window = tk.Toplevel(parent)
        self.window.attributes('-fullscreen', True)
        self.window.attributes('-alpha', 0.3)
        self.window.attributes('-topmost', True)
        
        # Force window update to get actual dimensions
        self.window.update()
        
        # Calculate scaling factor
        # Compare actual screen size to window size
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()
        screenshot_width, screenshot_height = self.screenshot.size
        
        self.scale_x = screenshot_width / window_width if window_width > 0 else 1.0
        self.scale_y = screenshot_height / window_height if window_height > 0 else 1.0
        
        # Canvas
        self.canvas = tk.Canvas(self.window, cursor="cross", bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Escape>', self.on_escape)
        self.window.bind('<Escape>', self.on_escape)
        
        # Instructions
        self.canvas.create_text(
            self.window.winfo_screenwidth() // 2,
            30,
            text="Click and drag to select area • ESC to cancel",
            font=('Arial', 16, 'bold'),
            fill='white'
        )
    
    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
    
    def on_mouse_move(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
    
    def on_mouse_up(self, event):
        if self.start_x is None:
            return
        
        # Get coordinates in canvas space
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Ensure we have a valid region
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.window.destroy()
            self.callback(None)
            return
        
        # Convert canvas coordinates to screenshot pixel coordinates
        # Apply scaling factor
        px1 = int(x1 * self.scale_x)
        py1 = int(y1 * self.scale_y)
        px2 = int(x2 * self.scale_x)
        py2 = int(y2 * self.scale_y)
        
        # Crop image using pixel coordinates
        cropped = self.screenshot.crop((px1, py1, px2, py2))
        
        # Close window
        self.window.destroy()
        
        # Callback
        self.callback(cropped)
    
    def on_escape(self, event=None):
        self.cancel()
    
    def cancel(self):
        self.window.destroy()
        self.callback(None)


if __name__ == "__main__":
    app = SnipLM()
    app.run()


