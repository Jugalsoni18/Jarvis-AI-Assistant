import sys
import os
import torch
import pyttsx3
import threading
import time
import subprocess
import webbrowser
import datetime
import psutil
import pyautogui
import platform
import json
import re
import cv2
import numpy as np
import sounddevice as sd
import requests
import socket
import pygetwindow as gw
import pyperclip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                           QWidget, QLabel, QTextEdit, QSystemTrayIcon, QMenu, QAction,
                           QComboBox, QHBoxLayout, QSlider, QTabWidget, QGridLayout,
                           QLineEdit, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
                           QGroupBox, QFrame, QScrollArea, QSplitter)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QTextCursor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QSettings, QSize
from transformers import AutoModelForCausalLM, AutoTokenizer# Add Gemini API import
import google.generativeai as genai
from dotenv import load_dotenv
import soundfile as sf
from gtts import gTTS
from playsound import playsound
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import simpleaudio as sa
import glob
import win32com.client

# Load .env for Gemini API key
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

WEATHER_API_KEY = os.environ.get("WEATHER_API") or os.environ.get("WEATHER_API_KEY")

def play_beep(frequency=800, duration=0.1):
    """Play a simple beep sound for user feedback"""
    try:
        # Generate a simple sine wave beep
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Play the beep
        sd.play(beep, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Could not play beep: {e}")

# Router function to detect local/system commands
LOCAL_COMMAND_PATTERNS = [
    r"\\bopen ", r"\\blaunch ", r"\\bplay ", r"\\bwhat time is it", r"\\bstart ", r"\\brun ", r"\\bexecute ", r"\\bshow me ", r"\\bbring up ", r"\\bload "
]

def is_local_command(query: str) -> bool:
    import re
    q = query.lower()
    for pat in LOCAL_COMMAND_PATTERNS:
        if re.search(pat, q):
            return True
    # Add more heuristics if needed
    return False

# Gemini API call

def ask_gemini(query: str) -> str:
    try:
        if not GEMINI_API_KEY:
            return "Gemini API key not set."
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(query)
        if hasattr(response, 'text'):
            return response.text.strip()
        return str(response)
    except Exception as e:
        return f"Error with Gemini API: {e}"

# Voice override detection

def get_model_override(query: str) -> str:
    q = query.lower()
    if q.startswith("use gemini"):
        return "gemini"
    if q.startswith("use local") or q.startswith("use tinyllama"):
        return "local"
    return ""

class EnhancedDesktopAssistant:
    """Enhanced AI Desktop Assistant with improved capabilities"""
    def save_model(self, path=None):
        """Save the model and tokenizer to the specified path"""
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            return "Model or tokenizer not found. Make sure the model is loaded first."

        try:
            save_path = path if path else "my_model"
            os.makedirs(save_path, exist_ok=True)

            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            return f"Model and tokenizer successfully saved to {save_path}"
        except Exception as e:
            return f"Error saving model: {e}"


    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", config_path="assistant_config.json"):
        """Initialize the assistant with default settings and configurations"""
        self.debug_mode = True
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.config_path = config_path
        self.load_config()
        self.engine.setProperty('rate', self.config.get("voice_rate", 150))
        self.available_voices = self.get_available_voices()
        self.set_voice(self.config.get("voice_id", None))

        # Speech recognition settings
        self.listening_timeout = self.config.get("listening_timeout", 5)
        self.phrase_time_limit = self.config.get("phrase_time_limit", 10)  # for 10 seconds
        self.silence_duration = self.config.get("silence_duration", 1.5)

        # Assistant settings
        self.wake_word = self.config.get("wake_word", "assistant")
        self.use_wake_word = self.config.get("use_wake_word", False)
        self.debug_mode = self.config.get("debug_mode", True)
        self.conversation_memory = []
        self.max_history_length = self.config.get("max_history_length", 10)

        # Add task automation settings
        self.automated_tasks = self.config.get("automated_tasks", {})
        self.task_shortcuts = self.config.get("task_shortcuts", {})
        self.user_preferences = self.config.get("user_preferences", {
            "default_browser": self.config.get("default_browser", "chrome"),
            "preferred_search_engine": self.config.get("preferred_search_engine", "google"),
            "theme": self.config.get("theme", "light"),
            "startup_apps": self.config.get("startup_apps", [])
        })

        # Load model
        try:
            if self.debug_mode:
                print(f"Loading {model_name} model and tokenizer...")
                print("Attempting to load tokenizer...")
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.debug_mode:
                print("Tokenizer loaded successfully")
                print("Attempting to load model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            if self.debug_mode:
                print("Model loaded successfully")
                device = next(self.model.parameters()).device
                print(f"Model is using device: {device}")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            print("Using fallback mode - basic command recognition only")

        # Detect system and setup platform-specific settings
        self.setup_system_info()
        self.setup_app_commands()
        self.setup_command_handlers()

        # Enhance application commands dictionary with more apps and platform-specific commands
        self.setup_app_commands()

        # Setup additional command handlers
        self.setup_command_handlers()

        # Enhanced system prompt with more capabilities
        self.setup_system_prompt()

        # Computer vision capabilities
        self.setup_vision_system()

        # Remote control capabilities (smartphone app via HTTP)
        self.remote_server_active = False

        # Setup file management
        self.recent_files = self.config.get("recent_files", [])
        self.favorite_folders = self.config.get("favorite_folders", [])

        # Setup clipboard manager
        self.clipboard_history = self.config.get("clipboard_history", [])
        self.max_clipboard_items = self.config.get("max_clipboard_items", 20)

        # Update startup time (used for uptime calculation)
        self.startup_time = time.time()

        print(f"Enhanced Desktop Assistant initialized on {self.os_type.capitalize()}")

        self.speaking = False
        self.stop_speaking_event = threading.Event()

    def load_config(self):
        """Load configuration from file or use defaults"""
        try:
            # Initialize all required attributes first to avoid attribute errors in save_config
            self.listening_timeout = 5
            self.phrase_time_limit = 10  # for 10 seconds
            self.silence_duration = 1.5


            # Initialize assistant settings
            self.wake_word = "jarvis"
            self.use_wake_word = True
            self.conversation_memory = []
            self.max_history_length = 10

            # Initialize task automation settings
            self.automated_tasks = {}
            self.task_shortcuts = {}

            # Initialize user preferences
            self.user_preferences = {
                "default_browser": "chrome",
                "preferred_search_engine": "google",
                "theme": "light",
                "startup_apps": []
            }

            # Initialize file management
            self.recent_files = []
            self.favorite_folders = []

            # Initialize clipboard manager
            self.clipboard_history = []
            self.max_clipboard_items = 20

            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                    if self.debug_mode:
                        print(f"Configuration loaded from {self.config_path}")
            else:
                self.config = {}
                if self.debug_mode:
                    print("No configuration file found, using defaults")
                # Create default config
                self.save_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = {}

    def save_config(self):
        """Save current configuration to file"""
        # Update configuration with current settings
        self.config.update({
            "listening_timeout": self.listening_timeout,
            "phrase_time_limit": self.phrase_time_limit,
            "silence_duration": self.silence_duration,
            "wake_word": self.wake_word,
            "use_wake_word": self.use_wake_word,
            "debug_mode": self.debug_mode,
            "voice_rate": self.engine.getProperty('rate'),
            "voice_id": self.engine.getProperty('voice'),
            "max_history_length": self.max_history_length,
            "automated_tasks": self.automated_tasks,
            "task_shortcuts": self.task_shortcuts,
            "user_preferences": self.user_preferences,
            "recent_files": self.recent_files,
            "favorite_folders": self.favorite_folders,
            "clipboard_history": self.clipboard_history,
            "max_clipboard_items": self.max_clipboard_items
        })

        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                if self.debug_mode:
                    print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def setup_system_info(self):
        """Setup system-specific information and capabilities"""
        # Detect operating system
        self.os_type = "windows" if os.name == "nt" else "mac" if sys.platform == "darwin" else "linux"

        # Get detailed system information
        self.system_info = {
            "os": self.os_type,
            "os_version": platform.version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "username": os.getlogin() if hasattr(os, 'getlogin') else os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            "total_memory": psutil.virtual_memory().total // (1024 * 1024),  # MB
            "cpu_count": psutil.cpu_count(),
            "screen_resolution": pyautogui.size()
        }

        if self.debug_mode:
            print(f"System info collected: {self.system_info}")

    def setup_app_commands(self):
        """Set up application commands dictionary with OS-specific entries"""
        # Common applications for all platforms
        common_apps = {
            "browser": self.user_preferences.get("default_browser", "chrome"),
            "text_editor": "notepad" if self.os_type == "windows" else "textedit" if self.os_type == "mac" else "gedit",
            "file_manager": "explorer" if self.os_type == "windows" else "finder" if self.os_type == "mac" else "nautilus"
        }

        # Windows-specific applications
        windows_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
            "firefox": "firefox.exe",
            "file explorer": "explorer.exe",
            "control panel": "control.exe",
            "paint": "mspaint.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "task manager": "taskmgr.exe",
            "settings": "ms-settings:",
            "photos": "ms-photos:",
            "mail": "olk.exe",
            "spotify": "spotify.exe",
            "discord": "discord.exe",
            "teams": "teams.exe",
            "visual studio code": "code.exe",
            "visual studio": "devenv.exe",
            "photoshop": "photoshop.exe",
            "illustrator": "illustrator.exe",
            "slack": "slack.exe",
            "zoom": "zoom.exe",
            "obs": "obs64.exe",
            "itunes": "itunes.exe",
            "vlc": "vlc.exe",
            "steam": "steam.exe",
            "epic games": "epicgameslauncher.exe",
            "origin": "origin.exe",
            "outlook": "olk.exe",
            "onenote": "onenote.exe",
            "skype": "skype.exe",
            "whatsapp": "whatsapp.exe",
            "telegram": "telegram.exe",
            "jupyter": "jupyter-notebook.exe",
            "python": "python.exe",
            "terminal": "wt.exe",  # Windows Terminal
            "vs code": "code.exe",
            "netflix": "netflix.exe",
            "command prompt": "cmd.exe"
        }

        # macOS-specific applications
        mac_apps = {
            "terminal": "open -a Terminal",
            "textedit": "open -a TextEdit",
            "safari": "open -a Safari",
            "chrome": "open -a 'Google Chrome'",
            "firefox": "open -a Firefox",
            "finder": "open -a Finder",
            "system preferences": "open -a 'System Preferences'",
            "mail": "open -a Mail",
            "photos": "open -a Photos",
            "music": "open -a Music",
            "calculator": "open -a Calculator",
            "calendar": "open -a Calendar",
            "notes": "open -a Notes",
            "pages": "open -a Pages",
            "numbers": "open -a Numbers",
            "keynote": "open -a Keynote",
            "preview": "open -a Preview",
            "spotify": "open -a Spotify",
            "discord": "open -a Discord",
            "slack": "open -a Slack",
            "zoom": "open -a 'zoom.us'",
            "visual studio code": "open -a 'Visual Studio Code'",
            "photoshop": "open -a 'Adobe Photoshop'",
            "illustrator": "open -a 'Adobe Illustrator'"
        }

        # Linux-specific applications
        linux_apps = {
            "terminal": "gnome-terminal",
            "firefox": "firefox",
            "chrome": "google-chrome",
            "nautilus": "nautilus",
            "gedit": "gedit",
            "calculator": "gnome-calculator",
            "settings": "gnome-control-center",
            "spotify": "spotify",
            "visual studio code": "code",
            "discord": "discord"
        }

        # Create the application commands dictionary
        self.app_commands = {
            "common": common_apps,
            "windows": windows_apps,
            "mac": mac_apps,
            "linux": linux_apps
        }

        # Create command shortcuts dictionary (easier to access)
        self.all_app_commands = {**common_apps}
        if self.os_type == "windows":
            self.all_app_commands.update(windows_apps)
        elif self.os_type == "mac":
            self.all_app_commands.update(mac_apps)
        else:  # linux
            self.all_app_commands.update(linux_apps)

        # Add custom application paths from config
        custom_apps = self.config.get("custom_applications", {})
        self.all_app_commands.update(custom_apps)

    def setup_command_handlers(self):
        """Setup specialized command handlers for different task categories"""
        # Command handlers dictionary - maps command categories to handler methods
        self.command_handlers = {
            "open": self.open_application,
            "search": self.search_web,
            "system": self.system_action,
            "file": self.file_operation,
            "media": self.media_control,
            "info": self.get_system_info,
            "automate": self.automate_task,
            "screenshot": self.take_screenshot,
            "type": self.type_text,
            "clipboard": self.clipboard_operation,
            "reminder": self.set_reminder,
            "timer": self.set_timer,
            "schedule": self.schedule_task,
            "email": self.email_operation,
            "translate": self.translate_text,
            "calculate": self.calculate,
            "weather": self.get_weather,
            "news": self.get_news,
            "note": self.note_operation,
            "calendar": self.calendar_operation,
            "contact": self.contact_operation,
            "message": self.send_message,
            "password": self.password_manager
        }

    # Add stub methods for other command handlers
    def automate_task(self, task_info: str) -> str:
        """Automate a task based on the provided information"""
        return "Task automation is not implemented yet."

    def take_screenshot(self, screenshot_type: str) -> str:
        """Take a screenshot of the specified type"""
        return "Screenshot functionality is not implemented yet."

    def type_text(self, text: str) -> str:
        """Type the specified text"""
        return "Text typing functionality is not implemented yet."

    def clipboard_operation(self, operation: str) -> str:
        """Perform clipboard operations"""
        return "Clipboard operations are not implemented yet."

    def set_reminder(self, reminder_info: str) -> str:
        """Set a reminder with the specified information"""
        return "Reminder functionality is not implemented yet."

    def set_timer(self, timer_info: str) -> str:
        """Set a timer with the specified information"""
        return "Timer functionality is not implemented yet."

    def schedule_task(self, task_info: str) -> str:
        """Schedule a task with the specified information"""
        return "Task scheduling is not implemented yet."

    def email_operation(self, email_info: str) -> str:
        """Perform email operations"""
        return "Email operations are not implemented yet."

    def translate_text(self, text_info: str) -> str:
        """Translate text"""
        return "Text translation is not implemented yet."

    def calculate(self, expression: str) -> str:
        """Calculate the result of an expression"""
        return "Calculation functionality is not implemented yet."

    def get_weather(self, location: str) -> str:
        global WEATHER_API_KEY
        print("[Weather] Using real OpenWeatherMap integration.")
        if not WEATHER_API_KEY:
            return "Weather API key not set. Please set WEATHER_API in your environment."
        if not location or location.strip() == "":
            return "Please specify a location for the weather report."
        try:
            weather = self._fetch_weather_data(location)
            if not weather:
                return f"Sorry, I couldn't retrieve weather information for {location}."
            desc = weather['weather'][0]['description'].capitalize()
            temp = weather['main']['temp']
            feels = weather['main']['feels_like']
            humidity = weather['main']['humidity']
            wind = weather['wind']['speed']
            city = weather.get('name', location)
            country = weather.get('sys', {}).get('country', '')
            return (f"The weather in {city}, {country} is {desc}. "
                    f"Temperature: {temp}°C, feels like {feels}°C. "
                    f"Humidity: {humidity}%. Wind speed: {wind} m/s.")
        except Exception as e:
            return f"Error retrieving weather: {e}"

    def _fetch_weather_data(self, location: str) -> dict:
        global WEATHER_API_KEY
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None

    def get_news(self, topic: str) -> str:
        """Get news about the specified topic"""
        return "News retrieval is not implemented yet."

    def note_operation(self, note_info: str) -> str:
        """Perform note operations"""
        return "Note operations are not implemented yet."

    def calendar_operation(self, calendar_info: str) -> str:
        """Perform calendar operations"""
        return "Calendar operations are not implemented yet."

    def contact_operation(self, contact_info: str) -> str:
        """Perform contact operations"""
        return "Contact operations are not implemented yet."

    def send_message(self, message_info: str) -> str:
        """Send a message"""
        return "Message sending is not implemented yet."

    def password_manager(self, password_info: str) -> str:
        """Perform password management operations"""
        return "Password management is not implemented yet."

    def manage_clipboard_history(self, text: str) -> None:
        """Add text to clipboard history"""
        if text and text not in self.clipboard_history:
            self.clipboard_history.append(text)
            if len(self.clipboard_history) > self.max_clipboard_items:
                self.clipboard_history.pop(0)

    def add_recent_file(self, path: str) -> None:
        """Add a file to recent files list"""
        if path and path not in self.recent_files:
            self.recent_files.append(path)
            if len(self.recent_files) > 20:  # Keep only 20 most recent files
                self.recent_files.pop(0)

    def get_system_info(self, info_type: str) -> str:
        """Get system information based on the requested type"""
        info_type = info_type.lower().strip()

        try:
            if info_type == "full" or info_type == "all":
                # Format all system information
                info_text = f"System Information:\n"
                info_text += f"- OS: {self.system_info['system']} {self.system_info['os_version']}\n"
                info_text += f"- Computer: {self.system_info['hostname']}\n"
                info_text += f"- User: {self.system_info['username']}\n"
                info_text += f"- Processor: {self.system_info['processor']}\n"
                info_text += f"- CPU Cores: {self.system_info['cpu_count']}\n"
                info_text += f"- Total Memory: {self.system_info['total_memory']} MB\n"
                info_text += f"- Python Version: {self.system_info['python_version']}\n"
                info_text += f"- Screen Resolution: {self.system_info['screen_resolution'].width}x{self.system_info['screen_resolution'].height}\n"

                # Add current usage
                cpu_usage = psutil.cpu_percent(interval=0.5)
                memory = psutil.virtual_memory()
                memory_used_percent = memory.percent
                memory_used = memory.used // (1024 * 1024)  # MB

                info_text += f"- Current CPU Usage: {cpu_usage}%\n"
                info_text += f"- Current Memory Usage: {memory_used} MB ({memory_used_percent}%)\n"

                # Add uptime
                uptime_seconds = time.time() - self.startup_time
                uptime_minutes = uptime_seconds // 60
                uptime_hours = uptime_minutes // 60
                uptime_minutes %= 60

                info_text += f"- Assistant Uptime: {int(uptime_hours)} hours, {int(uptime_minutes)} minutes\n"

                return info_text

            elif info_type == "cpu":
                cpu_usage = psutil.cpu_percent(interval=1)
                return f"Current CPU usage: {cpu_usage}%"

            elif info_type == "memory" or info_type == "ram":
                memory = psutil.virtual_memory()
                memory_used = memory.used // (1024 * 1024)  # MB
                memory_total = memory.total // (1024 * 1024)  # MB
                memory_percent = memory.percent

                return f"Memory usage: {memory_used} MB / {memory_total} MB ({memory_percent}%)"

            elif info_type == "disk":
                disk = psutil.disk_usage('/')
                disk_used = disk.used // (1024 * 1024 * 1024)  # GB
                disk_total = disk.total // (1024 * 1024 * 1024)  # GB
                disk_percent = disk.percent

                return f"Disk usage: {disk_used} GB / {disk_total} GB ({disk_percent}%)"

            elif info_type == "network":
                # Get network information
                net_io = psutil.net_io_counters()
                bytes_sent = net_io.bytes_sent // (1024 * 1024)  # MB
                bytes_recv = net_io.bytes_recv // (1024 * 1024)  # MB

                return f"Network usage: Sent {bytes_sent} MB, Received {bytes_recv} MB"

            elif info_type == "battery":
                if hasattr(psutil, "sensors_battery"):
                    battery = psutil.sensors_battery()
                    if battery:
                        percent = battery.percent
                        power_plugged = battery.power_plugged
                        status = "Charging" if power_plugged else "Discharging"

                        return f"Battery: {percent}% ({status})"
                    else:
                        return "No battery detected (desktop computer)"
                else:
                    return "Battery information not available on this system"

            elif info_type == "temperature":
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        temp_info = "System Temperatures:\n"
                        for name, entries in temps.items():
                            for entry in entries:
                                temp_info += f"- {name}: {entry.current}°C\n"
                        return temp_info
                    else:
                        return "No temperature sensors detected"
                else:
                    return "Temperature information not available on this system"

            elif info_type == "processes":
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent']
                    })

                # Sort by CPU usage
                processes.sort(key=lambda x: x['cpu'], reverse=True)

                result = "Top 5 processes by CPU usage:\n"
                for i, proc in enumerate(processes[:5], 1):
                    result += f"{i}. {proc['name']} (PID: {proc['pid']}): CPU {proc['cpu']}%, Memory {proc['memory']:.1f}%\n"

                return result

            elif info_type == "uptime":
                uptime_seconds = time.time() - self.startup_time
                uptime_minutes = uptime_seconds // 60
                uptime_hours = uptime_minutes // 60
                uptime_days = uptime_hours // 24

                uptime_hours %= 24
                uptime_minutes %= 60

                if uptime_days > 0:
                    return f"Assistant uptime: {int(uptime_days)} days, {int(uptime_hours)} hours, {int(uptime_minutes)} minutes"
                else:
                    return f"Assistant uptime: {int(uptime_hours)} hours, {int(uptime_minutes)} minutes"

            else:
                return f"Unknown system information type: {info_type}"

        except Exception as e:
            error_msg = f"Error getting system information: {e}"
            print(error_msg)
            return error_msg

    def setup_system_prompt(self):
        """Set up an enhanced system prompt with more capabilities"""
        self.system_prompt = f"""You are an advanced AI desktop assistant that can control the computer.
        You respond to user requests in a helpful, concise manner.
        You can answer questions, provide information, open applications, and help with various system tasks.
        You're running on a {self.os_type.capitalize()} computer with {self.system_info['os_version']} as user {self.system_info['username']}.

        Your capabilities include:
        1. Opening applications and websites
        2. Searching the web
        3. Performing system actions (shutdown, restart, etc.)
        4. File management (creating, moving, deleting files/folders)
        5. Taking screenshots and capturing images
        6. Typing text and automating keyboard/mouse actions
        7. Managing clipboard content
        8. Setting reminders and timers
        9. Getting system information (CPU, memory usage, etc.)
        10. Controlling media playback
        11. Taking notes and creating quick documents
        12. Checking weather and news
        13. Basic calculations and conversions
        14. Sending messages and emails
        15. Managing calendar events and contacts
        16. Password management assistance
        17. Language translation
        18. Voice modulation and customization

        When asked to perform any of these tasks, respond with a special command format
        starting with !SYSTEM! followed by the command type and value (e.g., "!SYSTEM!open:notepad" or "!SYSTEM!search:weather today").

        Command types include: open, search, system, file, media, info, automate, screenshot, type, clipboard, reminder, timer,
        schedule, email, translate, calculate, weather, news, note, calendar, contact, message, password

        If you need to check system information before performing a task, use !SYSTEM!info:requirement
        to check if the necessary capability is available.

        Keep your responses brief, friendly, and focus on completing the user's request efficiently.
        When appropriate, offer helpful tips or suggest related actions that might help the user.
        """

    def setup_vision_system(self):
        """Setup computer vision capabilities"""
        try:
            # Check if OpenCV is working
            self.vision_enabled = True

            # Create a flag to track if webcam is currently in use
            self.webcam_in_use = False

            if self.debug_mode:
                print("Vision system initialized successfully")

        except Exception as e:
            self.vision_enabled = False
            print(f"Error initializing vision system: {e}")

    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        voices = []
        for voice in self.engine.getProperty('voices'):
            voices.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender
            })
        return voices

    def set_voice(self, voice_id=None):
        """Set the TTS voice by ID"""
        if not voice_id and len(self.available_voices) > 0:
            # Default to first voice
            voice_id = self.available_voices[0]['id']

        if voice_id:
            try:
                self.engine.setProperty('voice', voice_id)
                if self.debug_mode:
                    print(f"Set voice to: {voice_id}")
                return True
            except Exception as e:
                print(f"Error setting voice: {e}")
                return False
        return False

    def list_microphones(self) -> list:
        """List all available microphones with their indices and names."""
        mic_list = []
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    mic_list.append((i, device['name']))
        except Exception as e:
            print(f"Could not list microphones: {e}")
        return mic_list

    def set_microphone(self, device_index: int = None):
        """Set the microphone device index for listening. Always use default microphone."""
        self.selected_mic_index = None  # Always use default
        if self.debug_mode:
            print("Using default microphone.")

    def tune_sensitivity(self, pause_threshold: float = 1.2):
        """Tune sensitivity for pause handling."""
        self.silence_duration = pause_threshold
        if self.debug_mode:
            print(f"Set silence_duration to {self.silence_duration}")

    def adjust_for_ambient_noise(self, duration=1.5):
        """Calculates a dynamic silence threshold based on ambient noise."""
        print("Adjusting for ambient noise, please be quiet for a moment...")
        fs = 16000
        chunk_size = 1024
        mic_index = getattr(self, 'selected_mic_index', None)
        
        noise_levels = []
        try:
            with sd.InputStream(samplerate=fs, channels=1, device=mic_index, dtype='float32', blocksize=chunk_size) as stream:
                for _ in range(int(duration * fs / chunk_size)):
                    audio_chunk, _ = stream.read(chunk_size)
                    noise_levels.append(np.sqrt(np.mean(audio_chunk**2)))
            
            # The multiplier can be tuned if detection is too sensitive or not sensitive enough
            dynamic_threshold = np.mean(noise_levels) * 2.0 
            # Set a minimum floor to avoid issues in very quiet environments
            self.dynamic_silence_threshold = max(dynamic_threshold, 0.008)

            if self.debug_mode:
                print(f"Dynamic silence threshold calibrated to: {self.dynamic_silence_threshold:.4f}")

        except Exception as e:
            print(f"Could not automatically adjust for ambient noise: {e}. Using default threshold.")

    def listen_command(self) -> str:
        """Listen for a command using Google Speech Recognition and return the recognized string."""
        recognizer = sr.Recognizer()
        mic_index = getattr(self, 'selected_mic_index', None)
        try:
            with sr.Microphone(device_index=mic_index) as source:
                print("Listening for command... (please start speaking after the beep)")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=self.listening_timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    print("⏰ Timeout: No speech detected. Please try again and speak promptly after the beep.")
                    return "none"
                print("Audio captured, recognizing with Google Speech Recognition...")
                try:
                    user_input = recognizer.recognize_google(audio, language='en-US')
                    print(f"User said: {user_input}")
                    return user_input
                except sr.UnknownValueError:
                    print("⚠️ Google Speech Recognition could not understand the audio. Try again.")
                    return "none"
                except sr.RequestError as e:
                    print(f"[SpeechRecognition] API error: {e}")
                    return "none"
        except Exception as e:
            print(f"[SpeechRecognition] Error in listen_command: {e}")
            import traceback
            traceback.print_exc()
            return "none"

    def generate_response(self, user_input: str) -> str:
        clean_input = user_input.strip()
        lower_input = clean_input.lower()
        print(f"Generate response for: '{clean_input}' (lowercase: '{lower_input}')")

        # Voice override
        override = get_model_override(clean_input)
        if override == "gemini":
            response = ask_gemini(clean_input.replace("use gemini", "", 1).strip())
            # self.speak(response)  # Silent mode: do not speak
            return response
        elif override == "local":
            response = self._tinyllama_respond(clean_input.replace("use local", "", 1).replace("use tinyllama", "", 1).strip())
            # self.speak(response)  # Silent mode: do not speak
            return response

        # Add to conversation memory
        if len(self.conversation_memory) >= self.max_history_length:
            self.conversation_memory.pop(0)
        self.conversation_memory.append({"role": "user", "content": clean_input})

        # Direct command detection
        task_response = self.detect_direct_task(lower_input)
        if task_response:
            print(f"Direct task detected, response: '{task_response}'")
            self.conversation_memory.append({"role": "assistant", "content": task_response})
            # self.speak(task_response)  # Silent mode: do not speak
            return task_response

        # Route based on command type
        if is_local_command(clean_input):
            response = self._tinyllama_respond(clean_input)
            if not response or len(response.strip()) < 3 or response.lower() in ["i don't know", "not sure", "", "i didn't understand what you are saying. can you repeat or want help with anything else"]:
                response = ask_gemini(clean_input)
        else:
            response = ask_gemini(clean_input)
            if not response or len(response.strip()) < 3:
                response = self._tinyllama_respond(clean_input)

        # self.speak(response)  # Silent mode: do not speak
        return response

    # Helper for TinyLlama response (original logic)
    def _tinyllama_respond(self, user_input: str) -> str:
        # This is the original generate_response logic, minus the new routing
        clean_input = user_input.strip()
        lower_input = clean_input.lower()
        # Add to conversation memory
        if len(self.conversation_memory) >= self.max_history_length:
            self.conversation_memory.pop(0)
        self.conversation_memory.append({"role": "user", "content": clean_input})
        # Direct command detection
        task_response = self.detect_direct_task(lower_input)
        if task_response:
            self.conversation_memory.append({"role": "assistant", "content": task_response})
            return task_response
        # If we get here and it looks like a command but wasn't recognized, return the default message
        open_patterns = ["open", "launch", "start", "run", "execute", "show me", "bring up", "load"]
        if any(pattern in lower_input for pattern in open_patterns):
            default_response = "I didn't understand what you are saying. can you repeat or want help with anything else"
            self.conversation_memory.append({"role": "assistant", "content": default_response})
            return default_response
        # Using language model for response generation with conversation history
        try:
            prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            for message in self.conversation_memory[-5:]:
                if message["role"] == "user":
                    prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                else:
                    prompt += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
            if self.debug_mode:
                print(f"Generating with prompt: {prompt[:200]}...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]
            )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if self.debug_mode:
                print(f"Raw model response: {response}")
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            self.conversation_memory.append({"role": "assistant", "content": response.strip()})
            return response.strip()
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(error_msg)
            self.conversation_memory.append({"role": "assistant", "content": error_msg})
            return error_msg

    def detect_direct_task(self, lower_input: str) -> Optional[str]:
        """
        Detects and processes direct task commands using regex for robust parsing.
        This allows for more natural language commands, ignoring filler words.
        """
        # Define command patterns with regex. The key is a command type, and the value is a regex pattern.
        # The patterns are designed to capture intent and entities from natural language.
        command_patterns = {
            "open_application": r'\b(open|launch|start|run|execute|show me|bring up|load)\s+(.+)',
            "search_web": r'\b(search for|search|look up|find|google)\s+(.+)',
            "system_action": r'\b(shutdown|restart|reboot|sleep|lock|log out|sign out|hibernate|cancel shutdown|cancel restart)\b',
            "media_control": r'\b(play|pause|next track|previous track|stop music|volume up|volume down|louder|quieter)\b',
            "get_system_info": r'\b(?:what is|what\'s|check|show me|get)\s+(?:my\s+|the\s+)?(cpu(?: usage)?|memory(?: usage)?|ram(?: usage)?|disk(?: usage)?|network|battery|temperature|processes|uptime|system info(?:rmation)?)\b',
            "get_time": r'\b(what time is it|what\'s the time|the time)\b',
            "get_date": r'\b(what\'s the date|what is today|today\'s date|the current date)\b',
            "save_model": r'\b(save model|save the model|save trained model|save the trained model|save tinyllama|save the tinyllama|save tiny llama|save the tiny llama)\b',
            "compose_email": r'\b(write mail to|email|send mail to)\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        }

        # Match and execute commands based on the patterns
        for command_type, pattern in command_patterns.items():
            match = re.search(pattern, lower_input)
            if not match:
                continue

            # --- Application Opening ---
            if command_type == "open_application":
                app_name = match.group(2).strip()
                # Clean up filler words from the end of the app name
                filler_words = [" for me", " please", " app"]
                for filler in filler_words:
                    if app_name.endswith(filler):
                        app_name = app_name[:-len(filler)].strip()
                result = self.open_application(app_name)
                return f"I'll try to open {app_name}. {result}"

            # --- Web Search ---
            elif command_type == "search_web":
                search_terms = match.group(2).strip()
                result = self.search_web(search_terms)
                return f"I'm searching for '{search_terms}'. {result}"

            # --- System Actions ---
            elif command_type == "system_action":
                action = match.group(1).strip()
                # Normalize action keywords
                if action == 'reboot': action = 'restart'
                if action in ['log out', 'sign out']: action = 'logout'
                if action in ['cancel shutdown', 'cancel restart']: action = 'cancel shutdown'
                return f"I'll {action} your computer. {self.system_action(action)}"

            # --- Media Controls ---
            elif command_type == "media_control":
                action = match.group(1).strip()
                # Normalize media keywords
                action_map = {
                    "next track": "next",
                    "previous track": "previous",
                    "stop music": "stop",
                    "volume up": "volume_up",
                    "volume down": "volume_down",
                    "louder": "volume_up",
                    "quieter": "volume_down"
                }
                final_action = action_map.get(action, action)
                return self.media_control(final_action)

            # --- System Info ---
            elif command_type == "get_system_info":
                info_type = match.group(2).strip()
                # Normalize info_type keywords
                if 'cpu' in info_type: info_type = 'cpu'
                elif 'memory' in info_type or 'ram' in info_type: info_type = 'memory'
                elif 'disk' in info_type: info_type = 'disk'
                elif 'system info' in info_type: info_type = 'full'
                return self.get_system_info(info_type)

            # --- Get Time ---
            elif command_type == "get_time":
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                return f"The current time is {current_time}."

            # --- Get Date ---
            elif command_type == "get_date":
                # Return the current date in YYYY-MM-DD format
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                return f"Today's date is {current_date}."

            # --- Save Model ---
            elif command_type == "save_model":
                custom_path = None
                path_indicators = ["to ", "in ", "at ", "as "]
                for indicator in path_indicators:
                    if indicator in lower_input:
                        parts = lower_input.split(indicator, 1)
                        if len(parts) > 1:
                            potential_path = parts[1].strip()
                            if potential_path:
                                custom_path = potential_path
                                break
                result = self.save_model(custom_path)
                return f"I'll save the trained TinyLlama model for you. {result}"

            # --- Compose Email ---
            elif command_type == "compose_email":
                recipient = match.group(2).strip()
                try:
                    import win32com.client
                    outlook = win32com.client.Dispatch("Outlook.Application")
                    mail = outlook.CreateItem(0)
                    mail.To = recipient
                    mail.Display()
                    return f"Opened Outlook draft to {recipient}."
                except Exception as e:
                    print(f"Failed to open Outlook draft: {e}")
                    return "Failed to open Outlook draft."

        # If no regex pattern matched after checking all, return None
        return None

    def execute_system_command(self, command_type: str, command_value: str) -> str:
        """Execute a system command by routing to the appropriate handler"""
        command_type = command_type.strip().lower()
        command_value = command_value.strip()

        if self.debug_mode:
            print(f"Executing system command: {command_type}:{command_value}")

        # Call the appropriate command handler
        if command_type in self.command_handlers:
            return self.command_handlers[command_type](command_value)
        else:
            return f"Unknown command type: {command_type}"

    def speak(self, text: str):
        """Speak the given text using Google gTTS and pydub/simpleaudio, with interruption support."""
        def tts_worker():
            try:
                tts = gTTS(text=text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_path = fp.name
                    tts.save(temp_path)
                audio = AudioSegment.from_file(temp_path, format="mp3")
                self._play_obj = sa.play_buffer(
                    audio.raw_data,
                    num_channels=audio.channels,
                    bytes_per_sample=audio.sample_width,
                    sample_rate=audio.frame_rate
                )
                self.speaking = True
                self._play_obj.wait_done()
                os.remove(temp_path)
            except Exception as e:
                print(f"Error in speak: {e}")
            finally:
                self.speaking = False
                self._play_obj = None

        # Stop any current speech
        self.stop_speaking()
        tts_thread = threading.Thread(target=tts_worker)
        tts_thread.start()

    def stop_speaking(self):
        """Stop speaking immediately if currently playing audio."""
        if hasattr(self, '_play_obj') and self._play_obj is not None:
            try:
                self._play_obj.stop()
            except Exception:
                pass
        self.speaking = False
        self._play_obj = None

    def listen_for_stop_during_speech(self):
        """Listen for the 'stop' command while speaking. If heard, stop speaking."""
        while self.speaking:
            command = self.listen_command()
            if command and command.lower().strip() == "stop":
                print("Stop command detected during speech. Stopping TTS.")
                self.stop_speaking()
                break
            # If not speaking anymore, break
            if not self.speaking:
                break
            time.sleep(0.1)

    def _find_executable(self, app_name):
        """Search for an executable matching app_name in common Windows locations."""
        app_name = app_name.lower().replace('.exe', '').strip()
        search_dirs = [
            os.environ.get('ProgramFiles', r'C:\Program Files'),
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            os.path.expanduser(r'~\Desktop'),
            os.path.expanduser(r'~\AppData\Local'),
            os.path.expanduser(r'~\AppData\Roaming'),
        ]
        # Search for .exe files
        for base in search_dirs:
            for root, dirs, files in os.walk(base):
                for file in files:
                    if file.lower() == app_name + '.exe':
                        return os.path.join(root, file)
        # Search Start Menu shortcuts
        start_menu_dirs = [
            os.path.expandvars(r'%APPDATA%\Microsoft\Windows\Start Menu\Programs'),
            os.path.expandvars(r'%PROGRAMDATA%\Microsoft\Windows\Start Menu\Programs'),
        ]
        shell = win32com.client.Dispatch('WScript.Shell')
        for base in start_menu_dirs:
            for root, dirs, files in os.walk(base):
                for file in files:
                    if file.lower().endswith('.lnk') and app_name in file.lower():
                        shortcut = os.path.join(root, file)
                        try:
                            target = shell.CreateShortCut(shortcut).Targetpath
                            if target.lower().endswith('.exe'):
                                return target
                        except Exception:
                            continue
        return None

    def open_application(self, app_name: str) -> str:
        app_name = app_name.lower().strip()
        print(f"Attempting to open application: '{app_name}'")

        # Special case for WhatsApp
        if app_name in ["whatsapp", "open whatsapp", "whatsapp.com", "www.whatsapp.com"]:
            url = "https://web.whatsapp.com/"
            print(f"Opening WhatsApp Web: {url}")
            webbrowser.open(url)
            return "Opened WhatsApp Web"

        # Special case for YouTube
        if app_name in ["youtube", "open youtube", "youtube.com", "www.youtube.com"]:
            url = "https://www.youtube.com"
            print(f"Opening YouTube: {url}")
            webbrowser.open(url)
            return "Opened YouTube"

        # Try to find and open a matching .exe or shortcut first
        exe_path = self._find_executable(app_name)
        if exe_path:
            print(f"Found executable: {exe_path}. Launching...")
            try:
                subprocess.Popen([exe_path])
                return f"Launched {app_name} from {exe_path}"
            except Exception as e:
                print(f"Failed to launch {exe_path}: {e}")
                return f"Failed to launch {app_name}: {e}"

        try:
            # Check if this is a website URL or special protocol
            if app_name.startswith(("http://", "https://", "www.", "ms-")):
                if app_name.startswith("www.") and not app_name.startswith(("http://", "https://")):
                    app_name = "https://" + app_name

                print(f"Opening as website or protocol: {app_name}")
                webbrowser.open(app_name)
                return f"Opened {app_name}"

            # Check if app name ends with common web domains
            web_domains = [".com", ".org", ".net", ".edu", ".gov", ".io", ".co"]
            is_website = any(app_name.endswith(domain) for domain in web_domains)

            if is_website:
                url = "https://" + app_name
                print(f"Opening as website with domain: {url}")
                webbrowser.open(url)
                return f"Opened website: {url}"

            # Special cases for apps that might need special handling
            special_cases = {
                "netflix": {
                    "url": "ms-windows-store://pdp/?ProductId=9WZDNCRFJ3TJ",
                    "message": "Opening Netflix from Microsoft Store"
                },
                "whatsapp": {
                    "url": "https://web.whatsapp.com/",
                    "message": "Opening WhatsApp Web in your browser"
                }
            }

            if app_name in special_cases:
                case = special_cases[app_name]
                print(f"{case['message']}: {case['url']}")
                webbrowser.open(case['url'])
                return case['message']

            # Windows-specific application handling
            if self.os_type == "windows":
                # Common applications with known paths
                common_app_paths = {
                    "chrome": [
                        "start chrome",
                        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
                    ],
                    "spotify": [
                        "start spotify",
                        r"C:\\Users\\%USERNAME%\\AppData\\Roaming\\Spotify\\Spotify.exe",
                        r"C:\\Program Files\\WindowsApps\\SpotifyAB.SpotifyMusic_*\\Spotify.exe"
                    ],
                    "firefox": [
                        "start firefox",
                        r"C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                        r"C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe"
                    ],
                    "edge": [
                        "start msedge",
                        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
                    ],
                    "notepad": ["notepad"],
                    "calculator": ["calc"],
                    "word": ["start winword"],
                    "excel": ["start excel"],
                    "powerpoint": ["start powerpnt"],
                    "cmd": ["start cmd"],
                    "command prompt": ["start cmd"],
                    "powershell": ["start powershell"],
                    "explorer": ["explorer"],
                    "control panel": ["control"],
                    "whatsapp": ["WhatsApp", "start WhatsApp", r"C:\\Users\\%USERNAME%\\AppData\\Local\\WhatsApp\\WhatsApp.exe"],
                    "vs code": ["code", "start code", r"C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"],
                    "visual studio code": ["code", "start code", r"C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"],
                    "netflix": ["ms-windows-store://pdp/?ProductId=9WZDNCRFJ3TJ", "start netflix", r"C:\\Program Files\\WindowsApps\\Netflix*\\Netflix.exe"]
                }

                # Check for special cases first
                for app_key, commands in common_app_paths.items():
                    if app_name == app_key or app_key in app_name:
                        print(f"Found special case for {app_key}")

                        # Try each command in order
                        for cmd in commands:
                            try:
                                # If it's a path that might contain wildcards, try to resolve it
                                if '*' in cmd:
                                    import glob
                                    expanded_path = cmd.replace('%USERNAME%', os.getenv('USERNAME', ''))
                                    matching_paths = glob.glob(expanded_path)
                                    if matching_paths:
                                        cmd = matching_paths[0]
                                    else:
                                        continue  # Skip if no matching paths

                                # Check if it's a file path that needs to exist
                                if cmd.startswith(('C:', 'D:', 'E:', 'F:')):
                                    cmd = cmd.replace('%USERNAME%', os.getenv('USERNAME', ''))
                                    if not os.path.exists(cmd):
                                        print(f"Path does not exist: {cmd}")
                                        continue
                                    # Use quotes around path to handle spaces
                                    cmd = f'"{cmd}"'

                                print(f"Trying command: {cmd}")
                                if cmd.startswith("start "):
                                    # Use os.system for start commands
                                    os.system(cmd)
                                else:
                                    # Use subprocess for direct commands
                                    subprocess.Popen(cmd, shell=True)

                                return f"Launched {app_key} using command: {cmd}"
                            except Exception as e:
                                print(f"Failed to launch with command {cmd}: {e}")

                        # If we get here, all commands failed
                        return f"Attempted to launch {app_key} but all methods failed"

                # Check if app name is in our dictionary
                print(f"Checking if '{app_name}' is in app dictionary")
                print(f"Available apps: {list(self.all_app_commands.keys())}")
                if app_name in self.all_app_commands:
                    app_command = self.all_app_commands[app_name]
                    print(f"Found app command: '{app_command}'")

                    try:
                        # Try using the start command first
                        print(f"Trying with START command: start {app_command}")
                        os.system(f"start {app_command}")
                        return f"Launched {app_name} using START command"
                    except Exception as start_error:
                        print(f"START command failed: {start_error}")
                        try:
                            # Fall back to direct subprocess
                            print(f"Trying direct subprocess: {app_command}")
                            subprocess.Popen(app_command, shell=True)
                            return f"Launched {app_name} using direct command"
                        except Exception as direct_error:
                            print(f"Direct command failed: {direct_error}")
                            return f"Failed to launch {app_name}"

                # Last resort: try with START command and direct command
                try:
                    print(f"Last resort: trying START {app_name}")
                    os.system(f"start {app_name}")

                    # Also try direct command as a fallback
                    try:
                        subprocess.Popen(app_name, shell=True)
                    except Exception:
                        pass  # Ignore errors from the fallback attempt

                    return f"Attempted to launch {app_name}. If the application doesn't open, please make sure it's installed on your system."
                except Exception as e:
                    print(f"START command failed: {e}")
                    return f"Failed to launch {app_name}. Please make sure the application is installed on your system."

            # macOS application handling
            elif self.os_type == "mac":
                # Check if app name is in our dictionary
                if app_name in self.all_app_commands:
                    app_command = self.all_app_commands[app_name]
                    print(f"Found app command for Mac: '{app_command}'")
                    os.system(app_command)
                    return f"Launched {app_name}"
                else:
                    # Try to open using the generic open command
                    os.system(f"open -a '{app_name}'")
                    return f"Attempted to launch {app_name}"

            # Linux application handling
            else:
                # Check if app name is in our dictionary
                if app_name in self.all_app_commands:
                    app_command = self.all_app_commands[app_name]
                    print(f"Found app command for Linux: '{app_command}'")
                    subprocess.Popen([app_command], shell=True)
                    return f"Launched {app_name}"
                else:
                    # Try to run directly
                    subprocess.Popen([app_name], shell=True)
                    return f"Attempted to launch {app_name}"

        except Exception as e:
            error_msg = f"Error opening application: {e}"
            print(error_msg)
            return error_msg

        # Special case for Outlook and olk.exe
        if app_name in ["outlook", "open outlook", "olk.exe", "microsoft outlook", "olk.exe"]:
            possible_paths = [
                r"C:\Program Files\WindowsApps\Microsoft.Office.Desktop.Outlook_*\olk.exe",
                r"C:\Program Files\Microsoft Office\root\Office16\olk.exe",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\olk.exe",
                r"C:\Program Files\Microsoft Office\Office16\olk.exe",
                r"C:\Program Files (x86)\Microsoft Office\Office16\olk.exe",
                r"C:\Program Files\Microsoft Office\Office15\olk.exe",
                r"C:\Program Files (x86)\Microsoft Office\Office15\olk.exe",
                r"C:\Program Files\WindowsApps\olk.exe",
                r"C:\Program Files\olk.exe",
                r"C:\Program Files (x86)\olk.exe"
            ]
            found = False
            for path in possible_paths:
                if "*" in path:
                    for match in glob.glob(path):
                        if os.path.exists(match):
                            print(f"Opening Outlook: {match}")
                            subprocess.Popen([match])
                            return "Opened Microsoft Outlook (olk.exe)"
                elif os.path.exists(path):
                    print(f"Opening Outlook: {path}")
                    subprocess.Popen([path])
                    return "Opened Microsoft Outlook (olk.exe)"
            # Fallback to COM automation
            try:
                import win32com.client
                outlook = win32com.client.Dispatch("Outlook.Application")
                mail = outlook.CreateItem(0)
                mail.Display()
                return "Opened Outlook via COM"
            except Exception as e:
                print("olk.exe or Outlook executable not found in standard locations.")
                return "olk.exe or Outlook executable not found."

        # Special case for Windows Calendar
        if app_name in ["calendar", "open calendar", "windows calendar"]:
            try:
                import os
                os.startfile("calendar:")
                return "Opened Windows Calendar"
            except Exception as e:
                print(f"Failed to open Calendar: {e}")
                return "Failed to open Calendar"

        # Special case for Google Calendar
        if app_name in ["google calendar", "open google calendar"]:
            url = "https://calendar.google.com"
            webbrowser.open(url)
            return "Opened Google Calendar"

        # Special case for calendar.exe
        if app_name in ["calendar.exe"]:
            possible_paths = [
                r"C:\Windows\System32\cal.exe",
                r"C:\Windows\System32\calendar.exe"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Opening Calendar: {path}")
                    subprocess.Popen([path])
                    return "Opened Calendar.exe"
            print("calendar.exe not found in standard locations.")
            return "calendar.exe not found."

        # Special case for Gmail
        if app_name in ["gmail", "open gmail", "gmail.com", "www.gmail.com"]:
            url = "https://mail.google.com"
            print(f"Opening Gmail: {url}")
            webbrowser.open(url)
            return "Opened Gmail"

    def search_web(self, query: str) -> str:
        """Search the web with the user's preferred search engine"""
        search_engine = self.user_preferences.get("preferred_search_engine", "google").lower()

        search_urls = {
            "google": "https://www.google.com/search?q=",
            "bing": "https://www.bing.com/search?q=",
            "yahoo": "https://search.yahoo.com/search?p=",
            "duckduckgo": "https://duckduckgo.com/?q=",
            "youtube": "https://www.youtube.com/results?search_query="
        }

        try:
            # Get the search URL or default to Google
            search_url = search_urls.get(search_engine, search_urls["google"])

            # Encode the query and create the full URL
            encoded_query = query.replace(" ", "+")
            full_url = f"{search_url}{encoded_query}"

            # Open the browser with the search URL
            webbrowser.open(full_url)

            return f"Searching for '{query}' using {search_engine.capitalize()}"

        except Exception as e:
            error_msg = f"Error searching the web: {e}"
            print(error_msg)
            return error_msg

    def system_action(self, action: str) -> str:
        """Perform system actions like shutdown, restart, etc."""
        action = action.lower().strip()

        try:
            if action == "shutdown":
                if self.os_type == "windows":
                    os.system("shutdown /s /t 60")
                    return "Shutting down in 60 seconds. Use 'cancel shutdown' to abort."
                elif self.os_type == "mac":
                    os.system("sudo shutdown -h +1")
                    return "Shutting down in 60 seconds."
                else:  # Linux
                    os.system("sudo shutdown -h +1")
                    return "Shutting down in 60 seconds."

            elif action == "restart" or action == "reboot":
                if self.os_type == "windows":
                    os.system("shutdown /r /t 60")
                    return "Restarting in 60 seconds. Use 'cancel restart' to abort."
                elif self.os_type == "mac":
                    os.system("sudo shutdown -r +1")
                    return "Restarting in 60 seconds."
                else:  # Linux
                    os.system("sudo shutdown -r +1")
                    return "Restarting in 60 seconds."

            elif action == "cancel shutdown" or action == "cancel restart":
                if self.os_type == "windows":
                    os.system("shutdown /a")
                    return "Shutdown or restart cancelled."
                elif self.os_type == "mac":
                    os.system("sudo killall shutdown")
                    return "Shutdown or restart cancelled."
                else:  # Linux
                    os.system("sudo shutdown -c")
                    return "Shutdown or restart cancelled."

            elif action == "sleep":
                if self.os_type == "windows":
                    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
                    return "Putting the computer to sleep."
                elif self.os_type == "mac":
                    os.system("pmset sleepnow")
                    return "Putting the computer to sleep."
                else:  # Linux
                    os.system("systemctl suspend")
                    return "Putting the computer to sleep."

            elif action == "lock":
                if self.os_type == "windows":
                    os.system("rundll32.exe user32.dll,LockWorkStation")
                    return "Locking the computer."
                elif self.os_type == "mac":
                    os.system("/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend")
                    return "Locking the computer."
                else:  # Linux
                    os.system("gnome-screensaver-command -l")
                    return "Locking the computer."

            elif action == "logout":
                if self.os_type == "windows":
                    os.system("shutdown -l")
                    return "Logging out."
                elif self.os_type == "mac":
                    os.system("osascript -e 'tell application \"System Events\" to log out'")
                    return "Logging out."
                else:  # Linux
                    os.system("gnome-session-quit --logout --no-prompt")
                    return "Logging out."

            elif action == "hibernate":
                if self.os_type == "windows":
                    os.system("shutdown /h")
                    return "Hibernating the computer."
                elif self.os_type == "mac":
                    return "Hibernation is not directly supported on macOS. Use sleep instead."
                else:  # Linux
                    os.system("systemctl hibernate")
                    return "Hibernating the computer."

            else:
                return f"Unknown system action: {action}"

        except Exception as e:
            error_msg = f"Error performing system action: {e}"
            print(error_msg)
            return error_msg

    def file_operation(self, operation: str) -> str:
        """Perform file operations like creating, moving, deleting files"""
        # Parse the operation string to extract the command and parameters
        parts = operation.split(":", 1)

        if len(parts) < 2:
            return "Invalid file operation format. Use 'command:parameters'"

        command = parts[0].lower().strip()
        parameters = parts[1].strip()

        try:
            if command == "create":
                # Format: create:type:path:content
                # Example: create:file:C:/temp/test.txt:Hello World
                # or: create:folder:C:/temp/new_folder
                sub_parts = parameters.split(":", 2)

                if len(sub_parts) < 2:
                    return "Invalid create format. Use 'create:type:path[:content]'"

                create_type = sub_parts[0].lower().strip()
                path = sub_parts[1].strip()

                if create_type == "file":
                    content = sub_parts[2] if len(sub_parts) > 2 else ""
                    with open(path, 'w') as f:
                        f.write(content)

                    # Add to recent files
                    self.add_recent_file(path)
                    return f"Created file: {path}"

                elif create_type == "folder" or create_type == "directory":
                    os.makedirs(path, exist_ok=True)
                    return f"Created folder: {path}"
                else:
                    return f"Unknown create type: {create_type}"

            elif command == "delete":
                # Format: delete:path
                path = parameters

                if os.path.isfile(path):
                    os.remove(path)

                    # Remove from recent files if present
                    if path in self.recent_files:
                        self.recent_files.remove(path)

                    return f"Deleted file: {path}"

                elif os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                    return f"Deleted folder: {path}"
                else:
                    return f"Path not found: {path}"

            elif command == "copy":
                # Format: copy:source:destination
                sub_parts = parameters.split(":", 1)

                if len(sub_parts) < 2:
                    return "Invalid copy format. Use 'copy:source:destination'"

                source = sub_parts[0].strip()
                destination = sub_parts[1].strip()

                if os.path.isfile(source):
                    import shutil
                    shutil.copy2(source, destination)

                    # Add to recent files
                    self.add_recent_file(destination)
                    return f"Copied file from {source} to {destination}"

                elif os.path.isdir(source):
                    import shutil
                    shutil.copytree(source, destination)
                    return f"Copied folder from {source} to {destination}"
                else:
                    return f"Source path not found: {source}"

            elif command == "move":
                # Format: move:source:destination
                sub_parts = parameters.split(":", 1)

                if len(sub_parts) < 2:
                    return "Invalid move format. Use 'move:source:destination'"

                source = sub_parts[0].strip()
                destination = sub_parts[1].strip()

                if os.path.exists(source):
                    import shutil
                    shutil.move(source, destination)

                    # Update recent files if needed
                    if source in self.recent_files:
                        self.recent_files.remove(source)
                        self.add_recent_file(destination)

                    return f"Moved from {source} to {destination}"
                else:
                    return f"Source path not found: {source}"

            elif command == "rename":
                # Format: rename:source:new_name
                sub_parts = parameters.split(":", 1)

                if len(sub_parts) < 2:
                    return "Invalid rename format. Use 'rename:source:new_name'"

                source = sub_parts[0].strip()
                new_name = sub_parts[1].strip()
                destination = os.path.join(os.path.dirname(source), new_name)

                if os.path.exists(source):
                    os.rename(source, destination)

                    # Update recent files if needed
                    if source in self.recent_files:
                        self.recent_files.remove(source)
                        self.add_recent_file(destination)

                    return f"Renamed {source} to {new_name}"
                else:
                    return f"Source path not found: {source}"

            elif command == "list" or command == "dir":
                # Format: list:path[:filter]
                sub_parts = parameters.split(":", 1)
                path = sub_parts[0].strip()
                file_filter = sub_parts[1].strip() if len(sub_parts) > 1 else None

                if os.path.isdir(path):
                    files = os.listdir(path)

                    if file_filter:
                        import fnmatch
                        files = [f for f in files if fnmatch.fnmatch(f, file_filter)]

                    if not files:
                        return f"No items found in {path}" + (f" matching {file_filter}" if file_filter else "")

                    result = f"Contents of {path}"
                    if file_filter:
                        result += f" (filtered by {file_filter})"
                    result += ":\\n"

                    for i, file in enumerate(files[:20], 1):  # Limit to 20 items
                        full_path = os.path.join(path, file)
                        type_indicator = "📁" if os.path.isdir(full_path) else "📄"
                        result += f"{i}. {type_indicator} {file}\\n"

                    if len(files) > 20:
                        result += f"... and {len(files) - 20} more items"

                    return result
                else:
                    return f"Directory not found: {path}"

            elif command == "open":
                # Format: open:path
                path = parameters

                if os.path.exists(path):
                    if self.os_type == "windows":
                        os.startfile(path)
                    elif self.os_type == "mac":
                        subprocess.Popen(["open", path])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", path])

                    # Add to recent files if it's a file
                    if os.path.isfile(path):
                        self.add_recent_file(path)

                    return f"Opened {path}"
                else:
                    return f"Path not found: {path}"

            elif command == "read":
                # Format: read:path[:lines]
                sub_parts = parameters.split(":", 1)
                path = sub_parts[0].strip()
                line_count = int(sub_parts[1].strip()) if len(sub_parts) > 1 else None

                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        if line_count:
                            lines = [next(f) for _ in range(line_count)]
                            content = ''.join(lines)
                        else:
                            content = f.read()

                    # Add to recent files
                    self.add_recent_file(path)

                    # Limit output size
                    max_display = 500  # Characters
                    if len(content) > max_display:
                        content = content[:max_display] + "...\\n(content truncated)"

                    return f"Content of {path}:\\n\\n{content}"
                else:
                    return f"File not found: {path}"

            elif command == "write" or command == "append":
                # Format: write:path:content or append:path:content
                sub_parts = parameters.split(":", 1)

                if len(sub_parts) < 2:
                    return f"Invalid {command} format. Use '{command}:path:content'"

                path = sub_parts[0].strip()
                content = sub_parts[1].strip()

                mode = 'w' if command == "write" else 'a'
                with open(path, mode, encoding='utf-8') as f:
                    f.write(content)

                # Add to recent files
                self.add_recent_file(path)

                operation_name = "Wrote to" if command == "write" else "Appended to"
                return f"{operation_name} file: {path}"

            else:
                return f"Unknown file operation: {command}"

        except Exception as e:
            error_msg = f"Error in file operation: {e}"
            print(error_msg)
            return error_msg

    def add_recent_file(self, path: str) -> None:
        """Add a file to the recent files list"""
        # Remove if already exists (to refresh position)
        if path in self.recent_files:
            self.recent_files.remove(path)

        # Add to the front of the list
        self.recent_files.insert(0, path)

        # Limit the size of recent files list
        max_recent = 20
        if len(self.recent_files) > max_recent:
            self.recent_files = self.recent_files[:max_recent]

    def media_control(self, command: str) -> str:
        """Control media playback (play, pause, next, previous, volume)"""
        command = command.lower().strip()

        try:
            if self.os_type == "windows":
                import pynput.keyboard as keyboard
                kb = keyboard.Controller()

                if command == "play" or command == "pause" or command == "playpause":
                    kb.press(keyboard.Key.media_play_pause)
                    kb.release(keyboard.Key.media_play_pause)
                    return "Toggled play/pause"

                elif command == "next":
                    kb.press(keyboard.Key.media_next)
                    kb.release(keyboard.Key.media_next)
                    return "Skipped to next track"

                elif command == "previous" or command == "prev":
                    kb.press(keyboard.Key.media_previous)
                    kb.release(keyboard.Key.media_previous)
                    return "Went to previous track"

                elif command == "stop":
                    kb.press(keyboard.Key.media_volume_mute)
                    kb.release(keyboard.Key.media_volume_mute)
                    return "Toggled mute"

                elif command == "volume_up" or command == "louder":
                    kb.press(keyboard.Key.media_volume_up)
                    kb.release(keyboard.Key.media_volume_up)
                    return "Increased volume"

                elif command == "volume_down" or command == "quieter":
                    kb.press(keyboard.Key.media_volume_down)
                    kb.release(keyboard.Key.media_volume_down)
                    return "Decreased volume"

                elif command.startswith("volume_set:"):
                    # Not directly possible with standard keyboard controls
                    return "Volume level setting is not supported on Windows through this interface"

                else:
                    return f"Unknown media command: {command}"

            elif self.os_type == "mac":
                if command == "play" or command == "pause" or command == "playpause":
                    os.system("osascript -e 'tell application \"Music\" to playpause'")
                    return "Toggled play/pause"

                elif command == "next":
                    os.system("osascript -e 'tell application \"Music\" to next track'")
                    return "Skipped to next track"

                elif command == "previous" or command == "prev":
                    os.system("osascript -e 'tell application \"Music\" to previous track'")
                    return "Went to previous track"

                elif command == "stop":
                    os.system("osascript -e 'tell application \"Music\" to stop'")
                    return "Stopped playback"

                elif command == "volume_up" or command == "louder":
                    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
                    return "Increased volume"

                elif command == "volume_down" or command == "quieter":
                    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
                    return "Decreased volume"

                elif command.startswith("volume_set:"):
                    try:
                        level = int(command.split(":", 1)[1].strip())
                        level = max(0, min(100, level))  # Ensure it's between 0-100
                        os.system(f"osascript -e 'set volume output volume {level}'")
                        return f"Set volume to {level}%"
                    except ValueError:
                        return "Invalid volume level. Please provide a number between 0-100."

                else:
                    return f"Unknown media command: {command}"

            else:  # Linux
                # Linux commands often depend on the desktop environment
                # These are generic commands that might work on many distributions

                if command == "play" or command == "pause" or command == "playpause":
                    os.system("playerctl play-pause")
                    return "Toggled play/pause"

                elif command == "next":
                    os.system("playerctl next")
                    return "Skipped to next track"

                elif command == "previous" or command == "prev":
                    os.system("playerctl previous")
                    return "Went to previous track"

                elif command == "stop":
                    os.system("playerctl stop")
                    return "Stopped playback"

                elif command == "volume_up" or command == "louder":
                    os.system("pactl set-sink-volume @DEFAULT_SINK@ +5%")
                    return "Increased volume"

                elif command == "volume_down" or command == "quieter":
                    os.system("pactl set-sink-volume @DEFAULT_SINK@ -5%")
                    return "Decreased volume"

                elif command.startswith("volume_set:"):
                    try:
                        level = int(command.split(":", 1)[1].strip())
                        level = max(0, min(100, level))  # Ensure it's between 0-100
                        os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {level}%")
                        return f"Set volume to {level}%"
                    except ValueError:
                        return "Invalid volume level. Please provide a number between 0-100."

                else:
                    return f"Unknown media command: {command}"

        except Exception as e:
            error_msg = f"Error in media control: {e}"
            print(error_msg)
            return error_msg

# Main entry point to run the assistant
if __name__ == "__main__":
    try:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        assistant = EnhancedDesktopAssistant()
        print("Jarvis AI Desktop Assistant is ready!")
        print("I will listen for your voice commands, transcribe them, and execute the corresponding actions.")
        print("Say 'exit' or 'quit' to end the session.")
        print("-" * 50)

        while True:
            print("\n🎤 Recording...")
            play_beep(frequency=800, duration=0.1)
            transcription = assistant.listen_command()
            play_beep(frequency=600, duration=0.1)

            if transcription == "none":
                continue

            # Step 3: Check for exit command (word boundary match)
            exit_phrases = ["exit", "quit", "bye", "stop", "end", "jarvis exit"]
            transcript = transcription.lower()
            if any(re.search(rf'\\b{re.escape(phrase)}\\b', transcript) for phrase in exit_phrases):
                print("👋 Exit command received. Shutting down.")
                assistant.speak("Goodbye!")
                break

            # Step 4: Execute the command
            print("⚡ Executing...")
            try:
                response = assistant.generate_response(transcription)
                print(f"🤖 Jarvis: {response}")
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                print(f"❌ {error_msg}")
                assistant.speak(error_msg)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Shutting down.")
        assistant.speak("Goodbye!")
    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


