import os
import uuid
import tempfile
import warnings
import io
import datetime

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template_string, request, jsonify, send_file, session, redirect, url_for
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv
from PIL import Image

# --- AUTHENTICATION & DB IMPORTS ---
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)

# Security Config
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key_change_in_prod")

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = "" 
    print("Warning: GOOGLE_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)

# Configure MongoDB
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(mongo_uri)
    db = client.cropsense_db
    users_collection = db.users
    print(f"✅ Connected to MongoDB at {mongo_uri}")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    users_collection = None

# Global Temp Directory for Audio
TEMP_DIR = tempfile.gettempdir()

# --- HTML/CSS/JS FRONTEND ---
# We use Jinja2 logic ({% if user %}) to switch between Login and Chat views
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CropSense AI | Ultimate Agri-Tech</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --primary-accent: #00ff88;
            --secondary-accent: #00b894;
            --recording-color: #ff4757;
            --text-color: #ffffff;
            --dark-overlay: rgba(5, 20, 10, 0.85);
            --error-color: #ff6b6b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Outfit', sans-serif; }

        body {
            background: url("https://images.unsplash.com/photo-1625246333195-5519a495d026?q=80&w=2574&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            color: var(--text-color);
            overflow-x: hidden;
        }

        body::before {
            content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: var(--dark-overlay); z-index: -1; backdrop-filter: blur(3px);
        }

        /* --- AUTH STYLES --- */
        .auth-container {
            display: flex; justify-content: center; align-items: center; height: 100vh;
        }
        .auth-box {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            padding: 3rem;
            border-radius: 24px;
            width: 100%; max-width: 420px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .auth-title { font-size: 2rem; margin-bottom: 0.5rem; color: var(--primary-accent); }
        .auth-subtitle { color: #ccc; margin-bottom: 2rem; font-size: 0.9rem; }
        
        .form-group { margin-bottom: 1.5rem; text-align: left; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-size: 0.9rem; color: #ddd; }
        .auth-input {
            width: 100%; background: rgba(0,0,0,0.3); border: 1px solid var(--glass-border);
            padding: 14px; color: white; border-radius: 12px; font-size: 1rem; outline: none; transition: 0.3s;
        }
        .auth-input:focus { border-color: var(--primary-accent); background: rgba(0,0,0,0.5); }
        
        .auth-btn {
            width: 100%; background: var(--primary-accent); color: #05140a;
            padding: 14px; border-radius: 12px; border: none; font-weight: 700; font-size: 1rem;
            cursor: pointer; transition: 0.3s; margin-top: 1rem;
        }
        .auth-btn:hover { transform: translateY(-2px); box-shadow: 0 0 20px rgba(0, 255, 136, 0.4); }
        
        .switch-auth { margin-top: 1.5rem; font-size: 0.9rem; color: #aaa; }
        .switch-auth a { color: var(--primary-accent); text-decoration: none; font-weight: 600; cursor: pointer; }
        .switch-auth a:hover { text-decoration: underline; }

        .hidden { display: none; }
        .error-msg { color: var(--error-color); font-size: 0.85rem; margin-top: 5px; min-height: 20px; }

        /* --- APP LAYOUT (Existing Styles) --- */
        .container {
            max-width: 1200px; margin: 0 auto; padding: 2rem;
            display: grid; grid-template-columns: 300px 1fr; gap: 2rem; height: 100vh;
        }
        .sidebar {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border); border-radius: 24px; padding: 2rem;
            display: flex; flex-direction: column; gap: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .logo-area { text-align: center; margin-bottom: 1rem; }
        .logo-icon { font-size: 3rem; color: var(--primary-accent); text-shadow: 0 0 20px rgba(0, 255, 136, 0.4); }
        .control-group label { display: block; margin-bottom: 0.5rem; font-size: 0.9rem; color: #ccc; }
        select, .toggle-btn {
            width: 100%; background: rgba(0, 0, 0, 0.3); border: 1px solid var(--glass-border);
            color: white; padding: 12px; border-radius: 12px; outline: none; cursor: pointer; transition: 0.3s;
        }
        select:hover { border-color: var(--primary-accent); }
        .main-content { display: flex; flex-direction: column; gap: 1.5rem; height: calc(100vh - 4rem); }
        .glass-panel {
            background: var(--glass-bg); backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border); border-radius: 24px; padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2); position: relative; overflow: hidden;
        }
        .chat-container { flex-grow: 1; overflow-y: auto; padding-right: 10px; display: flex; flex-direction: column; gap: 1rem; }
        .chat-container::-webkit-scrollbar { width: 6px; }
        .chat-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
        
        .message { display: flex; gap: 1rem; align-items: flex-start; max-width: 80%; animation: fadeIn 0.3s ease; }
        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0; }
        .avatar.ai { background: linear-gradient(135deg, #11998e, #38ef7d); box-shadow: 0 0 15px rgba(56, 239, 125, 0.4); }
        .avatar.user { background: rgba(255,255,255,0.2); }
        .bubble {
            background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 18px;
            border-top-left-radius: 2px; line-height: 1.6; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.05);
        }
        .bubble strong { color: var(--primary-accent); font-weight: 700; }
        .bubble ul { margin-left: 20px; margin-top: 5px; margin-bottom: 5px; }
        .message.user .bubble {
            background: rgba(0, 255, 136, 0.1); border-color: rgba(0, 255, 136, 0.2); border-radius: 18px; border-top-right-radius: 2px;
        }
        
        .input-area {
            display: flex; gap: 1rem; align-items: center; background: rgba(0,0,0,0.2);
            padding: 0.5rem; border-radius: 50px; border: 1px solid var(--glass-border);
        }
        .file-upload-label { cursor: pointer; padding: 10px; color: var(--primary-accent); transition: 0.3s; }
        .file-upload-label:hover { transform: scale(1.1); text-shadow: 0 0 10px var(--primary-accent); }
        #imageInput { display: none; }
        #textInput { flex-grow: 1; background: transparent; border: none; color: white; padding: 10px; font-size: 1rem; outline: none; }
        .action-btn {
            background: var(--primary-accent); color: #000; border: none; width: 45px; height: 45px;
            border-radius: 50%; cursor: pointer; transition: 0.3s; display: flex; align-items: center; justify-content: center;
        }
        .action-btn:hover { transform: scale(1.1); box-shadow: 0 0 15px var(--primary-accent); }
        #micBtn.listening { background: var(--recording-color); color: white; animation: pulse 1.5s infinite; }
        .image-preview { max-width: 100px; max-height: 100px; border-radius: 10px; display: none; border: 2px solid var(--primary-accent); margin-right: 10px; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 71, 87, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); } }
        
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; height: auto; }
            .sidebar { padding: 1rem; }
            .main-content { height: 600px; }
        }
    </style>
</head>
<body>

    {% if not user %}
    <!-- === LOGIN / REGISTER VIEW === -->
    <div class="auth-container">
        <div class="auth-box">
            <div class="logo-icon"><i class="fa-solid fa-leaf"></i></div>
            <h2 class="auth-title">CropSense AI</h2>
            <p class="auth-subtitle">Login to access advanced agri-intelligence</p>
            
            <!-- Login Form -->
            <form id="loginForm" onsubmit="handleAuth(event, 'login')">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" name="username" class="auth-input" required placeholder="Enter username">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" class="auth-input" required placeholder="Enter password">
                </div>
                <div id="loginError" class="error-msg"></div>
                <button type="submit" class="auth-btn">Sign In</button>
                <div class="switch-auth">
                    Don't have an account? <a onclick="toggleForms()">Sign Up</a>
                </div>
            </form>

            <!-- Register Form -->
            <form id="registerForm" class="hidden" onsubmit="handleAuth(event, 'register')">
                <div class="form-group">
                    <label>Choose Username</label>
                    <input type="text" name="username" class="auth-input" required placeholder="Choose username">
                </div>
                <div class="form-group">
                    <label>Choose Password</label>
                    <input type="password" name="password" class="auth-input" required placeholder="Choose password">
                </div>
                <div id="registerError" class="error-msg"></div>
                <button type="submit" class="auth-btn">Create Account</button>
                <div class="switch-auth">
                    Already have an account? <a onclick="toggleForms()">Sign In</a>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        function toggleForms() {
            document.getElementById('loginForm').classList.toggle('hidden');
            document.getElementById('registerForm').classList.toggle('hidden');
            document.querySelectorAll('.error-msg').forEach(e => e.innerText = '');
        }

        async function handleAuth(e, type) {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const errorDiv = document.getElementById(`${type}Error`);
            const btn = form.querySelector('button');
            
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
            errorDiv.innerText = '';

            try {
                const res = await fetch(`/auth/${type}`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                
                if (data.success) {
                    window.location.reload();
                } else {
                    errorDiv.innerText = data.message || "An error occurred";
                    btn.innerText = type === 'login' ? 'Sign In' : 'Create Account';
                }
            } catch (err) {
                errorDiv.innerText = "Connection Failed. Check Server.";
                btn.innerText = type === 'login' ? 'Sign In' : 'Create Account';
            }
        }
    </script>

    {% else %}
    <!-- === MAIN CHAT VIEW (Protected) === -->
    <div class="container">
        <!-- SIDEBAR -->
        <aside class="sidebar">
            <div class="logo-area">
                <div class="logo-icon"><i class="fa-solid fa-leaf"></i></div>
                <h2>CropSense AI</h2>
                <p style="opacity: 0.7; font-size: 0.8rem;">Welcome, {{ user }}</p>
            </div>

            <div class="control-group">
                <label><i class="fa-solid fa-language"></i> Language</label>
                <select id="languageSelect">
                    <option value="en">English</option>
                    <option value="hi">Hindi (हिंदी)</option>
                    <option value="es">Spanish (Español)</option>
                    <option value="fr">French (Français)</option>
                    <option value="ta">Tamil (தமிழ்)</option>
                    <option value="te">Telugu (తెలుగు)</option>
                </select>
            </div>

            <div class="control-group">
                <label><i class="fa-solid fa-volume-high"></i> Audio Response</label>
                <select id="audioToggle">
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                </select>
            </div>
            
            <a href="/logout" style="text-decoration: none;">
                <button class="toggle-btn" style="background: rgba(255, 71, 87, 0.2); color: #ff4757; border-color: #ff4757;">
                    <i class="fa-solid fa-right-from-bracket"></i> Logout
                </button>
            </a>

            <div style="margin-top: auto; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="font-size: 0.8rem; opacity: 0.5;"><i class="fa-solid fa-circle-info"></i> v2.5 Auth Edition</p>
            </div>
        </aside>

        <!-- MAIN INTERFACE -->
        <main class="main-content">
            <div class="glass-panel" style="flex-grow: 1; display: flex; flex-direction: column;">
                <div class="chat-container" id="chatContainer">
                    <div class="message ai">
                        <div class="avatar ai"><i class="fa-solid fa-robot"></i></div>
                        <div class="bubble">
                            Hello {{ user }}! I am CropSense AI. 🌱<br>
                            I'm ready to help with your crop analysis.
                        </div>
                    </div>
                </div>
            </div>

            <div class="glass-panel" style="padding: 0.8rem;">
                <div style="display: flex; align-items: center;">
                    <img id="preview" class="image-preview">
                    <div class="input-area" style="flex-grow: 1;">
                        <label for="imageInput" class="file-upload-label">
                            <i class="fa-solid fa-camera"></i>
                        </label>
                        <input type="file" id="imageInput" accept="image/*">
                        
                        <input type="text" id="textInput" placeholder="Ask about crop health..." autocomplete="off">
                        
                        <button id="micBtn" class="action-btn" title="Speak"><i class="fa-solid fa-microphone"></i></button>
                        <button id="sendBtn" class="action-btn" style="margin-left: 5px;"><i class="fa-solid fa-paper-plane"></i></button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- MAIN APP SCRIPT -->
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const langSelect = document.getElementById('languageSelect');
        const audioToggle = document.getElementById('audioToggle');

        marked.setOptions({ breaks: true, gfm: true });

        // Voice Input
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            const langMap = { 'en': 'en-US', 'hi': 'hi-IN', 'es': 'es-ES', 'fr': 'fr-FR', 'ta': 'ta-IN', 'te': 'te-IN' };

            recognition.onstart = () => { micBtn.classList.add('listening'); textInput.placeholder = "Listening..."; };
            recognition.onend = () => { micBtn.classList.remove('listening'); textInput.placeholder = "Ask about crop health..."; };
            recognition.onresult = (event) => { textInput.value = event.results[0][0].transcript; };
            micBtn.addEventListener('click', () => { recognition.lang = langMap[langSelect.value] || 'en-US'; recognition.start(); });
        } else { micBtn.style.display = 'none'; }

        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => { preview.src = e.target.result; preview.style.display = 'block'; };
                reader.readAsDataURL(file);
            }
        });

        async function sendMessage() {
            const text = textInput.value;
            const file = imageInput.files[0];
            const lang = langSelect.value;
            const audioEnabled = audioToggle.value === 'true';

            if (!text && !file) return;

            addMessage(text || "Analyzing...", 'user', file ? URL.createObjectURL(file) : null);
            textInput.value = ''; imageInput.value = ''; preview.style.display = 'none';

            const formData = new FormData();
            formData.append('prompt', text);
            formData.append('lang', lang);
            if (file) formData.append('image', file);

            const loadingId = addLoadingBubble();

            try {
                const response = await fetch('/api/chat', { method: 'POST', body: formData });
                const data = await response.json();
                removeMessage(loadingId);
                
                if (response.ok) {
                    addMessage(data.text, 'ai');
                    if (audioEnabled && data.audio_url) { new Audio(data.audio_url).play(); }
                } else {
                    addMessage("Error: " + (data.text || "Unknown Server Error"), 'ai');
                }
            } catch (error) {
                removeMessage(loadingId);
                addMessage("Connection failed.", 'ai');
            }
        }

        function addMessage(text, sender, imageUrl = null) {
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            let content = sender === 'ai' ? marked.parse(text) : text;
            if (imageUrl) content = `<img src="${imageUrl}" style="max-width: 150px; border-radius: 10px; margin-bottom: 5px; display:block;">` + content;
            div.innerHTML = `<div class="avatar ${sender}"><i class="fa-solid fa-${sender === 'ai' ? 'robot' : 'user'}"></i></div><div class="bubble">${content}</div>`;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function addLoadingBubble() {
            const id = 'loading-' + Date.now();
            const div = document.createElement('div'); div.id = id; div.className = 'message ai';
            div.innerHTML = `<div class="avatar ai"><i class="fa-solid fa-robot"></i></div><div class="bubble"><i class="fa-solid fa-spinner fa-spin"></i> Analyzing...</div>`;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return id;
        }
        function removeMessage(id) { const el = document.getElementById(id); if (el) el.remove(); }

        sendBtn.addEventListener('click', sendMessage);
        textInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    </script>
    {% endif %}
</body>
</html>
"""

# --- BACKEND ROUTES ---

@app.route('/')
def home():
    # Pass user session to template to trigger conditional rendering
    user = session.get('user')
    return render_template_string(HTML_TEMPLATE, user=user)

@app.route('/auth/register', methods=['POST'])
def register():
    # FIXED: Check explicitly against None
    if users_collection is None:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400

    # Check if user exists
    if users_collection.find_one({"username": username}):
        return jsonify({"success": False, "message": "Username already taken"}), 400

    # Hash and Store
    hashed_pw = generate_password_hash(password)
    users_collection.insert_one({
        "username": username,
        "password": hashed_pw,
        "created_at": datetime.datetime.utcnow()
    })
    
    # Auto-login
    session['user'] = username
    return jsonify({"success": True, "message": "Account created!"})

@app.route('/auth/login', methods=['POST'])
def login():
    # FIXED: Check explicitly against None
    if users_collection is None:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    username = request.form.get('username')
    password = request.form.get('password')

    user = users_collection.find_one({"username": username})
    
    if user and check_password_hash(user['password'], password):
        session['user'] = username
        return jsonify({"success": True})
    
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/chat', methods=['POST'])
def chat():
    # Protect Route
    if 'user' not in session:
        return jsonify({"text": "⚠️ Session expired. Please login again."}), 401

    try:
        if not api_key:
             return jsonify({"text": "⚠️ Server Error: GOOGLE_API_KEY not found."}), 500
        
        prompt = request.form.get('prompt', '')
        lang = request.form.get('lang', 'en')
        image_file = request.files.get('image')

        full_prompt = f"{prompt}. Reply in {lang} language. If this is about agriculture, act as an expert agronomist."
        if not prompt and image_file:
            full_prompt = f"Analyze this image. Identify crop, disease, and provide solution. Reply in {lang} language."

        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        
        if image_file:
            img_bytes = image_file.read()
            image = Image.open(io.BytesIO(img_bytes))
            response = model.generate_content([full_prompt, image])
        else:
            response = model.generate_content(full_prompt)
            
        ai_text = response.text

        try:
            tts = gTTS(text=ai_text, lang=lang, slow=False)
            filename = f"speech_{uuid.uuid4()}.mp3"
            filepath = os.path.join(TEMP_DIR, filename)
            tts.save(filepath)
            audio_url = f"/audio/{filename}"
        except Exception:
            audio_url = None

        return jsonify({"text": ai_text, "audio_url": audio_url})

    except Exception as e:
        return jsonify({"text": f"Error: {str(e)}"}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    if 'user' not in session: return "Unauthorized", 401
    return send_file(os.path.join(TEMP_DIR, filename))

if __name__ == '__main__':
    print("🌿 CropSense AI is running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)