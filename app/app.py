import os
import uuid
import tempfile
import warnings
import io

# --- SUPPRESS WARNINGS ---
# Google's library emits a FutureWarning about updates. 
# We suppress it here to keep the console clean and professional.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template_string, request, jsonify, send_file
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv
from PIL import Image

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Fallback to empty string to prevent crash on start, 
    # but actual calls will fail if not set in environment or via the tool.
    api_key = "" 
    print("Warning: GOOGLE_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)

# Global Temp Directory for Audio
TEMP_DIR = tempfile.gettempdir()

# --- HTML/CSS/JS FRONTEND (Embedded for Single-File Portability) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CropSense AI | Ultimate Agri-Tech</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Marked.js for Markdown Parsing (Fixes the stars issue) -->
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
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background: url("https://images.unsplash.com/photo-1625246333195-5519a495d026?q=80&w=2574&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
            color: var(--text-color);
            overflow-x: hidden;
        }

        /* Dark Liquid Overlay */
        body::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: var(--dark-overlay);
            z-index: -1;
            backdrop-filter: blur(3px);
        }

        /* --- LAYOUT --- */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 2rem;
            height: 100vh;
        }

        /* --- SIDEBAR (Controls) --- */
        .sidebar {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .logo-area {
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .logo-icon {
            font-size: 3rem;
            color: var(--primary-accent);
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        }

        .control-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #ccc;
        }

        select, .toggle-btn {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            color: white;
            padding: 12px;
            border-radius: 12px;
            outline: none;
            cursor: pointer;
            transition: 0.3s;
        }

        select:hover {
            border-color: var(--primary-accent);
        }

        /* --- MAIN CONTENT AREA --- */
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: calc(100vh - 4rem);
        }

        /* --- GLASS PANELS --- */
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }

        /* Liquid Shine Effect */
        .glass-panel::after {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 50%; height: 100%;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.05), transparent);
            transform: skewX(-25deg);
            transition: 0.5s;
            pointer-events: none;
        }
        .glass-panel:hover::after {
            left: 100%;
            transition: 0.7s;
        }

        /* --- CHAT AREA --- */
        .chat-container {
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 10px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        /* Custom Scrollbar */
        .chat-container::-webkit-scrollbar { width: 6px; }
        .chat-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }

        .message {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            max-width: 80%;
            animation: fadeIn 0.3s ease;
        }

        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .avatar {
            width: 40px; height: 40px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        
        .avatar.ai { background: linear-gradient(135deg, #11998e, #38ef7d); box-shadow: 0 0 15px rgba(56, 239, 125, 0.4); }
        .avatar.user { background: rgba(255,255,255,0.2); }

        .bubble {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 18px;
            border-top-left-radius: 2px;
            line-height: 1.6;
            font-size: 0.95rem;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Markdown Styling inside Bubbles */
        .bubble strong { color: var(--primary-accent); font-weight: 700; }
        .bubble ul { margin-left: 20px; margin-top: 5px; margin-bottom: 5px; }
        .bubble li { margin-bottom: 5px; }
        .bubble h1, .bubble h2, .bubble h3 { color: var(--primary-accent); margin-top: 10px; margin-bottom: 5px; font-size: 1.1rem;}

        .message.user .bubble {
            background: rgba(0, 255, 136, 0.1);
            border-color: rgba(0, 255, 136, 0.2);
            border-radius: 18px;
            border-top-right-radius: 2px;
        }

        /* --- INPUT AREA --- */
        .input-area {
            display: flex;
            gap: 1rem;
            align-items: center;
            background: rgba(0,0,0,0.2);
            padding: 0.5rem;
            border-radius: 50px;
            border: 1px solid var(--glass-border);
        }

        .file-upload-label {
            cursor: pointer;
            padding: 10px;
            color: var(--primary-accent);
            transition: 0.3s;
        }
        .file-upload-label:hover { transform: scale(1.1); text-shadow: 0 0 10px var(--primary-accent); }

        #imageInput { display: none; }

        #textInput {
            flex-grow: 1;
            background: transparent;
            border: none;
            color: white;
            padding: 10px;
            font-size: 1rem;
            outline: none;
        }

        .action-btn {
            background: var(--primary-accent);
            color: #000;
            border: none;
            width: 45px; height: 45px;
            border-radius: 50%;
            cursor: pointer;
            transition: 0.3s;
            display: flex; align-items: center; justify-content: center;
        }
        .action-btn:hover { transform: scale(1.1); box-shadow: 0 0 15px var(--primary-accent); }
        
        /* Microphone Button Styling */
        #micBtn {
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        #micBtn.listening {
            background: var(--recording-color);
            color: white;
            animation: pulse 1.5s infinite;
        }

        /* --- PREVIEW IMAGE --- */
        .image-preview {
            max-width: 100px;
            max-height: 100px;
            border-radius: 10px;
            display: none;
            border: 2px solid var(--primary-accent);
            margin-right: 10px;
        }
        
        /* --- AUDIO PLAYER --- */
        .audio-player-container {
            margin-top: 10px;
            width: 100%;
        }
        audio {
            width: 100%;
            height: 30px;
            opacity: 0.7;
            border-radius: 15px;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 71, 87, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; height: auto; }
            .sidebar { padding: 1rem; }
            .main-content { height: 600px; }
        }
    </style>
</head>
<body>

    <div class="container">
        <!-- SIDEBAR -->
        <aside class="sidebar">
            <div class="logo-area">
                <div class="logo-icon"><i class="fa-solid fa-leaf"></i></div>
                <h2>CropSense AI</h2>
                <p style="opacity: 0.7; font-size: 0.8rem;">Advanced Agri-Intelligence</p>
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

            <div style="margin-top: auto; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="font-size: 0.8rem; opacity: 0.5;"><i class="fa-solid fa-circle-info"></i> v2.0 Ultimate Edition</p>
            </div>
        </aside>

        <!-- MAIN INTERFACE -->
        <main class="main-content">
            <div class="glass-panel" style="flex-grow: 1; display: flex; flex-direction: column;">
                <div class="chat-container" id="chatContainer">
                    <!-- Welcome Message -->
                    <div class="message ai">
                        <div class="avatar ai"><i class="fa-solid fa-robot"></i></div>
                        <div class="bubble">
                            Hello! I am CropSense AI. 🌱<br>
                            I can analyze crop diseases from images or answer your farming questions. How can I help today?
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
                        
                        <input type="text" id="textInput" placeholder="Ask about crop health, weather, or soil..." autocomplete="off">
                        
                        <!-- Voice Input Button -->
                        <button id="micBtn" class="action-btn" title="Speak"><i class="fa-solid fa-microphone"></i></button>
                        
                        <!-- Send Button -->
                        <button id="sendBtn" class="action-btn" style="margin-left: 5px;"><i class="fa-solid fa-paper-plane"></i></button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- JAVASCRIPT LOGIC -->
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const textInput = document.getElementById('textInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const langSelect = document.getElementById('languageSelect');
        const audioToggle = document.getElementById('audioToggle');

        // --- MARKDOWN CONFIGURATION ---
        // Configure marked to handle line breaks properly
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        // --- VOICE INPUT (SPEECH RECOGNITION) ---
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;
        let isListening = false;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            
            // Map our simple language codes to Speech Recognition codes
            const langMap = {
                'en': 'en-US',
                'hi': 'hi-IN',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'ta': 'ta-IN',
                'te': 'te-IN'
            };

            recognition.onstart = function() {
                isListening = true;
                micBtn.classList.add('listening');
                textInput.placeholder = "Listening...";
            };

            recognition.onend = function() {
                isListening = false;
                micBtn.classList.remove('listening');
                textInput.placeholder = "Ask about crop health, weather, or soil...";
            };

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                textInput.value = transcript;
                // Optional: Auto-send after speaking
                // sendMessage();
            };

            micBtn.addEventListener('click', function() {
                if (isListening) {
                    recognition.stop();
                } else {
                    // Update language based on dropdown
                    recognition.lang = langMap[langSelect.value] || 'en-US';
                    recognition.start();
                }
            });
        } else {
            micBtn.style.display = 'none'; // Hide if not supported
            console.log("Web Speech API not supported in this browser.");
        }


        // Handle Image Preview
        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        // Send Message Logic
        async function sendMessage() {
            const text = textInput.value;
            const file = imageInput.files[0];
            const lang = langSelect.value;
            const audioEnabled = audioToggle.value === 'true';

            if (!text && !file) return;

            // 1. Add User Message to Chat
            addMessage(text || "Analyzing uploaded image...", 'user', file ? URL.createObjectURL(file) : null);
            
            // Clear inputs
            textInput.value = '';
            imageInput.value = '';
            preview.style.display = 'none';

            // 2. Prepare Data
            const formData = new FormData();
            formData.append('prompt', text);
            formData.append('lang', lang);
            if (file) formData.append('image', file);

            // 3. Show Loading
            const loadingId = addLoadingBubble();

            try {
                // 4. Send to Backend
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // 5. Remove Loading & Add AI Response
                removeMessage(loadingId);
                addMessage(data.text, 'ai');

                // 6. Handle Audio
                if (audioEnabled && data.audio_url) {
                    const audio = new Audio(data.audio_url);
                    audio.play();
                }

            } catch (error) {
                removeMessage(loadingId);
                addMessage("Error connecting to server. Please check your API key.", 'ai');
                console.error(error);
            }
        }

        // UI Helpers
        function addMessage(text, sender, imageUrl = null) {
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            
            // Process text with markdown if it's from AI
            let content = text;
            if (sender === 'ai') {
                content = marked.parse(text);
            }

            if (imageUrl) {
                content = `<img src="${imageUrl}" style="max-width: 150px; border-radius: 10px; margin-bottom: 5px; display:block;">` + content;
            }

            const icon = sender === 'ai' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-solid fa-user"></i>';

            div.innerHTML = `
                <div class="avatar ${sender}">${icon}</div>
                <div class="bubble">${content}</div>
            `;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function addLoadingBubble() {
            const id = 'loading-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = 'message ai';
            div.innerHTML = `
                <div class="avatar ai"><i class="fa-solid fa-robot"></i></div>
                <div class="bubble"><i class="fa-solid fa-spinner fa-spin"></i> Analyzing...</div>
            `;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return id;
        }

        function removeMessage(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        sendBtn.addEventListener('click', sendMessage);
        textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

    </script>
</body>
</html>
"""

# --- BACKEND ROUTES ---

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Prevent 404 errors for favicon
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Check API key first
        if not api_key:
             return jsonify({"text": "⚠️ Server Error: GOOGLE_API_KEY not found in .env file."}), 500
        
        prompt = request.form.get('prompt', '')
        lang = request.form.get('lang', 'en')
        image_file = request.files.get('image')

        # Construct Prompt with Language instruction
        full_prompt = f"{prompt}. Reply in {lang} language. If this is about agriculture, act as an expert agronomist."
        
        if not prompt and image_file:
            full_prompt = f"Analyze this image. Identify crop, disease, and provide solution. Reply in {lang} language."

        # Corrected Model Name to the stable version
        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        
        # Call Gemini
        if image_file:
            img_bytes = image_file.read()
            image = Image.open(io.BytesIO(img_bytes))
            response = model.generate_content([full_prompt, image])
        else:
            response = model.generate_content(full_prompt)
            
        ai_text = response.text

        # Generate Audio
        try:
            tts = gTTS(text=ai_text, lang=lang, slow=False)
            filename = f"speech_{uuid.uuid4()}.mp3"
            filepath = os.path.join(TEMP_DIR, filename)
            tts.save(filepath)
            audio_url = f"/audio/{filename}"
        except Exception as e:
            print(f"TTS Error: {e}")
            audio_url = None

        return jsonify({
            "text": ai_text,
            "audio_url": audio_url
        })

    except Exception as e:
        return jsonify({"text": f"Error: {str(e)}"}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(os.path.join(TEMP_DIR, filename))

if __name__ == '__main__':
    print("🌿 CropSense AI is running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)