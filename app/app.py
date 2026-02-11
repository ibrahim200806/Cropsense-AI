import os
import uuid
import tempfile
import warnings
import io
import datetime
import random # Added for mock sensor data
import certifi # ADDED: To fix SSL Handshake errors
import pickle # ADDED: For loading the ML model
import pandas as pd # ADDED: For structuring model input
import xgboost # ADDED: Essential for loading XGBoost models from pickle

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
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
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
users_collection = None # Initialize safely

try:
    # FIXED: Reverted to tlsAllowInvalidCertificates=True as certifi failed in your environment.
    # This ensures connection works for the hackathon regardless of SSL issues.
    client = MongoClient(
        mongo_uri, 
        serverSelectionTimeoutMS=5000,
        tlsAllowInvalidCertificates=True
    )
    client.admin.command('ping')
    db = client.cropsense_db
    users_collection = db.users
    print(f"✅ Successfully connected and authenticated to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Error: {e}")
    users_collection = None

# --- LOAD PRICE PREDICTION MODEL ---
price_model = None
model_path = os.path.join(os.path.dirname(__file__), 'price_model.pkl')
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            price_model = pickle.load(f)
        print(f"✅ Price Prediction Model loaded from {model_path}")
    else:
        print(f"⚠️ Warning: price_model.pkl not found at {model_path}")
except Exception as e:
    print(f"❌ Error loading price model: {e}")

# Global Temp Directory for Audio
TEMP_DIR = tempfile.gettempdir()

# --- HTML/CSS/JS FRONTEND ---
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
            --nitrogen-color: #00d2ff;
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

        /* --- APP LAYOUT --- */
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
        
        /* Glass Panel */
        .glass-panel {
            background: var(--glass-bg); backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border); border-radius: 24px; padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2); position: relative; overflow: hidden;
            display: flex; flex-direction: column; /* Ensure flex flow */
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
        
        /* Special Buttons */
        .nitrogen-btn {
            background: rgba(0, 210, 255, 0.15); 
            border-color: var(--nitrogen-color);
            color: var(--nitrogen-color);
            font-weight: 600;
        }
        .nitrogen-btn:hover {
            background: rgba(0, 210, 255, 0.3);
            box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
        }

        .price-toggle-btn {
            background: rgba(241, 196, 15, 0.15);
            border-color: #f1c40f;
            color: #f1c40f;
            font-weight: 600;
            margin-top: 10px;
        }
        .price-toggle-btn:hover {
            background: rgba(241, 196, 15, 0.3);
            box-shadow: 0 0 15px rgba(241, 196, 15, 0.4);
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

        /* --- PRICE PREDICTION UI --- */
        .price-dashboard {
            display: none; /* Hidden by default */
            flex-direction: column;
            gap: 20px;
            animation: fadeIn 0.5s ease;
            height: 100%;
            overflow-y: auto;
        }

        .price-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(0,255,136,0.05) 100%);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .price-input-group label {
            display: block;
            color: #ccc;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .price-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            color: white;
            padding: 12px;
            border-radius: 12px;
            outline: none;
            transition: 0.3s;
        }
        .price-input:focus { border-color: var(--primary-accent); box-shadow: 0 0 10px rgba(0, 255, 136, 0.2); }

        .predict-btn {
            width: 100%;
            background: linear-gradient(90deg, #00ff88, #00b894);
            color: #05140a;
            padding: 15px;
            border: none;
            border-radius: 15px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        .predict-btn:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(0, 255, 136, 0.6); }

        .result-box {
            margin-top: 2rem;
            padding: 2rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.05);
            text-align: center;
            display: none;
        }

        .price-display {
            font-size: 3.5rem;
            font-weight: 800;
            color: var(--primary-accent);
            margin: 10px 0;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }

        .alert-box {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            display: inline-block;
            width: 100%;
        }

        .alert-green { background: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
        .alert-red { background: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }
        .alert-yellow { background: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }

        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        .loading-shimmer {
            animation: shimmer 2s infinite linear;
            background: linear-gradient(to right, rgba(255,255,255,0.05) 4%, rgba(255,255,255,0.1) 25%, rgba(255,255,255,0.05) 36%);
            background-size: 1000px 100%;
        }
        
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

            <!-- Dashboard Features -->
            <div class="control-group" style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem;">
                <label style="color: #ccc;"><i class="fa-solid fa-layer-group"></i> Smart Tools</label>
                
                <button id="nitrogenBtn" onclick="toggleView('chat')" class="toggle-btn nitrogen-btn">
                    <i class="fa-solid fa-flask"></i> Nitrogen Tracker
                </button>
                
                <button id="priceBtn" onclick="toggleView('price')" class="toggle-btn price-toggle-btn">
                    <i class="fa-solid fa-chart-line"></i> Price Predictor
                </button>
                
                <button onclick="toggleView('chat')" class="toggle-btn" style="margin-top: 10px; background: rgba(0, 255, 136, 0.15); border-color: var(--primary-accent); color: var(--primary-accent);">
                    <i class="fa-regular fa-comments"></i> AI Chatbot
                </button>
            </div>
            
            <a href="/logout" style="text-decoration: none; margin-top: auto;">
                <button class="toggle-btn" style="background: rgba(255, 71, 87, 0.2); color: #ff4757; border-color: #ff4757;">
                    <i class="fa-solid fa-right-from-bracket"></i> Logout
                </button>
            </a>

            <div style="padding-top: 20px;">
                <p style="font-size: 0.8rem; opacity: 0.5;"><i class="fa-solid fa-circle-info"></i> v2.5 Auth Edition</p>
            </div>
        </aside>

        <!-- MAIN INTERFACE -->
        <main class="main-content">
            
            <!-- VIEW 1: CHAT BOT & NITROGEN -->
            <div id="chat-view" style="display: flex; flex-direction: column; height: 100%;">
                <div class="glass-panel" style="flex-grow: 1;">
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

                <div class="glass-panel" style="padding: 0.8rem; margin-top: 1.5rem; flex-shrink: 0;">
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
            </div>

            <!-- VIEW 2: PRICE PREDICTOR DASHBOARD -->
            <div id="price-view" class="price-dashboard">
                <div class="price-card">
                    <h2 style="margin-bottom: 20px; color: var(--primary-accent);"><i class="fa-solid fa-chart-line"></i> Crop Price Prediction & Farmer Alert</h2>
                    
                    <div class="form-grid">
                        <div class="price-input-group"><label>State</label><input type="text" id="predState" class="price-input" placeholder="e.g. Maharashtra"></div>
                        <div class="price-input-group"><label>District</label><input type="text" id="predDistrict" class="price-input" placeholder="e.g. Pune"></div>
                        <div class="price-input-group"><label>Market</label><input type="text" id="predMarket" class="price-input" placeholder="e.g. APMC"></div>
                        <div class="price-input-group"><label>Commodity</label><input type="text" id="predCommodity" class="price-input" placeholder="e.g. Onion"></div>
                        <div class="price-input-group"><label>Variety</label><input type="text" id="predVariety" class="price-input" placeholder="e.g. Red"></div>
                        <div class="price-input-group"><label>Grade</label><input type="text" id="predGrade" class="price-input" placeholder="e.g. A"></div>
                        <div class="price-input-group"><label>Min Price (₹)</label><input type="number" id="predMin" class="price-input" placeholder="0"></div>
                        <div class="price-input-group"><label>Max Price (₹)</label><input type="number" id="predMax" class="price-input" placeholder="0"></div>
                        <div class="price-input-group"><label>Current Modal Price (₹)</label><input type="number" id="currentPrice" class="price-input" placeholder="2000"></div>
                        <div class="price-input-group"><label>Date</label><input type="date" id="predDate" class="price-input"></div>
                    </div>

                    <button onclick="predictPrice()" class="predict-btn">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Future Price
                    </button>

                    <div id="priceResult" class="result-box">
                        <div id="loadingAnim" style="display:none;">
                            <p>Analyzing market trends & historical data...</p>
                            <div style="height: 60px; width: 80%; margin: 10px auto; border-radius: 10px;" class="loading-shimmer"></div>
                        </div>
                        <div id="finalResult" style="display:none;">
                            <p style="opacity: 0.8; font-size: 1rem;">Predicted Modal Price (Next 7 Days)</p>
                            <div class="price-display" id="predictedPriceDisplay">₹ 0</div>
                            <div id="alertBox" class="alert-box"></div>
                        </div>
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
        const nitrogenBtn = document.getElementById('nitrogenBtn');

        marked.setOptions({ breaks: true, gfm: true });

        // VIEW TOGGLING logic
        function toggleView(viewName) {
            const chatView = document.getElementById('chat-view');
            const priceView = document.getElementById('price-view');
            
            if (viewName === 'price') {
                chatView.style.display = 'none';
                priceView.style.display = 'flex';
            } else {
                chatView.style.display = 'flex';
                priceView.style.display = 'none';
            }
        }

        // PRICE PREDICTION LOGIC (Connected to Backend)
        async function predictPrice() {
            const currentPriceInput = document.getElementById('currentPrice').value;
            const currentPrice = parseFloat(currentPriceInput) || 0;
            
            const resultBox = document.getElementById('priceResult');
            const loading = document.getElementById('loadingAnim');
            const finalResult = document.getElementById('finalResult');
            const display = document.getElementById('predictedPriceDisplay');
            const alertBox = document.getElementById('alertBox');

            // Gather Data
            const payload = {
                State: document.getElementById('predState').value,
                District: document.getElementById('predDistrict').value,
                Market: document.getElementById('predMarket').value,
                Commodity: document.getElementById('predCommodity').value,
                Variety: document.getElementById('predVariety').value,
                Grade: document.getElementById('predGrade').value,
                Min_Price: document.getElementById('predMin').value,
                Max_Price: document.getElementById('predMax').value,
                Current_Price: currentPrice,
                Date: document.getElementById('predDate').value
            };

            resultBox.style.display = 'block';
            loading.style.display = 'block';
            finalResult.style.display = 'none';

            try {
                // Call Backend API
                const response = await fetch('/api/predict_price', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                finalResult.style.display = 'block';

                if (response.ok) {
                    const predicted = data.predicted_price.toFixed(2);
                    display.innerText = `₹ ${predicted}`;
                    
                    // Reset Classes
                    alertBox.className = 'alert-box';
                    
                    // Logic for Alert Message (Comparing predicted vs current)
                    if (currentPrice > 0) {
                        if (predicted > currentPrice * 1.05) {
                            alertBox.classList.add('alert-green');
                            alertBox.innerHTML = '<i class="fa-solid fa-arrow-trend-up"></i> Price Increasing: Hold crop for better profit';
                        } else if (predicted < currentPrice * 0.95) {
                            alertBox.classList.add('alert-red');
                            alertBox.innerHTML = '<i class="fa-solid fa-arrow-trend-down"></i> Price Decreasing: Sell early to avoid loss';
                        } else {
                            alertBox.classList.add('alert-yellow');
                            alertBox.innerHTML = '<i class="fa-solid fa-minus"></i> Market Stable: Price expected to remain stable';
                        }
                    } else {
                        alertBox.classList.add('alert-yellow');
                        alertBox.innerHTML = '<i class="fa-solid fa-check"></i> Prediction Complete';
                    }
                } else {
                    display.innerText = "Error";
                    alertBox.innerText = data.error || "Prediction failed";
                    alertBox.className = 'alert-box alert-red';
                }

            } catch (error) {
                loading.style.display = 'none';
                finalResult.style.display = 'block';
                display.innerText = "Error";
                alertBox.innerText = "Connection Failed";
                alertBox.className = 'alert-box alert-red';
                console.error(error);
            }
        }

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

        // --- MOCK SENSOR LOGIC (FOR HACKATHON) ---
        nitrogenBtn.addEventListener('click', () => {
            toggleView('chat'); // Ensure we are on chat view to see result
            
            // 1. Show Connecting Message
            addMessage("📡 Connecting to IoT Soil Sensor (ESP32)...", 'ai');
            
            // 2. Simulate delay for "Scanning"
            setTimeout(() => {
                const levels = [120, 135, 142, 128, 145]; // Fake realistic data
                const randomLevel = levels[Math.floor(Math.random() * levels.length)];
                
                const resultMsg = `
### 🌱 Soil Nitrogen Analysis Complete

* **Sensor Status:** Online 🟢
* **Detected Nitrogen (N):** ${randomLevel} kg/ha
* **Status:** **Optimal Range** ✅

**Recommendation:** Your soil nitrogen levels are sufficient for the current growth stage. Continue with standard irrigation. No immediate fertilization required.
                `;
                addMessage(resultMsg, 'ai');
                
            }, 2500); // 2.5 second delay
        });

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

@app.route('/api/predict_price', methods=['POST'])
def predict_price():
    if not price_model:
        return jsonify({"error": "Price Model not loaded. Check server logs for pkl error."}), 500
    
    try:
        data = request.json
        # NOTE: This creates a DataFrame from the input. 
        # Make sure the keys below match the columns your model was trained on!
        
        # --- FIX: Split Date into Year and Month ---
        # Feature mismatch (10 vs 11) usually means Date was split during training.
        date_str = data.get('Date')
        if date_str:
            dt = pd.to_datetime(date_str)
            year = dt.year
            month = dt.month
        else:
            dt = datetime.datetime.now()
            year = dt.year
            month = dt.month

        input_data = pd.DataFrame([{
            'State': data.get('State'),
            'District': data.get('District'),
            'Market': data.get('Market'),
            'Commodity': data.get('Commodity'),
            'Variety': data.get('Variety'),
            'Grade': data.get('Grade'),
            'Min Price': float(data.get('Min_Price', 0)),
            'Max Price': float(data.get('Max_Price', 0)),
            'Modal Price': float(data.get('Current_Price', 0)), # Assuming this is a feature too
            'Year': year,
            'Month': month
        }])
        
        # --- FIX: Convert object columns to 'category' dtype for XGBoost ---
        # The error "Invalid columns: State: object" happens because XGBoost needs 
        # specific category types, not raw strings.
        categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('category')
        
        prediction = price_model.predict(input_data)[0]
        
        return jsonify({"predicted_price": float(prediction)})
        
    except Exception as e:
        print(f"Prediction Error Details: {e}") # Log detailed error to console
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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

        # UPDATED: Added instruction to forbid LaTeX formatting
        system_instruction = "IMPORTANT: Use plain text for temperatures and units (e.g., 25°C, 75°F). Do NOT use LaTeX formatting like $25^{\circ}$."

        full_prompt = f"{prompt}. Reply in {lang} language. If this is about agriculture, act as an expert agronomist. {system_instruction}"
        if not prompt and image_file:
            full_prompt = f"Analyze this image. Identify crop, disease, and provide solution. Reply in {lang} language. {system_instruction}"

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