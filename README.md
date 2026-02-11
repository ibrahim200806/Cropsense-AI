🌿 CropSense AI

Link (Temp) : https://christoper-uneleemosynary-liana.ngrok-free.dev/

CropSense AI is an advanced agricultural intelligence platform designed to assist farmers and agronomists. It leverages Google's Gemini Generative AI, Computer Vision, and RAG (Retrieval-Augmented Generation) technology to provide real-time crop analysis, disease identification, and farming advice.

🚀 Features

AI-Powered Chatbot: Interactive chat interface powered by Google Gemini models for expert agricultural advice.

Image Analysis: Upload photos of crops to identify diseases, pests, or nutrient deficiencies instantly.

Voice Interaction: Supports Speech-to-Text for queries and Text-to-Speech (TTS) for reading out responses, making it accessible for users in the field.

Multilingual Support: Communicate in multiple languages (English, Hindi, Spanish, French, Tamil, Telugu).

RAG Knowledge Base: Ingests agricultural manuals (PDFs) and data (CSVs) to provide grounded, factual answers using LangChain and ChromaDB.

User Authentication: Secure Login and Registration system backed by MongoDB.

Modern UI: Responsive, glassmorphism-styled interface built with Flask and Jinja2.

🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript (Embedded in Flask), FontAwesome.

Backend: Python, Flask.

Database: MongoDB (User Data), ChromaDB (Vector Embeddings).

AI & LLM: Google Gemini (gemini-2.5-flash, gemini-3-flash-preview), LangChain.

Speech & Audio: gTTS (Google Text-to-Speech), SpeechRecognition, PyAudio.

Data Processing: Pandas, PyPDF, Pytesseract (OCR).

📋 Prerequisites

Before running the application, ensure you have the following installed:

Python 3.9+

MongoDB: Installed locally or a cloud connection string (Atlas).

Tesseract OCR: Required for image-based text ingestion.

Windows: Download Installer

Linux: sudo apt-get install tesseract-ocr

Mac: brew install tesseract

System Audio Tools:

Linux: sudo apt-get install python3-pyaudio portaudio19-dev

Mac: brew install portaudio

Windows: You may need Microsoft C++ Build Tools if PyAudio installation fails.

📦 Installation

Clone the Repository

git clone [https://github.com/yourusername/cropsense-ai.git](https://github.com/yourusername/cropsense-ai.git)
cd cropsense-ai


Create a Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt


Set Up Environment Variables
Create a .env file in the root directory and add your credentials:

# Google AI Studio Key
GOOGLE_API_KEY=your_google_api_key_here

# MongoDB Connection (Default local shown)
MONGO_URI=mongodb://localhost:27017/

# Flask Session Secret
SECRET_KEY=your_random_secret_string


🏃‍♂️ Usage

1. Ingest Knowledge Base (Optional)

If you have agricultural PDFs or CSVs, place them in data/raw_pdfs and run the ingestion script to populate the vector database.

python ingest.py


2. Run the Application

Start the Flask server:

python app.py


The app will run at http://127.0.0.1:5000.

Register a new account or log in to start using the AI.

📂 Project Structure

cropsense-ai/
├── app/
│   └── app.py            # Main Flask Application & Routes
├── data/
│   ├── raw_pdfs/         # Place input documents here
│   └── chroma_db/        # Generated Vector Database
├── ingest.py             # Script to process docs for RAG
├── retriever.py          # RAG Retrieval Logic
├── voice.py              # Audio processing utilities
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation


🧩 Key Libraries

flask: Web framework.

pymongo: Database connector.

google-generativeai: Access to Gemini API.

langchain-*: Framework for RAG pipeline.

chromadb: Vector store for document embeddings.

gTTS: Text-to-Speech conversion.

SpeechRecognition: Voice input processing.

🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

📄 License

This project is licensed under the MIT License.
