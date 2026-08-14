import os
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
# --- 1. CONFIGURATION AND INITIALIZATION ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRETKEY")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration (MUST MATCH TRAINING)
MODEL_PATH = 'CNN & ANN/best_model_ann.h5'
'''SAMPLE_RATE = 44100
N_MFCC = 40
TARGET_FRAME_COUNT = 193'''
 # exact frames used in notebook

# 5 classes only
CLASS_LABELS = ['COPD', 'Healthy','URTI', 'Pneumonia','Bronchial_Disease']

# Load model
try:
    LUNG_MODEL = load_model(MODEL_PATH)
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"ERROR: Could not load model. Ensure '{MODEL_PATH}' exists. Details: {e}")
    LUNG_MODEL = None

# --- 2. PREPROCESSING ---
# app.py

def allowed_file(filename):
    """Checks file extension case-insensitively."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path, sample_rate=44100):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sample_rate)

        # MFCC (40)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

        # Chroma STFT (12)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        # Mel Spectrogram (128)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        # Spectral Contrast (7)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        # Tonnetz (6)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sr).T, axis=0)

        # Concatenate all features
        feature_vector = np.concatenate([
            mfcc,
            chroma,
            mel,
            contrast,
            tonnetz
        ])

        return feature_vector

    except Exception as e:
        print("Feature Extraction Error:", e)
        return None
    
    
# --- 3. FLASK ROUTES ---
@app.before_request
def require_login():
    allowed_routes = ['login', 'static']
    if request.endpoint not in allowed_routes and 'logged_in' not in session:
        return redirect(url_for('login'))

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('upload'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'user' and password == 'lungapp':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if LUNG_MODEL is None:
        return render_template('upload.html', error="Model failed to load on startup.")

    if request.method == 'POST':
        if 'audiofile' not in request.files:
            return render_template('upload.html', error='No file part in request.')
        file = request.files['audiofile']
        if file.filename == '':
            return render_template('upload.html', error='No selected file.')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # CORRECTED CALL: Pass the required sample rate and MFCC count
            features = extract_features(filepath)
           
            if features is not None:
                input_data = np.expand_dims(features, axis=0)
                try:
                    predictions_prob = LUNG_MODEL.predict(input_data)
                    predicted_index = np.argmax(predictions_prob)
                    predicted_label = CLASS_LABELS[predicted_index]
                    confidence = predictions_prob[0][predicted_index] * 100
                except Exception as e:
                    return render_template('upload.html', error=f"Prediction error: {e}")

                os.remove(filepath)
                return redirect(url_for('results', disease=predicted_label,
                                        confidence=f"{confidence:.2f}", filename=filename))

        return render_template('upload.html', error='File type not allowed or upload failed.')

    return render_template('upload.html')


@app.route('/results', methods=['GET'])
def results():
    disease = request.args.get('disease')
    confidence = request.args.get('confidence')
    filename = request.args.get('filename')
    if disease and confidence:
        return render_template('results.html', disease=disease, confidence=confidence, filename=filename)
    return redirect(url_for('upload'))

if __name__ == '__main__':
    if LUNG_MODEL is not None:
        app.run(debug=True)
    else:
        print("Application startup failed due to model loading error.")
