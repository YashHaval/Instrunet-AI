from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import torch
import torch.nn as nn
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from torchvision import models

# ================= APP =================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_resnet18_irmas.pth")

print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 11)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ ResNet18 model loaded successfully")

# ================= LABELS (ORDER MUST MATCH TRAINING) =================
INSTRUMENTS = [
    "Cello",
    "Clarinet",
    "Flute",
    "Acoustic Guitar",
    "Electric Guitar",
    "Organ",
    "Piano",
    "Saxophone",
    "Trumpet",
    "Violin",
    "Voice"
]

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["audio"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return jsonify({"filename": filename})

@app.route("/uploads/<filename>")
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= ANALYZE =================
@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.json["filename"]
    path = os.path.join(UPLOAD_FOLDER, filename)

    # Load audio
    y, sr = librosa.load(path, sr=22050, mono=True)

    # 🔑 REAL TIMELINE SETTINGS
    window_sec = 3.0
    hop_sec = 0.5

    seg_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))

    timeline = []

    with torch.no_grad():
        for start in range(0, len(y) - seg_len + 1, hop_len):
            seg = y[start:start + seg_len]

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=seg,
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                fmax=8000
            )
            mel = librosa.power_to_db(mel, ref=np.max)

            # Fix time dimension
            if mel.shape[1] < 128:
                mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
            else:
                mel = mel[:, :128]

            # Match training input
            x = torch.tensor(mel).float()
            x = x.unsqueeze(0)          # (1, H, W)
            x = x.repeat(3, 1, 1)       # (3, H, W)
            x = (x - x.mean()) / (x.std() + 1e-6)
            x = x.unsqueeze(0).to(device)

            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

            # 🔑 EMA SMOOTHING (VISUAL ONLY)
            if timeline:
                alpha = 0.6
                probs = alpha * probs + (1 - alpha) * np.array(timeline[-1]["probs"])

            timeline.append({
                "time": round(start / sr, 2),
                "probs": probs.tolist()
            })

    # Global average (UNCHANGED)
    avg = np.mean([t["probs"] for t in timeline], axis=0)

    return jsonify({
        "average": {
            INSTRUMENTS[i]: float(avg[i])
            for i in range(len(INSTRUMENTS))
        },
        "timeline": timeline
    })

# ================= PDF EXPORT =================
@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    data = request.json

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "InstruNet AI – Instrument Recognition Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"File: {data.get('filename','Unknown')}")
    y -= 25

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Detected Instruments")
    y -= 20

    c.setFont("Helvetica", 12)
    for k, v in data["average"].items():
        c.drawString(70, y, f"{k}: {v*100:.2f}%")
        y -= 16
        if y < 40:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="instrument_report.pdf"
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0")
