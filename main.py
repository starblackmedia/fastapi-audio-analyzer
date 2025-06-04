
print(firebase_admin.__version__)

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import tempfile
import requests
from typing import Optional, Dict
import firebase_admin
from firebase_admin import auth, credentials

# ============ Firebase Init ============
cred = credentials.Certificate("firebase/service_account_key.json")
firebase_admin.initialize_app(cred)

# ============ FastAPI App ============
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Firebase Token Verification ============
async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    id_token = auth_header.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============ Protected Test Route ============
@app.get("/protected-data")
async def protected_data(user=Depends(verify_token)):
    return {"message": "Access granted", "user": user}

# ============ Audio Analysis Logic ============

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class AudioAnalyzeRequest(BaseModel):
    songId: str
    audioUrl: str
    analyze_mfcc: Optional[bool] = True
    analyze_structure: Optional[bool] = False
    predict_genre: Optional[bool] = True

def extract_african_features(y, sr) -> Dict[str, float]:
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "tempo": tempo,
        "key": key_idx,
        "duration": librosa.get_duration(y=y, sr=sr),
        "loudness_db": 10 * np.log10(np.mean(librosa.feature.rms(y=y)) + 1e-10),
        "percussive_strength": np.mean(y_percussive**2),
        "polyrhythm_score": np.std(pulse),
        "offbeat_emphasis": np.mean(pulse[1::2]) / (np.mean(pulse) + 1e-10),
        "harmonic_ratio": np.mean(y_harmonic / (y + 1e-10)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        **{f"mfcc_{i}": np.mean(coeff) for i, coeff in enumerate(mfccs[:5])}
    }

def predict_genre_from_features(features: Dict[str, float]) -> Dict[str, object]:
    tempo = features["tempo"]
    mfcc_0 = features.get("mfcc_0", 0)
    mfcc_1 = features.get("mfcc_1", 0)
    zcr = features.get("zero_crossing_rate", 0)
    centroid = features.get("spectral_centroid", 0)
    loudness = features.get("loudness_db", 0)

    rules = {
        "Afrobeats": [
            tempo >= 95 and tempo <= 125,
            mfcc_1 > 120,
            zcr >= 0.025 and zcr <= 0.05,
            centroid >= 1800 and centroid <= 3000,
        ],
        "R&B": [
            tempo < 90,
            mfcc_0 < -200,
            centroid < 2000,
            loudness < -5,
        ],
        "Hip-Hop": [
            tempo >= 80 and tempo <= 110,
            mfcc_0 > -190 and mfcc_0 < -130,
            zcr >= 0.015 and zcr <= 0.035,
        ],
        "Electronic/Pop": [
            tempo > 120,
            centroid > 3500,
            zcr > 0.05,
        ],
        "Dancehall/Pop": [
            tempo >= 100 and tempo <= 130,
            loudness > -6,
            centroid > 2500,
        ]
    }

    scores = {genre: (sum(conds), len(conds)) for genre, conds in rules.items()}
    best_genre, (passed, total) = max(scores.items(), key=lambda x: x[1][0]/x[1][1])
    confidence = passed / total

    if confidence < 0.5:
        return {"genre": "Unknown", "confidence": 0.0}
    
    return {"genre": best_genre, "confidence": round(confidence, 2)}

@app.post("/analyze-audio")
async def analyze_audio(request: AudioAnalyzeRequest):
    try:
        response = requests.get(request.audioUrl)
        response.raise_for_status()
        audio_data = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_file:
        tmp_file.write(audio_data)
        tmp_file.flush()

        try:
            y, sr = librosa.load(tmp_file.name, sr=None, res_type='kaiser_fast')
            features = extract_african_features(y, sr)

            result = {
                "songId": request.songId,
                "tempo": round(float(features["tempo"]), 0),
                "duration": round(float(features["duration"]), 2),
                "key": f"{KEYS[features['key']]}",
                "loudness_db": round(float(features["loudness_db"]), 2),
                "spectral_centroid": round(float(features["spectral_centroid"]), 2),
                "zero_crossing_rate": round(float(features["zero_crossing_rate"]), 4),
            }

            if request.analyze_mfcc:
                result["mfccs"] = {
                    k: round(float(v), 4)
                    for k, v in features.items()
                    if k.startswith("mfcc_")
                }

            if request.predict_genre:
                prediction = predict_genre_from_features(features)
                result["predicted_genre"] = prediction["genre"]
                result["confidence"] = prediction["confidence"]

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
