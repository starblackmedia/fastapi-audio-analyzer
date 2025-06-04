import os
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import pandas as pd
from dotenv import load_dotenv
import tempfile
import requests

# Load .env variables
load_dotenv()

# Initialize Spotify client
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
    ))
except Exception as e:
    print("❌ Error initializing Spotify client:", e)
    exit()

# Use known working public playlists for testing
AFRICAN_PLAYLISTS = {
    "afrobeats": "2DfNaw9Z1nuddjI6NczTXS",  # African Heat
    "amapiano": "4Ymf8eaPQGT7HMTymoX82f",   # AmaPiano Grooves
}

def extract_features(audio_url):
    """Extract features from preview MP3 URL."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            audio_data = requests.get(audio_url).content
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            y, sr = librosa.load(tmp_path, sr=None, duration=30)
            features = {
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                "key": int(np.argmax(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1))),
                "mfcc_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr)),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=y))
            }
            os.unlink(tmp_path)
            return features
        except Exception as e:
            os.unlink(tmp_path)
            print(f"❌ Error analyzing audio: {e}")
            return None
    except Exception as e:
        print(f"❌ Error downloading preview URL: {e}")
        return None

def build_dataset():
    dataset = []

    for genre, playlist_id in AFRICAN_PLAYLISTS.items():
        print(f"\n🎵 Processing genre: {genre} — Playlist ID: {playlist_id}")
        
        try:
            results = sp.playlist_tracks(playlist_id)
            tracks = results['items']

            # Pagination (get all tracks)
            while results['next']:
                results = sp.next(results)
                tracks.extend(results['items'])
        except Exception as e:
            print(f"❌ Failed to load playlist '{playlist_id}': {e}")
            continue
        
        print(f"✅ Found {len(tracks)} tracks in playlist.")
        
        for item in tracks:
            if item.get('track') is None:
                continue
            
            track = item['track']
            name = track.get('name')
            artist = track['artists'][0]['name']
            preview_url = track.get('preview_url')

            if not preview_url:
                print(f"  ⚠️ Skipping '{name}' by {artist} (no preview)")
                continue

            print(f"  🎧 Analyzing '{name}' by {artist}...")
            features = extract_features(preview_url)

            if features:
                features.update({
                    "genre": genre,
                    "track_name": name,
                    "artist": artist
                })
                dataset.append(features)

    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv("african_music_dataset.csv", index=False)
        print(f"\n✅ Dataset saved successfully: {len(df)} tracks in 'african_music_dataset.csv'")
    else:
        print("\n❌ No data collected. Check playlist access, preview availability, and network.")

if __name__ == "__main__":
    build_dataset()
