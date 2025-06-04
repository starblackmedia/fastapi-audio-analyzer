import requests
import librosa
import numpy as np
import pandas as pd
import os
import tempfile

# Search Deezer for songs in a genre or query
def search_deezer(query, limit=10):
    url = f"https://api.deezer.com/search?q={query}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error searching Deezer: {response.status_code}")
        return []
    return response.json().get("data", [])

# Download and extract audio features from a preview URL
def extract_features(audio_url):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            audio_data = requests.get(audio_url).content
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        y, sr = librosa.load(tmp_path, sr=None, duration=30)
        os.unlink(tmp_path)

        return {
            "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
            "mfcc_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=y))
        }
    except Exception as e:
        print(f"❌ Error analyzing audio: {e}")
        return None

# Build dataset for a genre
def build_deezer_dataset(query, limit=10):
    results = search_deezer(query, limit)
    dataset = []

    for track in results:
        title = track['title']
        artist = track['artist']['name']
        preview_url = track.get('preview')

        if not preview_url:
            print(f"⚠️ Skipping '{title}' by {artist} (no preview)")
            continue

        print(f"🎧 Analyzing '{title}' by {artist}...")
        features = extract_features(preview_url)

        if features:
            features.update({
                "track_name": title,
                "artist": artist,
                "genre": query
            })
            dataset.append(features)

    if dataset:
        df = pd.DataFrame(dataset)
        filename = f"deezer_{query}_dataset.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Saved {len(df)} tracks to {filename}")
    else:
        print("\n❌ No data collected. Try another query or check network.")

# Run it!
if __name__ == "__main__":
    build_deezer_dataset("afrobeats", limit=20)  # change query as needed
