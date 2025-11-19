#!/usr/bin/env python3
"""
Simple web viewer for CC3M images and captions.
Serves images and captions from the organized directory structure.
"""

import json
import random
from pathlib import Path
from flask import Flask, send_file, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
DATA_DIR = Path("/mnt/drive_a/Projects/sae/data/cc3m")
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_JSON = DATA_DIR / "captions.json"
PATHS_JSON = DATA_DIR / "image_paths.json"

# Cache for caption lookups (load on-demand)
_captions_cache = None
_paths_cache = None
_total_images = None

def load_captions():
    """Load captions JSON file (lazy loading)"""
    global _captions_cache
    if _captions_cache is None:
        print(f"Loading captions from {CAPTIONS_JSON}...")
        with open(CAPTIONS_JSON, 'r', encoding='utf-8') as f:
            _captions_cache = json.load(f)
        print(f"Loaded {len(_captions_cache)} captions")
    return _captions_cache

def load_paths():
    """Load image paths JSON file (lazy loading)"""
    global _paths_cache
    if _paths_cache is None:
        if PATHS_JSON.exists():
            print(f"Loading paths from {PATHS_JSON}...")
            with open(PATHS_JSON, 'r', encoding='utf-8') as f:
                _paths_cache = json.load(f)
            print(f"Loaded {len(_paths_cache)} paths")
        else:
            # Fallback: generate paths by mapping global index to local paths
            # This requires scanning directories - might be slow for large datasets
            print("Paths JSON not found, generating from directory structure...")
            print("This might take a while. Consider running the write command to generate image_paths.json")
            _paths_cache = {}
            # Build mapping by processing parquet files in order
            parquet_files = sorted(DATA_DIR.glob("*.parquet"))
            global_idx = 0
            for parquet_file in parquet_files:
                folder_name = parquet_file.stem
                folder_path = IMAGES_DIR / folder_name
                if folder_path.exists():
                    img_files = sorted(folder_path.glob("*.jpg"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
                    for img_file in img_files:
                        rel_path = f"images/{folder_name}/{img_file.name}"
                        _paths_cache[str(global_idx)] = rel_path
                        global_idx += 1
            print(f"Generated {len(_paths_cache)} paths")
    return _paths_cache

def get_total_images():
    """Get total number of images"""
    global _total_images
    if _total_images is None:
        captions = load_captions()
        _total_images = len(captions)
    return _total_images

@app.route('/')
def index():
    """Serve the HTML viewer"""
    html_file = Path(__file__).parent / "viewer.html"
    return send_file(html_file)

@app.route('/api/total')
def get_total():
    """Get total number of images"""
    return jsonify({"total": get_total_images()})

@app.route('/api/caption/<int:idx>')
def get_caption(idx):
    """Get caption for a specific image index"""
    captions = load_captions()
    caption = captions.get(str(idx), "No caption available")
    return jsonify({"index": idx, "caption": caption})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    return send_from_directory(str(DATA_DIR), f'images/{filename}')

@app.route('/api/image-path/<int:idx>')
def get_image_path(idx):
    """Get relative image path for a specific index"""
    paths = load_paths()
    if str(idx) in paths:
        return jsonify({"index": idx, "path": paths[str(idx)]})
    else:
        return jsonify({"error": "Path not found"}), 404

@app.route('/api/random')
def get_random():
    """Get a random image index"""
    total = get_total_images()
    if total == 0:
        return jsonify({"error": "No images available"}), 404
    random_idx = random.randint(0, total - 1)
    return jsonify({"index": random_idx})

if __name__ == '__main__':
    print(f"Starting CC3M Viewer...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Captions file: {CAPTIONS_JSON}")
    print(f"\nOpen http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)

