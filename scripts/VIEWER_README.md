# CC3M Image Viewer

A simple web-based viewer for browsing CC3M dataset images and captions.

## Features

- 🖼️ Browse images organized by parquet file folders
- 📝 View captions for each image
- 🎲 Random image selection
- ⬅️➡️ Sequential navigation
- 🔍 Jump to specific image index
- ⌨️ Keyboard navigation (arrow keys, R for random)

## Requirements

```bash
pip install flask
```

## Usage

1. Make sure you have:
   - Run `python scripts/conversion.py write` to generate images and JSON files
   - Generated `captions.json` and `image_paths.json` in the data directory

2. Start the viewer server:
   ```bash
   python scripts/viewer.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## How It Works

### Smart JSON Loading

The viewer uses **lazy loading** to handle large JSON files (1-2GB):

1. **On-demand loading**: JSON files are only loaded when first accessed
2. **In-memory caching**: Once loaded, captions and paths are cached for fast lookups
3. **Single load**: The JSON is loaded once and reused for all requests

### Performance Considerations

- **Initial load**: Loading a 1-2GB JSON file may take 10-30 seconds on first access
- **Memory usage**: The JSON will be kept in memory (~1-2GB RAM)
- **Subsequent requests**: After initial load, caption lookups are instant
- **Image serving**: Images are served directly from disk (no caching needed)

### API Endpoints

- `GET /` - Serve the HTML viewer
- `GET /api/total` - Get total number of images
- `GET /api/caption/<idx>` - Get caption for image index
- `GET /api/image-path/<idx>` - Get image path for index
- `GET /api/random` - Get a random image index
- `GET /images/<path>` - Serve image files

## Configuration

Edit `viewer.py` to change:
- Data directory path
- Server port (default: 5000)
- Host address (default: 0.0.0.0)

## Keyboard Shortcuts

- `←` (Left Arrow) - Previous image
- `→` (Right Arrow) - Next image
- `R` - Random image

## Troubleshooting

### JSON file not found
Make sure you've run the `write` command to generate `captions.json` and `image_paths.json`.

### Images not loading
- Check that images are organized in folders by parquet file name
- Verify `image_paths.json` contains correct paths
- Check server logs for errors

### Slow performance
- First load of JSON will be slow (10-30 seconds for 1-2GB file)
- Subsequent requests should be fast
- Consider using SSD for faster disk access

### Memory issues
If you have limited RAM, consider:
- Using a smaller subset of the dataset
- Implementing streaming JSON parsing (requires code changes)
- Using a database instead of JSON files

