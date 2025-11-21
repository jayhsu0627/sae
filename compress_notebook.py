import nbformat
import base64
import io
from PIL import Image

# SETTINGS
INPUT_FILENAME = 'toy_batch.ipynb'
OUTPUT_FILENAME = 'toy_batch_compressed.ipynb'
QUALITY = 90  # 0-100 (70 is usually a great balance)
MAX_WIDTH = 2000 # Resize images wider than this (set None to disable)

def compress_image(b64_string, mime_type):
    # Decode base64 string to image
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data))
    
    # Convert RGBA to RGB (JPEG doesn't support transparency)
    if img.mode in ('RGBA', 'LA'):
        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        background.paste(img, img.split()[-1])
        img = background.convert('RGB')
    else:
        img = img.convert('RGB')

    # Resize if too wide
    if MAX_WIDTH and img.width > MAX_WIDTH:
        new_height = int(img.height * (MAX_WIDTH / img.width))
        img = img.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)

    # Compress to JPEG
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Load Notebook
print(f"Reading {INPUT_FILENAME}...")
with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

count = 0
total_saved = 0

# Iterate through cells
for cell in nb.cells:
    if 'outputs' in cell:
        for output in cell['outputs']:
            # Look for image data
            if 'data' in output:
                # We prioritize PNG/JPEG keys
                for mime in ['image/png', 'image/jpeg']:
                    if mime in output['data']:
                        original_b64 = output['data'][mime]
                        
                        try:
                            # Compress
                            new_b64 = compress_image(original_b64, mime)
                            
                            # Replace data with new JPEG key
                            # (Remove old PNG key to save space)
                            output['data'].pop('image/png', None)
                            output['data']['image/jpeg'] = new_b64
                            
                            count += 1
                        except Exception as e:
                            print(f"Skipped an image due to error: {e}")

# Save New Notebook
print(f"Compressed {count} images.")
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Saved to {OUTPUT_FILENAME}. You can now convert this file to HTML.")