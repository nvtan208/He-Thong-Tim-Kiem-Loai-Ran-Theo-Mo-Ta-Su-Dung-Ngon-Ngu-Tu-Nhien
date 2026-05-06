import os
import pickle
import numpy as np
import faiss
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from werkzeug.utils import secure_filename

# Try to import translator
try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='vi', target='en')
    print("[OK] Translator initialized successfully")
except Exception as e:
    print(f"[WARN] Translator initialization failed ({e}), using fallback (no translation)")
    translator = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS index and metadata
# ⚠️ IMPORTANT: FAISS index contains ONLY image embeddings (normalized L2)
print("Loading FAISS index and metadata...")
index = faiss.read_index('embeddings/faiss_index.bin')

with open('embeddings/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Loaded {len(metadata)} metadata entries")

def translate_to_english(text):
    """Translate Vietnamese text to English before encoding"""
    if translator is None:
        print(f"[TRANSLATE] Translator not available, using original text: '{text}'")
        return text
    
    try:
        # Check if text contains Vietnamese characters
        vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        has_vietnamese_chars = any(c in text.lower() for c in vietnamese_chars)
        
        if has_vietnamese_chars:
            # Translate using deep_translator
            translated_text = translator.translate(text)
            print(f"[TRANSLATE] Vietnamese detected: '{text}' -> '{translated_text}'")
            return translated_text
        else:
            print(f"[TRANSLATE] Not Vietnamese, using original text: '{text}'")
            return text
    except Exception as e:
        print(f"[TRANSLATE ERROR] {str(e)}, using original text: '{text}'")
        return text

def get_text_embedding(text):
    """Get CLIP embedding for text (normalized)"""
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        # Extract tensor from output
        text_features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        # Normalize the features
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    
    return text_features.cpu().numpy()

def get_image_embedding(image_pil):
    """Get CLIP embedding for image (normalized)"""
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        # Extract tensor from output
        image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        # Normalize the features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
    
    return image_features.cpu().numpy()

def search_by_text(query, top_k=5):
    """Search using text description only"""
    # Translate Vietnamese to English before encoding
    translated_query = translate_to_english(query)
    query_embedding = get_text_embedding(translated_query)
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            result = {
                'rank': i + 1,
                'score': float(distances[0][i]),
                'binomial': metadata[idx]['binomial'],
                'class_id': metadata[idx]['class_id'],
                'image_path': metadata[idx]['image_path'],
                'description': metadata[idx]['description'],
                'poisonous': bool(metadata[idx]['poisonous']),
                'country': metadata[idx]['country'],
                'continent': metadata[idx]['continent']
            }
            results.append(result)
    
    return results

def search_by_image(image_pil, top_k=5):
    """Search using image only"""
    query_embedding = get_image_embedding(image_pil)
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            result = {
                'rank': i + 1,
                'score': float(distances[0][i]),
                'binomial': metadata[idx]['binomial'],
                'class_id': metadata[idx]['class_id'],
                'image_path': metadata[idx]['image_path'],
                'description': metadata[idx]['description'],
                'poisonous': bool(metadata[idx]['poisonous']),
                'country': metadata[idx]['country'],
                'continent': metadata[idx]['continent']
            }
            results.append(result)
    
    return results

def search_fusion(text_query, image_pil, top_k=5, text_weight=0.3, image_weight=0.7):
    """Fusion search: combine text and image (default: 70% image + 30% text)"""
    # Translate Vietnamese to English before encoding
    translated_query = translate_to_english(text_query)
    text_embedding = get_text_embedding(translated_query)
    image_embedding = get_image_embedding(image_pil)
    
    # Weighted fusion
    fusion_embedding = (image_weight * image_embedding + text_weight * text_embedding).astype('float32')
    fusion_embedding = fusion_embedding / np.linalg.norm(fusion_embedding)
    
    # Normalize for FAISS
    faiss.normalize_L2(fusion_embedding)
    
    distances, indices = index.search(fusion_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            result = {
                'rank': i + 1,
                'score': float(distances[0][i]),
                'binomial': metadata[idx]['binomial'],
                'class_id': metadata[idx]['class_id'],
                'image_path': metadata[idx]['image_path'],
                'description': metadata[idx]['description'],
                'poisonous': bool(metadata[idx]['poisonous']),
                'country': metadata[idx]['country'],
                'continent': metadata[idx]['continent']
            }
            results.append(result)
    
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search/text', methods=['POST'])
def search_text():
    """Search by text description"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 5))
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if top_k < 1 or top_k > 10:
            top_k = 5
        
        # Translate query for display
        translated_query = translate_to_english(query)
        
        results = search_by_text(query, top_k)
        
        return jsonify({
            'search_type': 'text',
            'original_query': query,
            'translated_query': translated_query,
            'top_k': top_k,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        print(f"Error in text search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/image', methods=['POST'])
def search_image():
    """Search by image upload"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        top_k = int(request.form.get('top_k', 5))
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if top_k < 1 or top_k > 10:
            top_k = 5
        
        # Load image
        image = Image.open(file.stream).convert('RGB')
        
        results = search_by_image(image, top_k)
        
        return jsonify({
            'search_type': 'image',
            'top_k': top_k,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        print(f"Error in image search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/fusion', methods=['POST'])
def search_fusion_api():
    """Fusion search: text + image (70% image + 30% text)"""
    try:
        # Get text query
        query = request.form.get('query', '').strip()
        top_k = int(request.form.get('top_k', 5))
        image_weight = float(request.form.get('image_weight', 0.7))
        text_weight = 1.0 - image_weight
        
        if 'image' not in request.files:
            return jsonify({'error': 'Image required for fusion search'}), 400
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if top_k < 1 or top_k > 10:
            top_k = 5
        
        # Load image
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Translate query for display
        translated_query = translate_to_english(query)
        
        results = search_fusion(query, image, top_k, text_weight, image_weight)
        
        return jsonify({
            'search_type': 'fusion',
            'original_query': query,
            'translated_query': translated_query,
            'top_k': top_k,
            'image_weight': image_weight,
            'text_weight': text_weight,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        print(f"Error in fusion search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """Serve image"""
    try:
        full_path = os.path.join('..', image_path)
        if os.path.exists(full_path):
            return jsonify({'url': f'/serve_image/{image_path}'})
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/serve_image/<path:image_path>')
def serve_image(image_path):
    """Serve image file from archive folders"""
    try:
        from flask import send_file
        
        print(f"\n[DEBUG] Requesting image: {image_path}")
        
        # Try direct path first (from app directory)
        full_path = os.path.join(os.path.dirname(__file__), image_path)
        print(f"[DEBUG] Trying path 1: {full_path}")
        
        if not os.path.exists(full_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.path.dirname(__file__), 'archive', 'train', os.path.basename(image_path)),
                os.path.join(os.path.dirname(__file__), 'archive', 'test', os.path.basename(image_path)),
            ]
            for i, alt_path in enumerate(alt_paths):
                print(f"[DEBUG] Trying path {i+2}: {alt_path}")
                if os.path.exists(alt_path):
                    full_path = alt_path
                    print(f"[DEBUG] ✓ Found at: {full_path}")
                    break
        
        if os.path.exists(full_path):
            print(f"[DEBUG] ✓ Serving image: {full_path}")
            return send_file(full_path, mimetype='image/jpeg')
        else:
            print(f"[DEBUG] ✗ Image not found: {full_path}")
            print(f"[DEBUG] Original path: {image_path}")
            return "Image not found", 404
    except Exception as e:
        print(f"[ERROR] Exception serving image: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)