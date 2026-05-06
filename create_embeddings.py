import os
import pandas as pd
import numpy as np
import faiss
import pickle
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load data
print("Loading data...")
df = pd.read_csv('train_with_descriptions.csv')

# Create output directory
os.makedirs('embeddings', exist_ok=True)

# Store metadata
metadata = []
embeddings_list = []

print(f"Processing {len(df)} images...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # Get image path
        image_path = row['image_path']
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get description
        description = row['description']
        
        # Create CLIP input (image only for FAISS index)
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get image embeddings only
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize L2
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings_list.append(image_features.cpu().numpy())
        
        # Store metadata
        metadata.append({
            'binomial': row['binomial'],
            'class_id': row['class_id'],
            'uuid': row['UUID'],
            'image_path': image_path,
            'description': description,
            'poisonous': row['poisonous'],
            'country': row['country'],
            'continent': row['continent']
        })
        
    except Exception as e:
        print(f"Error processing {row['UUID']}: {str(e)}")
        continue

# Convert to numpy array
embeddings = np.vstack(embeddings_list).astype('float32')

print(f"\nCreated {len(embeddings)} image embeddings with shape {embeddings.shape}")

# Normalize L2 before adding to FAISS
faiss.normalize_L2(embeddings)

# Create FAISS index with Inner Product (cosine similarity on normalized vectors)
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and metadata
print("Saving index and metadata...")
faiss.write_index(index, 'embeddings/faiss_index.bin')

with open('embeddings/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

with open('embeddings/embeddings.npy', 'wb') as f:
    np.save(f, embeddings)

print("✓ FAISS index created successfully!")
print(f"✓ Saved {len(metadata)} metadata entries")
print(f"✓ Embedding dimension: {dimension}")