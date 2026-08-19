import os
import pickle
import logging
from typing import List, Tuple
import numpy as np
import faiss
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIndexBuilder:
    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        # Using IndexFlatIP for Cosine Similarity (requires L2-normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]):
        """Add pre-computed embeddings with metadata."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata items.")
        
        # Normalize embeddings for cosine similarity
        embeddings_copy = embeddings.copy()
        faiss.normalize_L2(embeddings_copy)
        
        self.index.add(embeddings_copy)
        self.metadata.extend(metadata)
        
    def build_from_dataset(self, dataset_gallery: List[Tuple], extractor, batch_size: int = 64):
        """Build index from MEVIDDataset gallery.
        dataset_gallery: list of (img_path, pid, camid, outfit_id)
        """
        logger.info(f"Building FAISS index from dataset with {len(dataset_gallery)} images...")
        
        from PIL import Image
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        device = next(extractor.parameters()).device
        extractor.eval()
        
        for i in range(0, len(dataset_gallery), batch_size):
            batch_items = dataset_gallery[i:i + batch_size]
            batch_tensors = []
            batch_meta = []
            
            for item in batch_items:
                img_path, pid, camid, outfit_id = item
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img)
                    batch_tensors.append(img_t)
                    batch_meta.append({
                        'image_path': img_path,
                        'person_id': pid,
                        'camera_id': camid,
                        'outfit_id': outfit_id
                    })
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    
            if not batch_tensors:
                continue
                
            batch_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = extractor(batch_tensor).cpu().numpy()
            
            self.add_embeddings(embeddings, batch_meta)
            
        logger.info(f"Finished building index. Total items: {self.index.ntotal}")

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'FAISSIndexBuilder':
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")
            
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        builder = cls(embedding_dim=index.d)
        builder.index = index
        builder.metadata = metadata
        logger.info(f"Loaded index from {index_path} with {index.ntotal} items")
        return builder
