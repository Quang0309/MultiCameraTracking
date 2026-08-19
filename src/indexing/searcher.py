import numpy as np
import faiss
from typing import List, Dict
from .index_builder import FAISSIndexBuilder

class FAISSSearcher:
    def __init__(self, index_builder: FAISSIndexBuilder):
        self.index = index_builder.index
        self.metadata = index_builder.metadata
        
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[dict]:
        """Search for top_k most similar embeddings.
        Returns list of dicts with keys: similarity, image_path, camera_id, person_id, etc.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        query = query_embedding.copy()
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta['similarity'] = float(dist)
            results.append(meta)
            
        return results

    def search_batch(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[List[dict]]:
        """Batch search for multiple queries."""
        query_embeddings = query_embeddings.copy()
        faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, top_k)
        
        batch_results = []
        for i in range(len(query_embeddings)):
            results = []
            for dist, idx in zip(distances[i], indices[i]):
                if idx == -1:
                    continue
                meta = self.metadata[idx].copy()
                meta['similarity'] = float(dist)
                results.append(meta)
            batch_results.append(results)
            
        return batch_results

    def group_by_camera(self, results: List[dict]) -> Dict[int, List[dict]]:
        """Group search results by camera ID for multi-camera display."""
        grouped = {}
        for res in results:
            camid = res.get('camera_id')
            if camid not in grouped:
                grouped[camid] = []
            grouped[camid].append(res)
        return grouped

    def filter_by_threshold(self, results: List[dict], threshold: float = 0.5) -> List[dict]:
        """Filter results by minimum similarity threshold."""
        return [res for res in results if res['similarity'] >= threshold]
