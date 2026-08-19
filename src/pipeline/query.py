import numpy as np
from typing import List, Dict, Any

from src.models.feature_extractor import ReIDFeatureExtractor
from src.indexing.searcher import FAISSSearcher

class QueryProcessor:
    """Processes search queries for the ReID system."""
    
    def __init__(self, extractor: ReIDFeatureExtractor, searcher: FAISSSearcher):
        """
        Initialize the QueryProcessor.
        
        Args:
            extractor: ReID feature extractor instance.
            searcher: FAISS searcher instance connected to the index.
        """
        self.extractor = extractor
        self.searcher = searcher

    def process_query(self, image: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Process a query image and return top matches.
        
        Args:
            image (np.ndarray): Query image in BGR format.
            top_k (int): Number of top search results to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of dictionary results from the searcher.
        """
        embedding = self.extractor.extract(image)
        results = self.searcher.search(embedding, top_k=top_k)
        return results
