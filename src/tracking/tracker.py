import numpy as np
import cv2
import torch
from typing import List, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import torchvision.transforms as T

class ReIDTracker:
    """DeepSORT tracker using custom ReID embeddings.
    
    Uses deep-sort-realtime with pre-computed embeddings from our
    trained ReID model, instead of the built-in feature extractor.
    """
    def __init__(self, reid_extractor, 
                 max_age: int = 30, n_init: int = 3,
                 max_cosine_distance: float = 0.2):
        """Initialize tracker."""
        self.reid_extractor = reid_extractor
        
        # Disable deep-sort-realtime's built-in embedder
        self.tracker = DeepSort(max_age=max_age,
                                n_init=n_init,
                                max_cosine_distance=max_cosine_distance,
                                nn_budget=100,
                                embedder=None)
                                
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((0, 2048))
            
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
                
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop_rgb)
            crops.append(self.transform(img))
            
        batch = torch.stack(crops)
        device = next(self.reid_extractor.parameters()).device
        batch = batch.to(device)
        
        self.reid_extractor.eval()
        with torch.no_grad():
            embeddings = self.reid_extractor(batch).cpu().numpy()
            
        return embeddings

    def update(self, frame: np.ndarray, bboxes: np.ndarray, confidences: np.ndarray) -> List[dict]:
        """Update tracker with new detections.
        Returns list of confirmed tracks: {track_id, bbox_ltrb, embedding}
        """
        if len(bboxes) == 0:
            self.tracker.update_tracks([], frame=frame)
            return []
            
        embeddings = self._extract_embeddings(frame, bboxes)
        
        ltwh_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            ltwh_bboxes.append([x1, y1, x2 - x1, y2 - y1])
            
        detections = []
        for ltwh, conf, embed in zip(ltwh_bboxes, confidences, embeddings):
            detections.append((ltwh, conf, 'person', embed))
            
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # The deep_sort_realtime library stores features in track.features
            features = track.features
            track_embed = features[-1] if len(features) > 0 else np.zeros((2048,))
            
            results.append({
                'track_id': track_id,
                'bbox_ltrb': ltrb,
                'embedding': track_embed
            })
            
        return results

    def find_target(self, tracks: List[dict], query_embedding: np.ndarray, threshold: float = 0.5) -> Optional[dict]:
        """Find the target person among active tracks by comparing embeddings.
        Returns the matching track or None.
        """
        if not tracks:
            return None
            
        import faiss
        
        query = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query)
        
        track_embeddings = np.array([t['embedding'] for t in tracks])
        faiss.normalize_L2(track_embeddings)
        
        similarities = np.dot(query, track_embeddings.T)[0]
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= threshold:
            result = tracks[best_idx].copy()
            result['similarity'] = best_sim
            return result
            
        return None

    def reset(self):
        """Reset tracker state."""
        self.tracker.tracker.tracks = []
        self.tracker.tracker._next_id = 1
