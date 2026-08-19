import cv2
import numpy as np
from typing import Optional

from src.tracking.detector import PersonDetector
from src.models.feature_extractor import ReIDFeatureExtractor
from src.indexing.index_builder import FAISSIndexBuilder

class VideoIngestor:
    """Ingests videos to build a gallery index of person embeddings."""

    def __init__(self, detector: PersonDetector, extractor: ReIDFeatureExtractor):
        """
        Initialize the VideoIngestor.
        
        Args:
            detector: Model to detect persons in frames.
            extractor: Model to extract embeddings from cropped images.
        """
        self.detector = detector
        self.extractor = extractor

    def ingest(self, video_path: str, camera_id: str, index_builder: FAISSIndexBuilder, skip_frames: int = 10):
        """
        Processes a video and adds detected person embeddings to the index.

        Args:
            video_path (str): Path to the video file to ingest.
            camera_id (str): String identifier for the camera.
            index_builder (FAISSIndexBuilder): The index builder instance.
            skip_frames (int): Number of frames to skip between processing.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                bboxes, confidences = self.detector.detect(frame)
                
                crops = []
                valid_bboxes = []
                valid_confs = []
                
                for bbox, conf in zip(bboxes, confidences):
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Boundary checks
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crops.append(crop)
                        valid_bboxes.append(bbox)
                        valid_confs.append(conf)

                if crops:
                    embeddings = self.extractor.extract(crops)
                    
                    metadata = [
                        {
                            'camera_id': camera_id,
                            'frame_idx': frame_idx,
                            'bbox': b.tolist(),
                            'confidence': float(c)
                        }
                        for b, c in zip(valid_bboxes, valid_confs)
                    ]
                    
                    index_builder.add_embeddings(embeddings, metadata)

            frame_idx += 1
            
        cap.release()
