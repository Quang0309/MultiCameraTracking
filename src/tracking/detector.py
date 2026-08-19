import numpy as np
from typing import Tuple, List
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.4, device: str = 'auto'):
        """Initialize YOLOv8 person detector."""
        self.conf_threshold = conf_threshold
        logger.info(f"Loading YOLO model {model_name}...")
        
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = YOLO(model_name)
        self.device = device
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect persons in a frame.
        Returns:
            bboxes: np.ndarray of shape (N, 4) in xyxy format
            confidences: np.ndarray of shape (N,)
        """
        # class 0 is person in COCO dataset
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False, device=self.device)[0]
        
        bboxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        
        return bboxes, confidences

    def detect_and_crop(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Detect persons and return crops.
        Returns bboxes, confidences, and list of cropped images.
        """
        bboxes, confidences = self.detect(frame)
        
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
            
        return bboxes, confidences, crops
