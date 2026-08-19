import cv2
import numpy as np
from typing import List
from tqdm import tqdm
import logging
import os
from .detector import PersonDetector
from .tracker import ReIDTracker

logger = logging.getLogger(__name__)

class VideoAnnotator:
    """Annotates video frames with tracking results."""
    
    def __init__(self, target_color=(0, 255, 0), other_color=(128, 128, 128), thickness=2):
        self.target_color = target_color
        self.other_color = other_color
        self.thickness = thickness
    
    def annotate_frame(self, frame: np.ndarray, tracks: List[dict], target_track_id: int = None) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame.
        Highlights the target track in target_color.
        """
        annotated = frame.copy()
        
        for track in tracks:
            track_id = int(track['track_id'])
            x1, y1, x2, y2 = map(int, track['bbox_ltrb'])
            
            is_target = (track_id == target_track_id)
            color = self.target_color if is_target else self.other_color
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.thickness)
            
            label = f"ID: {track_id}"
            if is_target:
                label += " (TARGET)"
                
            cv2.putText(annotated, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, self.thickness)
                        
        return annotated

    def process_video(self, input_path: str, output_path: str, 
                      detector: PersonDetector, tracker: ReIDTracker,
                      query_embedding: np.ndarray,
                      progress_callback=None) -> str:
        """Process entire video: detect -> track -> annotate -> save.
        Returns path to output video.
        """
        logger.info(f"Processing video: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracker.reset()
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            bboxes, confidences = detector.detect(frame)
            
            tracks = tracker.update(frame, bboxes, confidences)
            
            target_track = tracker.find_target(tracks, query_embedding, threshold=0.45)
            target_track_id = int(target_track['track_id']) if target_track else None
            
            annotated_frame = self.annotate_frame(frame, tracks, target_track_id)
            
            out.write(annotated_frame)
            
            pbar.update(1)
            if progress_callback:
                progress_callback(frame_idx, total_frames)
                
            frame_idx += 1
            
        pbar.close()
        cap.release()
        out.release()
        
        logger.info(f"Video saved to {output_path}")
        return output_path
