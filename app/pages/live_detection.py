import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from src.tracking.detector import PersonDetector
from src.models.feature_extractor import ReIDFeatureExtractor
from src.tracking.tracker import ReIDTracker

def main():
    st.title("Live Detection & Tracking")

    # Upload inputs
    query_img_file = st.file_uploader("Upload Query Image", type=['jpg', 'jpeg', 'png'])
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

    # Config options
    threshold = st.slider("Target Match Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if query_img_file and video_file:
        st.write("Initializing models...")
        # Initialize correct API signatures
        extractor = ReIDFeatureExtractor(model_name='resnet50', weights_path='weights/resnet50_mevid.pth', device='auto')
        detector = PersonDetector(model_name='yolov8n.pt', conf_threshold=0.4, device='auto')
        tracker = ReIDTracker(reid_extractor=extractor, max_age=30, n_init=3, max_cosine_distance=0.2)

        # Process query image
        query_bytes = np.asarray(bytearray(query_img_file.read()), dtype=np.uint8)
        query_bgr = cv2.imdecode(query_bytes, cv2.IMREAD_COLOR)
        query_embedding = extractor.extract(query_bgr)
        
        st.image(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB), caption="Query Image Target", width=250)

        # Process video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        frame_idx = 0
        
        st.write("Processing video frame by frame...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detection
            bboxes, confidences = detector.detect(frame)
            
            # Tracking
            tracks = tracker.update(frame, bboxes, confidences)
            
            # ReID Target Finding
            target_track = tracker.find_target(tracks, query_embedding, threshold=threshold)
            target_id = target_track.get('track_id') if target_track else None

            # Annotation
            for track in tracks:
                track_id = track.get('track_id')
                bbox = track.get('bbox_ltrb')
                x1, y1, x2, y2 = map(int, bbox)

                if track_id == target_id:
                    color = (0, 0, 255) # RED for target
                    label = f"TARGET ID: {track_id}"
                else:
                    color = (0, 255, 0) # GREEN for others
                    label = f"ID: {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        out.release()
        
        st.success("Processing complete!")
        st.video(out_path)

if __name__ == "__main__":
    main()
