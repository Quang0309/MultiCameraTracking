import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.models.feature_extractor import ReIDFeatureExtractor
from src.indexing.index_builder import FAISSIndexBuilder
from src.indexing.searcher import FAISSSearcher
from src.pipeline.query import QueryProcessor

@st.cache_resource
def load_models_and_index(model_display_name, index_path, metadata_path):
    """Loads the model and FAISS index, cached by Streamlit."""
    model_map = {
        'ResNet-50': 'resnet50',
        'OSNet': 'osnet',
        'TransReID': 'transreid'
    }
    model_name = model_map.get(model_display_name, 'resnet50')
    weights_path = f"weights/{model_name}_mevid.pth"
    
    extractor = ReIDFeatureExtractor(model_name=model_name, weights_path=weights_path, device='auto')
    
    builder = FAISSIndexBuilder()
    builder.load(index_path, metadata_path)
    searcher = FAISSSearcher(index_builder=builder)
    
    return extractor, searcher

def main():
    st.title("Gallery Search")
    
    model_display_name = st.selectbox("Select Model", ['ResNet-50', 'OSNet', 'TransReID'])
    index_path = st.text_input("Index Path", "index.faiss")
    metadata_path = st.text_input("Metadata Path", "metadata.json")
    
    try:
        extractor, searcher = load_models_and_index(model_display_name, index_path, metadata_path)
    except Exception as e:
        st.error(f"Error loading model or index: {e}")
        return

    query_img_file = st.file_uploader("Upload Query Image", type=['jpg', 'jpeg', 'png'])
    top_k = st.slider("Top K results", min_value=1, max_value=50, value=10)
    
    if query_img_file is not None:
        file_bytes = np.asarray(bytearray(query_img_file.read()), dtype=np.uint8)
        query_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB), caption="Query Image", width=300)
        
        with st.spinner("Extracting features and searching..."):
            embedding = extractor.extract(query_bgr)
            results = searcher.search(embedding, top_k=top_k)
            grouped_results = searcher.group_by_camera(results)
            
        st.subheader("Search Results")
        
        for cam_id, cam_results in grouped_results.items():
            st.markdown(f"### Camera: {cam_id}")
            cols = st.columns(min(len(cam_results), 5))
            for idx, res in enumerate(cam_results):
                col = cols[idx % 5]
                meta = res.get('metadata', {})
                img_path = meta.get('img_path', '')
                score = res.get('score', 0.0)
                pid = meta.get('pid', 'Unknown')
                
                with col:
                    try:
                        res_img = Image.open(img_path)
                        st.image(res_img, caption=f"PID: {pid}\nScore: {score:.2f}")
                    except Exception as e:
                        st.error(f"Image not found: {img_path}")

if __name__ == "__main__":
    main()
