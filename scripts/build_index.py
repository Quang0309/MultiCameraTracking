import argparse
import os

from src.data.mevid_dataset import MEVIDDataset
from src.models.feature_extractor import ReIDFeatureExtractor
from src.indexing.index_builder import FAISSIndexBuilder

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from MEVID dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MEVID dataset root")
    parser.add_argument("--model", type=str, default="resnet50", help="ReID model name")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--index_path", type=str, default="index.faiss", help="Output index path")
    parser.add_argument("--metadata_path", type=str, default="metadata.json", help="Output metadata path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    args = parser.parse_args()

    print(f"Loading MEVIDDataset from {args.data_dir}...")
    dataset = MEVIDDataset(root_dir=args.data_dir)
    gallery_data = dataset.gallery

    print(f"Loaded {len(gallery_data)} items from gallery.")
    
    paths = [item[0] for item in gallery_data]
    metadata = [
        {
            'img_path': item[0],
            'pid': item[1],
            'camid': item[2],
            'outfit_id': item[3]
        } for item in gallery_data
    ]

    print(f"Initializing ReIDFeatureExtractor ({args.model})...")
    extractor = ReIDFeatureExtractor(model_name=args.model, weights_path=args.weights_path, device='auto')

    print("Extracting features from gallery paths...")
    embeddings = extractor.extract_from_paths(paths, batch_size=args.batch_size)

    print("Building FAISS index...")
    builder = FAISSIndexBuilder()
    builder.add_embeddings(embeddings, metadata)
    
    print(f"Saving index to {args.index_path} and metadata to {args.metadata_path}...")
    builder.save(args.index_path, args.metadata_path)
    print("Index built successfully!")

if __name__ == "__main__":
    main()
