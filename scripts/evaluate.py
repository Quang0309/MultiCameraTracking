"""
Evaluation script for ReID models on MEVID dataset.

Evaluates under 3 protocols: general, same_outfit, cross_outfit.
Uses standalone MEVIDDataset and MEVIDEvaluator (no Fast-ReID dependency).

Usage:
    python scripts/evaluate.py --model resnet50 --weights weights/resnet50_mevid.pth --data-dir data/mevid
"""

import argparse
import logging
import sys
from pathlib import Path
from tabulate import tabulate

import numpy as np
from scipy.spatial.distance import cdist

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mevid_dataset import MEVIDDataset
from src.data.mevid_evaluator import MEVIDEvaluator
from src.models.feature_extractor import ReIDFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate(args):
    """Run evaluation on MEVID dataset under all 3 protocols."""
    # Load dataset
    logger.info(f"Loading MEVID dataset from {args.data_dir}...")
    dataset = MEVIDDataset(args.data_dir)
    logger.info(
        f"Loaded: {len(dataset.train)} train, "
        f"{len(dataset.query)} query, {len(dataset.gallery)} gallery images"
    )

    # Initialize feature extractor
    logger.info(f"Loading {args.model} model from {args.weights}...")
    extractor = ReIDFeatureExtractor(
        model_name=args.model,
        weights_path=args.weights,
        device=args.device
    )

    # Extract query features
    logger.info("Extracting query features...")
    query_paths = [sample[0] for sample in dataset.query]
    query_features = extractor.extract_from_paths(query_paths, batch_size=args.batch_size)
    logger.info(f"Query features shape: {query_features.shape}")

    # Extract gallery features
    logger.info("Extracting gallery features...")
    gallery_paths = [sample[0] for sample in dataset.gallery]
    gallery_features = extractor.extract_from_paths(gallery_paths, batch_size=args.batch_size)
    logger.info(f"Gallery features shape: {gallery_features.shape}")

    # Compute distance matrix (cosine distance)
    logger.info("Computing distance matrix...")
    distmat = cdist(query_features, gallery_features, metric='cosine')
    logger.info(f"Distance matrix shape: {distmat.shape}")

    # Prepare metadata for evaluator
    query_metas = [(pid, camid, outfit_id) for _, pid, camid, outfit_id in dataset.query]
    gallery_metas = [(pid, camid, outfit_id) for _, pid, camid, outfit_id in dataset.gallery]

    evaluator = MEVIDEvaluator(query_metas, gallery_metas)

    # Evaluate under all 3 protocols
    protocols = ['general', 'same_outfit', 'cross_outfit']
    all_results = {}

    for protocol in protocols:
        logger.info(f"Evaluating protocol: {protocol}...")
        results = evaluator.evaluate(distmat, protocol=protocol)
        all_results[protocol] = results

    # Print formatted results
    print("\n" + "=" * 70)
    print(f"  MEVID Evaluation Results - {args.model.upper()}")
    print("=" * 70)

    table_data = []
    for protocol in protocols:
        r = all_results[protocol]
        table_data.append([
            protocol.replace('_', ' ').title(),
            f"{r['mAP'] * 100:.1f}",
            f"{r['cmc']['rank-1'] * 100:.1f}",
            f"{r['cmc']['rank-5'] * 100:.1f}",
            f"{r['cmc']['rank-10'] * 100:.1f}",
        ])

    headers = ["Protocol", "mAP (%)", "Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Save results to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Weights: {args.weights}\n")
            f.write(f"Dataset: {args.data_dir}\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n")
        logger.info(f"Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ReID model on MEVID under 3 protocols"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=['resnet50', 'osnet', 'transreid'],
        help="Model architecture"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to MEVID dataset root (e.g., data/mevid)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help="Device for inference"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results text file"
    )
    args = parser.parse_args()

    evaluate(args)
