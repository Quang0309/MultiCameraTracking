"""
Multi-Protocol Evaluation Script for MEVID Dataset on Fast-ReID.
Evaluates:
  1. General Protocol (standard ReID)
  2. Same-Outfit Protocol
  3. Cross-Outfit Protocol (the core MEVID challenge)
"""

import os
import sys
import argparse
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable

import torch
import numpy as np
from scipy.spatial.distance import cdist
from tabulate import tabulate
from tqdm import tqdm

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_setup
from fastreid.utils.checkpoint import Checkpointer

# Register MEVID dataset
from src.data.mevid_dataset import MEVID, MEVID_Mini
from src.data.mevid_evaluator import MEVIDEvaluator


def parse_outfit_from_path(img_path):
    """Extract (pid, outfit_id, camid) directly from MEVID filename:
    e.g. 0201O003C330T004F00192.jpg -> pid=201, outfit=3, camid=330
    """
    fname = os.path.basename(img_path)
    try:
        pid = int(fname.split('O')[0])
        outfit_id = int(fname.split('O')[1].split('C')[0])
        camid = int(fname.split('C')[1].split('T')[0])
        return pid, outfit_id, camid
    except Exception:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3 MEVID protocols on Kaggle")
    parser.add_argument("--config", dest="config_file", default="configs/mevid_agw_r50_ibn.yml")
    parser.add_argument("--weights", dest="weights", required=True, help="Path to model_best.pth")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()

    print(f"\n[INFO] Loading model from checkpoint: {cfg.MODEL.WEIGHTS}")
    model = DefaultTrainer.build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    dataset_name = cfg.DATASETS.TESTS[0]
    print(f"[INFO] Building test dataloader for {dataset_name}...")
    test_loader, num_query = DefaultTrainer.build_test_loader(cfg, dataset_name=dataset_name)

    all_features = []
    all_pids = []
    all_camids = []
    all_outfits = []

    print(f"[INFO] Extracting features for {len(test_loader.dataset)} test images (Num Query: {num_query})...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            imgs = batch["images"].to(model.device)
            feats = model(imgs)
            # L2 Normalize
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_features.append(feats.cpu().numpy())

            for path in batch["img_paths"]:
                pid, oid, cid = parse_outfit_from_path(path)
                all_pids.append(pid)
                all_outfits.append(oid)
                all_camids.append(cid)

    all_features = np.concatenate(all_features, axis=0)
    print(f"[INFO] Feature extraction complete. Matrix shape: {all_features.shape}")

    # Split into query and gallery
    query_feats = all_features[:num_query]
    gallery_feats = all_features[num_query:]

    query_metas = [(p, c, o) for p, c, o in zip(all_pids[:num_query], all_camids[:num_query], all_outfits[:num_query])]
    gallery_metas = [(p, c, o) for p, c, o in zip(all_pids[num_query:], all_camids[num_query:], all_outfits[num_query:])]

    device = model.device
    del model
    del test_loader
    torch.cuda.empty_cache()
    print("[INFO] Model freed from GPU. Proceeding to memory-safe ranking...")

    print("[INFO] Computing Distance & Sorting Rankings in GPU chunks...")
    g_tensor = torch.tensor(gallery_feats, device=device, dtype=torch.float32)
    indices_list = []

    chunk_size = 1000
    for i in tqdm(range(0, len(query_feats), chunk_size), desc="Ranking Chunks"):
        q_chunk = torch.tensor(query_feats[i:i + chunk_size], device=device, dtype=torch.float32)
        dist_chunk = 1.0 - torch.mm(q_chunk, g_tensor.t())
        idx_chunk = torch.argsort(dist_chunk, dim=1).cpu().numpy()
        indices_list.append(idx_chunk)
        del q_chunk, dist_chunk
        torch.cuda.empty_cache()

    indices = np.concatenate(indices_list, axis=0)
    del g_tensor
    torch.cuda.empty_cache()
    print(f"[INFO] Ranking indices computed: {indices.shape}")

    print("\n[INFO] Running Multi-Protocol MEVID Evaluation...")
    evaluator = MEVIDEvaluator(query_metas, gallery_metas)

    protocols = ['general', 'same_outfit', 'cross_outfit']
    results = {}
    table_rows = []

    for proto in tqdm(protocols, desc="Protocols"):
        res = evaluator.evaluate(protocol=proto, indices=indices)
        results[proto] = res
        table_rows.append([
            proto.replace('_', ' ').title(),
            f"{res['cmc']['rank-1'] * 100:.2f}%",
            f"{res['cmc']['rank-5'] * 100:.2f}%",
            f"{res['cmc']['rank-10'] * 100:.2f}%",
            f"{res['mAP'] * 100:.2f}%",
        ])

    headers = ["Protocol", "Rank-1", "Rank-5", "Rank-10", "mAP"]
    print("\n" + "="*70)
    print(f"      MEVID MULTI-PROTOCOL EVALUATION: {cfg.MODEL.BACKBONE.NAME.upper()}")
    print("="*70)
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    print("="*70)


if __name__ == "__main__":
    main()
