"""MEVID Evaluator supporting general, same-outfit, and cross-outfit protocols."""

import numpy as np
from typing import List, Tuple, Dict, Any


class MEVIDEvaluator:
    """MEVID Dataset Evaluator."""

    def __init__(self, query_metas: List[Tuple[int, int, int]], gallery_metas: List[Tuple[int, int, int]]):
        """Initialize the evaluator.
        
        Args:
            query_metas: List of (pid, camid, outfit_id) for each query image.
            gallery_metas: List of (pid, camid, outfit_id) for each gallery image.
        """
        self.q_pids = np.array([m[0] for m in query_metas])
        self.q_camids = np.array([m[1] for m in query_metas])
        self.q_outfits = np.array([m[2] for m in query_metas])

        self.g_pids = np.array([m[0] for m in gallery_metas])
        self.g_camids = np.array([m[1] for m in gallery_metas])
        self.g_outfits = np.array([m[2] for m in gallery_metas])

    def evaluate(self, distmat: np.ndarray, protocol: str = 'general') -> Dict[str, Any]:
        """Evaluate the distance matrix under a specific protocol.
        
        Args:
            distmat: Numpy array of shape (num_query, num_gallery), distance matrix.
            protocol: 'general', 'same_outfit', or 'cross_outfit'.
            
        Returns:
            Dict containing 'mAP' and 'cmc' (list of rank-k accuracies).
        """
        num_q, num_g = distmat.shape
        if num_q != len(self.q_pids) or num_g != len(self.g_pids):
            raise ValueError(f"Distance matrix shape {distmat.shape} does not match meta data {len(self.q_pids)}x{len(self.g_pids)}")

        indices = np.argsort(distmat, axis=1)
        
        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            q_pid = self.q_pids[q_idx]
            q_cam = self.q_camids[q_idx]
            q_out = self.q_outfits[q_idx]

            order = indices[q_idx]
            g_pid = self.g_pids[order]
            g_cam = self.g_camids[order]
            g_out = self.g_outfits[order]

            is_same_pid = (g_pid == q_pid)
            is_same_cam = (g_cam == q_cam)
            is_same_out = (g_out == q_out)
            is_diff_out = (g_out != q_out)

            if protocol == 'general':
                valid = is_same_pid & ~is_same_cam
                junk = is_same_pid & is_same_cam
            elif protocol == 'same_outfit':
                valid = is_same_pid & is_same_out & ~is_same_cam
                junk = (is_same_pid & is_same_cam) | (is_same_pid & is_diff_out)
            elif protocol == 'cross_outfit':
                valid = is_same_pid & is_diff_out & ~is_same_cam
                junk = (is_same_pid & is_same_cam) | (is_same_pid & is_same_out)
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

            if not np.any(valid):
                continue
                
            num_valid_q += 1

            keep = ~junk
            valid_k = valid[keep]
            
            cmc = valid_k.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:50])

            num_rel = valid_k.sum()
            tmp_cmc = valid_k.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * valid_k
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        if num_valid_q == 0:
            return {'mAP': 0.0, 'cmc': {'rank-1': 0.0, 'rank-5': 0.0, 'rank-10': 0.0}}

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return {
            'mAP': float(mAP),
            'cmc': {
                'rank-1': float(all_cmc[0]) if len(all_cmc) > 0 else 0.0,
                'rank-5': float(all_cmc[4]) if len(all_cmc) > 4 else 0.0,
                'rank-10': float(all_cmc[9]) if len(all_cmc) > 9 else 0.0,
            }
        }
