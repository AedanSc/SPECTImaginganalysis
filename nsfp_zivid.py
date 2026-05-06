"""
batch_nsfp_corrected.py
Use the ORIGINAL nsfp_zivid.py code (which works) on transformed+cropped data
"""

import sys
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


# ─── Copy EXACT code from nsfp_zivid.py ──────────────────────────────────────

class NSFPrior(nn.Module):
    """EXACT same model as original"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, pc):
        return self.net(pc)


def load_pcd(filepath: str, npoints: int, device: torch.device):
    """EXACT same loading as original"""
    pcd = o3d.io.read_point_cloud(str(filepath))
    
    if len(pcd.points) == 0:
        return None
    
    points = np.asarray(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) >= npoints:
        idx = np.random.choice(len(points), npoints, replace=False)
    else:
        idx = np.random.choice(len(points), npoints, replace=True)
    
    points = points[idx]
    return torch.tensor(points, dtype=torch.float32).to(device)


def normalize_pcds(pc1, pc2):
    """EXACT same normalization as original"""
    combined = torch.cat([pc1, pc2], dim=0)
    centroid = combined.mean(dim=0)
    scale = (combined - centroid).abs().max()
    
    pc1_norm = (pc1 - centroid) / scale
    pc2_norm = (pc2 - centroid) / scale
    
    return pc1_norm, pc2_norm, centroid, scale


def chamfer_loss(pc1_warped, pc2, batch_size=2000):
    """EXACT same as original"""
    total_loss = torch.tensor(0.0, device=pc1_warped.device)
    n = len(pc1_warped)
    
    for i in range(0, n, batch_size):
        batch = pc1_warped[i:i+batch_size]
        dist = torch.cdist(batch.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0)
        min_dist, _ = dist.min(dim=1)
        total_loss += min_dist.mean()
    
    return total_loss / (n / batch_size)


def smoothness_loss(flow, pc1, k=8, batch_size=5000):
    """EXACT same as original"""
    total_loss = torch.tensor(0.0, device=flow.device)
    n = len(pc1)
    
    for i in range(0, min(n, 10000), batch_size):
        batch_pts = pc1[i:i+batch_size]
        batch_flow = flow[i:i+batch_size]
        
        dist = torch.cdist(batch_pts.unsqueeze(0), pc1.unsqueeze(0)).squeeze(0)
        _, nn_idx = dist.topk(k+1, dim=1, largest=False)
        nn_idx = nn_idx[:, 1:]
        
        nn_flow = flow[nn_idx]
        smooth = (batch_flow.unsqueeze(1) - nn_flow).norm(dim=2).mean()
        total_loss += smooth
    
    return total_loss / (min(n, 10000) / batch_size)


def optimize_flow(pc1, pc2, iters, lr, device):
    """EXACT same as original"""
    model = NSFPrior(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
    
    best_loss = float('inf')
    best_flow = None
    
    for i in range(iters):
        optimizer.zero_grad()
        
        flow = model(pc1)
        pc1_warped = pc1 + flow
        
        loss_chamfer = chamfer_loss(pc1_warped, pc2)
        loss_smooth = smoothness_loss(flow, pc1)
        loss = loss_chamfer + 0.1 * loss_smooth
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_flow = flow.detach().clone()
    
    return best_flow


# ─── Batch processing ─────────────────────────────────────────────────────────

def main():
    folder = Path("platform_only/")
    ply_files = sorted(folder.glob("*.ply"))
    
    # TEST: Only first 20 pairs
    num_pairs = 20
    ply_files = ply_files[:num_pairs + 1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("  NSFP with ORIGINAL code on transformed+cropped data")
    print("=" * 70)
    print(f"  Folder: {folder}")
    print(f"  Pairs:  {num_pairs}")
    print(f"  Device: {device}")
    print("=" * 70)
    
    results = []
    
    with open("nsfp_original_code_test.csv", 'w', newline='') as csvfile:
        writer = None
        
        for i in range(num_pairs):
            f1 = ply_files[i]
            f2 = ply_files[i + 1]
            
            print(f"\n[{i+1}/{num_pairs}] {f1.name}")
            
            # Parse timestamps
            try:
                t1 = datetime.strptime(f1.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
                t2 = datetime.strptime(f2.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
                dt_ms = (t2 - t1).total_seconds() * 1000
            except:
                dt_ms = None
            
            # Load with ORIGINAL function
            pc1 = load_pcd(str(f1), 5000, device)
            pc2 = load_pcd(str(f2), 5000, device)
            
            if pc1 is None or pc2 is None:
                continue
            
            # Normalize with ORIGINAL function
            pc1_norm, pc2_norm, centroid, scale = normalize_pcds(pc1, pc2)
            
            # Optimize with ORIGINAL function
            flow = optimize_flow(pc1_norm, pc2_norm, 500, 0.001, device)
            
            # Denormalize with ORIGINAL method
            flow_np = flow.cpu().numpy()
            flow_mm = flow_np * scale.cpu().numpy()
            magnitudes = np.linalg.norm(flow_mm, axis=1)
            
            mean_disp = magnitudes.mean()
            print(f"  Mean displacement: {mean_disp:.3f} mm")
            
            row = {
                'pair': i + 1,
                'frame1': f1.name,
                'frame2': f2.name,
                'dt_ms': dt_ms,
                'mean_displacement_mm': mean_disp,
                'max_displacement_mm': magnitudes.max(),
                'tx_mm': flow_mm[:, 0].mean(),
                'ty_mm': flow_mm[:, 1].mean(),
                'tz_mm': flow_mm[:, 2].mean(),
            }
            
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                writer.writeheader()
            
            writer.writerow(row)
            csvfile.flush()
    
    print("\n[DONE] Results saved to nsfp_original_code_test.csv")


if __name__ == "__main__":
    main()