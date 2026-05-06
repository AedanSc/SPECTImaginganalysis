"""
batch_icp_bidirectional.py
Run ICP forward AND backward, then average to cancel systematic bias.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Bidirectional ICP")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--voxel", type=float, default=1.5)
    parser.add_argument("--output", default="icp_results_bidirectional.csv")
    parser.add_argument("--max-displacement", type=float, default=10.0)
    parser.add_argument("--min-fitness", type=float, default=0.6)
    return parser.parse_args()


def load_pcd(filepath: Path, voxel_size: float):
    pcd = o3d.io.read_point_cloud(str(filepath))
    if len(pcd.points) == 0:
        return None
    
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if len(pcd.points) < 100:
        return None
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    
    if len(pcd.points) < 50:
        return None
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 7, max_nn=50
        )
    )
    pcd.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
    
    return pcd


def run_bidirectional_icp(pc1, pc2, threshold):
    """Run ICP in both directions and average the translation."""
    
    # Forward: pc1 → pc2
    result_fwd = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    # Backward: pc2 → pc1
    result_bwd = o3d.pipelines.registration.registration_icp(
        pc2, pc1, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    # Extract translations
    t_fwd = result_fwd.transformation[:3, 3]
    t_bwd = -result_bwd.transformation[:3, 3]  # Invert to match forward direction
    
    # Average to cancel bias
    t_avg = (t_fwd + t_bwd) / 2.0
    
    # Use best quality metrics
    fitness = max(result_fwd.fitness, result_bwd.fitness)
    rmse = min(result_fwd.inlier_rmse, result_bwd.inlier_rmse)
    
    # Compute rotation (use forward)
    rotation_matrix = result_fwd.transformation[:3, :3]
    trace = np.trace(rotation_matrix)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    rotation_angle = np.degrees(np.arccos(cos_angle))
    
    return t_avg, fitness, rmse, rotation_angle


def is_outlier(displacement: float, fitness: float, max_disp: float, min_fit: float):
    return displacement > max_disp or fitness < min_fit


def main():
    args = parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Folder not found: {folder}")
    
    ply_files = sorted(folder.glob("*.ply"))
    if len(ply_files) < 2:
        sys.exit(f"[ERROR] Need at least 2 files")
    
    print("=" * 70)
    print("  Bidirectional ICP (Cancels Systematic Bias)")
    print("=" * 70)
    print(f"  Folder: {folder}")
    print(f"  Frames: {len(ply_files)}")
    print(f"  Method: Forward + Backward ICP, averaged")
    print("=" * 70)
    
    results = []
    total_displacement = 0.0
    failed = 0
    outliers = 0
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = None
        
        for i in range(len(ply_files) - 1):
            f1 = ply_files[i]
            f2 = ply_files[i + 1]
            
            # Parse timestamps
            try:
                t1 = datetime.strptime(f1.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
                t2 = datetime.strptime(f2.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
                dt_ms = (t2 - t1).total_seconds() * 1000
            except ValueError:
                dt_ms = None
            
            # Load
            source = load_pcd(f1, args.voxel)
            target = load_pcd(f2, args.voxel)
            
            if source is None or target is None:
                print(f"  [{i+1:04d}/{len(ply_files)-1}] FAILED - empty")
                failed += 1
                continue
            
            # Run bidirectional ICP
            translation, fitness, rmse, rotation = run_bidirectional_icp(
                source, target, args.threshold
            )
            
            displacement = np.linalg.norm(translation)
            
            # Check outliers
            is_outlier_flag = is_outlier(displacement, fitness, args.max_displacement, args.min_fitness)
            
            if is_outlier_flag:
                outliers += 1
                status = "OUTLIER"
            else:
                status = "OK"
                total_displacement += displacement
            
            # Print
            dt_str = f"{dt_ms:.1f}ms" if dt_ms else "N/A"
            print(f"  [{i+1:04d}/{len(ply_files)-1}] {f1.name}")
            print(f"           Δt={dt_str}  disp={displacement:.3f}mm  fitness={fitness:.3f}  [{status}]")
            
            row = {
                "pair": i + 1,
                "frame1": f1.name,
                "frame2": f2.name,
                "dt_ms": dt_ms,
                "displacement_mm": displacement,
                "tx_mm": translation[0],
                "ty_mm": translation[1],
                "tz_mm": translation[2],
                "rotation_deg": rotation,
                "fitness": fitness,
                "rmse": rmse,
                "outlier": is_outlier_flag,
            }
            results.append(row)
            
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                writer.writeheader()
            
            writer.writerow(row)
            csvfile.flush()
    
    # Summary
    if results:
        valid_results = [r for r in results if not r['outlier']]
        
        if valid_results:
            displacements = [r["displacement_mm"] for r in valid_results]
            print("\n" + "═" * 70)
            print("  Motion Summary (Bidirectional ICP)")
            print("═" * 70)
            print(f"  Total pairs       : {len(results)}")
            print(f"  Valid pairs       : {len(valid_results)}")
            print(f"  Outliers rejected : {outliers}")
            print(f"  Failed pairs      : {failed}")
            print(f"  Mean displacement : {np.mean(displacements):.3f} mm")
            print(f"  Std deviation     : {np.std(displacements):.3f} mm")
            print(f"  Total displacement: {total_displacement:.3f} mm")
            print("═" * 70)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()