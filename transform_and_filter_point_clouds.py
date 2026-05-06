# transform_and_filter_point_clouds.py
"""
Transform point clouds to robot base frame and apply platform bounding box.
Creates filtered point clouds ready for ICP processing.

Usage:
    python transform_and_filter_point_clouds.py --folder point_clouds/point_clouds/ --output point_clouds/point_clouds_filtered/
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from tqdm import tqdm


# ─── CALIBRATED Camera-to-Base Transformation ────────────────────────────────
CAMERA_TO_BASE = np.array([
    [-0.0576, -0.8382,  0.5424,  235.4],
    [-0.9929, -0.0087, -0.1190,  100.3],
    [ 0.1045, -0.5453, -0.8317,  814.7],
    [ 0.0000,  0.0000,  0.0000,  1.00],
])

# ─── Platform Bounding Box (Robot Base Frame) ────────────────────────────────
PLATFORM_X_MIN = 475.0  # mm
PLATFORM_X_MAX = 525.0  # mm
PLATFORM_Y_MIN = -25.0  # mm
PLATFORM_Y_MAX = 25.0   # mm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform and filter point clouds to platform region"
    )
    parser.add_argument("--folder", required=True, help="Input folder with.ply files")
    parser.add_argument("--output", required=True, help="Output folder for filtered point clouds")
    parser.add_argument("--visualize", action="store_true", help="Visualize first frame")
    return parser.parse_args()


def transform_and_filter(pcd):
    """Transform to base frame and apply platform bounding box."""
    
    # Transform to robot base frame
    pcd_base = o3d.geometry.PointCloud(pcd)
    pcd_base.transform(CAMERA_TO_BASE)
    
    # Apply XY bounding box
    points = np.asarray(pcd_base.points)
    mask = (
        (points[:, 0] > PLATFORM_X_MIN) & (points[:, 0] < PLATFORM_X_MAX) &
        (points[:, 1] > PLATFORM_Y_MIN) & (points[:, 1] < PLATFORM_Y_MAX)
    )
    
    pcd_filtered = pcd_base.select_by_index(np.where(mask)[0])
    
    return pcd_filtered, mask.sum(), len(points)


def main():
    args = parse_args()
    
    input_dir = Path(args.folder)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        sys.exit(f"[ERROR] Input folder not found: {input_dir}")
    
    ply_files = sorted(input_dir.glob("*.ply"))
    
    if len(ply_files) == 0:
        sys.exit(f"[ERROR] No.ply files found in {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  Transform & Filter Point Clouds")
    print("=" * 70)
    print(f"  Input folder  : {input_dir}")
    print(f"  Output folder : {output_dir}")
    print(f"  Files found   : {len(ply_files)}")
    print(f"  Platform bbox : X=[{PLATFORM_X_MIN}, {PLATFORM_X_MAX}], Y=[{PLATFORM_Y_MIN}, {PLATFORM_Y_MAX}]")
    print("=" * 70 + "\n")
    
    success = 0
    total_points_before = 0
    total_points_after = 0
    
    for i, filepath in enumerate(tqdm(ply_files, desc="Processing")):
        try:
            # Load
            pcd = o3d.io.read_point_cloud(str(filepath))
            
            if len(pcd.points) == 0:
                continue
            
            # Transform and filter
            pcd_filtered, kept, total = transform_and_filter(pcd)
            
            if len(pcd_filtered.points) == 0:
                print(f"  [WARN] No points after filtering: {filepath.name}")
                continue
            
            # Save
            output_path = output_dir / filepath.name
            o3d.io.write_point_cloud(str(output_path), pcd_filtered)
            
            total_points_before += total
            total_points_after += kept
            success += 1
            
            # Visualize first frame
            if args.visualize and i == 0:
                print(f"\n[INFO] Visualizing first frame: {filepath.name}")
                print(f"  Original: {total:,} points")
                print(f"  Filtered: {kept:,} points ({kept/total*100:.1f}%)")
                
                pcd_filtered_vis = o3d.geometry.PointCloud(pcd_filtered)
                pcd_filtered_vis.paint_uniform_color([0.2, 1.0, 0.2])
                
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])
                
                o3d.visualization.draw_geometries([pcd_filtered_vis, coord], 
                                                   window_name="Filtered platform (green)")
            
        except Exception as e:
            print(f"  [ERROR] {filepath.name}: {e}")
    
    print("\n" + "=" * 70)
    print("  Processing Complete")
    print("=" * 70)
    print(f"  Successfully processed: {success}/{len(ply_files)}")
    print(f"  Total points before: {total_points_before:,}")
    print(f"  Total points after:  {total_points_after:,}")
    print(f"  Reduction: {(1 - total_points_after/total_points_before)*100:.1f}%")
    print(f"  Output saved to: {output_dir.resolve()}")
    print("=" * 70)
    print("\nNext step: Run ICP on the filtered point clouds:")
    print(f"  python batch_icp.py --folder {output_dir} --voxel 1.0 --threshold 1.5 --output icp_results_filtered.csv")


if __name__ == "__main__":
    main()