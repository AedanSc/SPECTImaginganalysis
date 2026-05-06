"""
visualize_transformation_fixed.py
─────────────────────────────────────────────────────────────────────────────
Visualize transformation with proper camera positioning and zoom.
"""

import numpy as np
import open3d as o3d
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

# Camera to robot base transformation
T_camera_to_base = np.array([
    [-0.0576, -0.8382,  0.5424,  235.4],
    [-0.9929, -0.0087, -0.1190,  100.3],
    [ 0.1045, -0.5453, -0.8317,  814.7],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
])


# Bounding box for platform (in robot base frame, mm)
bbox_min = np.array([440, -75, 0])
bbox_max = np.array([540, 75, 1000])

# Select a sample frame
sample_frame = "point_clouds\point_clouds\point_cloud_20260305_140722_874707.ply"


# ─── Helper functions ─────────────────────────────────────────────────────────

def create_bounding_box_lineset(bbox_min, bbox_max, color=[1, 0, 0], line_width=3):
    """Create thick line set for bounding box visualization."""
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])
    
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def crop_to_bbox(pcd, bbox_min, bbox_max):
    """Crop point cloud to bounding box."""
    points = np.asarray(pcd.points)
    
    mask = (
        (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
        (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
        (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
    )
    
    pcd_cropped = o3d.geometry.PointCloud()
    pcd_cropped.points = o3d.utility.Vector3dVector(points[mask])
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd_cropped.colors = o3d.utility.Vector3dVector(colors[mask])
    
    return pcd_cropped


def visualize_with_camera(geometries, window_name, lookat, eye, up=[0, 0, 1]):
    """Visualize with custom camera position."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(eye - lookat)
    ctr.set_up(up)
    ctr.set_zoom(0.5)
    
    # Render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.line_width = 5.0
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    vis.run()
    vis.destroy_window()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Transformation and Bounding Box Visualization")
    print("=" * 70)
    
    # Check if file exists
    if not Path(sample_frame).exists():
        print(f"\n[ERROR] File not found: {sample_frame}")
        print("Please update the 'sample_frame' path in the script.")
        return
    
    # Load original
    print(f"\nLoading: {sample_frame}")
    pcd_original = o3d.io.read_point_cloud(sample_frame)
    print(f"  Original points: {len(pcd_original.points):,}")
    
    if len(pcd_original.points) == 0:
        print("[ERROR] Point cloud is empty!")
        return
    
    # Get original bounds
    points_orig = np.asarray(pcd_original.points)
    orig_center = points_orig.mean(axis=0)
    orig_extent = points_orig.max(axis=0) - points_orig.min(axis=0)
    
    print(f"\n  Original center: [{orig_center[0]:.1f}, {orig_center[1]:.1f}, {orig_center[2]:.1f}]")
    print(f"  Original extent: [{orig_extent[0]:.1f}, {orig_extent[1]:.1f}, {orig_extent[2]:.1f}]")
    
    # Transform
    print("\nApplying transformation...")
    pcd_transformed = o3d.geometry.PointCloud(pcd_original)
    pcd_transformed.transform(T_camera_to_base)
    
    points_trans = np.asarray(pcd_transformed.points)
    trans_center = points_trans.mean(axis=0)
    
    print(f"  Transformed center: [{trans_center[0]:.1f}, {trans_center[1]:.1f}, {trans_center[2]:.1f}]")
    
    # Crop
    print("\nCropping to bounding box...")
    pcd_cropped = crop_to_bbox(pcd_transformed, bbox_min, bbox_max)
    print(f"  Cropped points: {len(pcd_cropped.points):,}")
    
    if len(pcd_cropped.points) == 0:
        print("[WARNING] No points inside bounding box!")
        print("  Check if bounding box matches your data.")
    
    # Create visualizations
    bbox_lines = create_bounding_box_lineset(bbox_min, bbox_max, color=[1, 0, 0])
    frame_small = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    frame_large = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
    
    # Color the point clouds
    pcd_original.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_transformed.paint_uniform_color([0.2, 0.4, 0.8])
    pcd_cropped.paint_uniform_color([0.1, 0.6, 0.1])
    
    print("\n" + "=" * 70)
    print("  Starting Visualizations")
    print("=" * 70)
    
    # VIS 1: Original (Camera Frame)
    print("\n[1/3] BEFORE: Camera Frame")
    print("      Gray points = original point cloud")
    print("      Close window to continue...")
    
    visualize_with_camera(
        [pcd_original, frame_small],
        "BEFORE: Camera Frame (Original)",
        lookat=orig_center,
        eye=orig_center + np.array([500, 500, 500])
    )
    
    # VIS 2: Transformed with Bounding Box
    print("\n[2/3] AFTER: Robot Base Frame + Bounding Box")
    print("      Blue points = transformed point cloud")
    print("      RED BOX = platform bounding box")
    print("      Close window to continue...")
    
    bbox_center = (bbox_min + bbox_max) / 2
    
    visualize_with_camera(
        [pcd_transformed, bbox_lines, frame_large],
        "AFTER: Transformed + Bounding Box (RED)",
        lookat=bbox_center,
        eye=bbox_center + np.array([800, 800, 800])
    )
    
    # VIS 3: Cropped Only
    print("\n[3/3] CROPPED: Platform Only")
    print("      Green points = cropped to platform")
    print("      This is what ICP uses")
    print("      Close window to finish...")
    
    if len(pcd_cropped.points) > 0:
        crop_center = np.asarray(pcd_cropped.points).mean(axis=0)
        visualize_with_camera(
            [pcd_cropped, bbox_lines, frame_large],
            "CROPPED: Platform Only (Used for ICP)",
            lookat=crop_center,
            eye=crop_center + np.array([400, 400, 400])
        )
    else:
        print("      [SKIPPED - No points in bounding box]")
    
    # Statistics
    print("\n" + "=" * 70)
    print("  Statistics")
    print("=" * 70)
    print(f"  Original:    {len(pcd_original.points):,} points")
    print(f"  Transformed: {len(pcd_transformed.points):,} points")
    print(f"  Cropped:     {len(pcd_cropped.points):,} points")
    if len(pcd_original.points) > 0:
        print(f"  Reduction:   {100*(1 - len(pcd_cropped.points)/len(pcd_original.points)):.1f}%")
    print("=" * 70)
    
    print("\n[DONE] Use mouse to rotate, scroll to zoom, take screenshots!")


if __name__ == "__main__":
    main()