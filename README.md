# SPECT Patient Motion Correction Using 3D Surface Tracking

Research project at UT TRAIL Lab evaluating surface-based motion estimation algorithms for correcting patient movement during SPECT imaging scans.

## Problem Statement

SPECT imaging requires patients to remain motionless for 15-30 minutes. Any head motion degrades image quality and diagnostic accuracy. This work develops and evaluates 3D camera-based tracking to detect and correct for patient motion in real-time.

## Approach

**Hardware:**
- Zivid 3D camera for surface capture (point cloud data)
- KUKA iiwa 14 robotic arm for ground truth motion generation
- Linear stage for controlled, precise motion validation

**Algorithms Evaluated:**
1. **ICP (Iterative Closest Point)** - Classic point cloud registration
2. **NSFP (Normal Space Filtering and Projection)** - Advanced surface-based method

## Repository Files

### Motion Estimation Algorithms
- `batch_icp.py` - ICP implementation with improved normal estimation and transform chaining
- `nsfp_zivid.py` - NSFP algorithm implementation for Zivid point cloud data

### Data Processing Pipeline
- `transform_and_filter_point_clouds.py` - Applies coordinate transformations and boundary box filtering to ICP output data
- `visualize_transformation.py` - Visualization tool used to determine optimal boundary box parameters for region of interest

### Evaluation & Calibration
- `evaluate_ground_truth.m` - MATLAB script comparing algorithm outputs against known linear stage motion
- `Transformation.m` - Coordinate frame transformation utilities
- `cameraToEndEffectorTform.mat` - Calibrated transformation between camera and robot end effector

## Results Summary

**Algorithm Performance (RMSE against ground truth, ~593 frames):**
- **NSFP**: 1.110mm (best honest estimate)
- **ICP**: ~2× overestimation (consistent positive bias)
- Geometric projection: 0.926mm (invalid - systematic underestimation, likely calibration issue)

**Motion Analysis:**
- PCA revealed dominant motion axis: [0, 0.990, -0.139] in world frame
- Motion primarily along vertical axis with slight forward tilt component

## Workflow

1. Capture point cloud sequences with Zivid camera
2. Run motion estimation (`batch_icp.py` or `nsfp_zivid.py`)
3. Transform results to appropriate coordinate frame (`transform_and_filter_point_clouds.py`)
4. Evaluate against ground truth linear stage data (`evaluate_ground_truth.m`)
5. Analyze performance metrics and motion characteristics

## Technical Improvements

**batch_icp.py v2 enhancements:**
- Improved normal vector estimation for point cloud surfaces
- Camera-oriented normals to ensure consistent surface direction
- Increased iteration count for better convergence
- Proper transform chaining across frame sequences to maintain global consistency

## Dependencies
- Python: NumPy, Open3D, Zivid SDK
- MATLAB: Point Cloud Toolbox

## Research Context
This work supports development of real-time motion correction for SPECT imaging at UT TRAIL Lab under the supervision of Lorenzo Confalonieri.

---

*Note: Point cloud datasets are not included in this repository due to file size constraints. Sample data available upon request.*
