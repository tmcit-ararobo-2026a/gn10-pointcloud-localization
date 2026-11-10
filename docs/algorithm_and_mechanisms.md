# gn10-pointcloud-localization: Processing Algorithms & Invariant Mechanics Reference

[Japanese](./algorithm_and_mechanisms_jp.md)

This document provides an in-depth reference on the processing algorithms, internal architecture, mathematical formulations, and invariant mechanics (mathematical properties, control stability guarantees, and boundary conditions) of `gn10-pointcloud-localization`. The system fuses 3D LiDAR (Livox MID360) and 2D planar LiDAR (Lakibeam 1) over a GPU-accelerated Euclidean Signed Distance Field (ESDF), ensuring robust, oscillation-free, and fail-safe pose estimation (`map -> base_link` TF) even under extreme angular rates and high translational velocities.

---

## 1. System Architecture & Core Design Principles

Traditional point cloud registration pipelines (e.g., iterative closest point or normal distributions transform) often suffer from sensitivity to dynamic obstacles, sensor calibration/timing discrepancies, tracking loss during rapid maneuvers, and runaway behavior caused by extrapolating unverified poses during matching failure.

`gn10-pointcloud-localization` is built around four fundamental invariant principles to resolve these challenges:
1. **Spatial Representation & $\mathcal{O}(1)$ Distance Field Query**: A precomputed 3D ESDF is stored in NVIDIA GPU 3D Texture Memory, allowing hardware-accelerated trilinear interpolation to query nearest-surface distances in constant $\mathcal{O}(1)$ time.
2. **Strict Spatiotemporal Rigidity**: Every point from 3D point clouds and 2D planar scans is deskewed using high-frequency IMU integration and linear velocity extrapolation to a single, consistent reference timestamp $t_{\text{ref}}$ in the robot coordinate frame.
3. **Independent Residual Evaluation with Two-Guard Fallback**: Surface residuals of 3D and 2D clouds are computed independently. A hysteresis deadband and a local continuity window prevent state chattering, spatial jumping, and limit-cycle oscillations between combined, 3D-only, and 2D-only modes.
4. **Safety-First Fail-Safe TF Gating**: When a scan fails acceptance criteria, the system immediately ceases TF and pose broadcasting rather than extrapolating on dead reckoning, directly tripping downstream controller safety timeouts.

### Architecture Diagrams (PlantUML)
- **End-to-End Pipeline & Parameters**: [pipeline_flow_and_parameters.pu](pipeline_flow_and_parameters.pu)
- **Cost Formulation, Two-Guard Fallback & Acceptance Logic**: [matching_cost_and_weighting.pu](matching_cost_and_weighting.pu)

---

## 2. Preprocessing & Spatiotemporal Motion Compensation

```
[Livox 3D LiDAR] ──> ScanAccumulator (window_s) ──> filterGroundWithDeskewKernel (CUDA)
[Livox 6-DoF IMU] ──> YawHistory (trapezoidal)  ───┘ (Rotational + Translational Deskew)
[Lakibeam 2D LiDAR] ─> deskewLaserScan ─────────> alignPointsToTimestamp (dt_sync)
                                                           │
                                                           ▼
                                                d_obstacle_ (CUDA unified memory)
```

### 2.1 3D Scan Accumulation and Dual Motion Deskew
A 3D LiDAR sweeps continuously over its measurement window (100 ms). During fast robot motion, this creates geometric distortion if points are assumed simultaneous.

- **Timestamp Extraction & Reference Selection**:
  The per-point timestamp $t_i$ is extracted for every point $(x_i, y_i, z_i)$ in `scan_accumulation.window_s`. The newest timestamp is designated as the reference timestamp $t_{\text{ref}}$.
- **IMU Angular Velocity Quadratic Interpolation**:
  In CUDA kernel `filterGroundWithDeskewKernel`, the rotation $\Delta \mathbf{R}(t_i \to t_{\text{ref}})$ is computed by binary-searching a 200 Hz IMU buffer and applying trapezoidal yaw integration.
- **Translational Motion Deskew**:
  Using estimated body velocity $\mathbf{v}_{\text{local}}$, the displacement $\Delta \mathbf{p} = \mathbf{v}_{\text{local}} \cdot (t_{\text{ref}} - t_i)$ (clamped to $\pm 100\text{ ms}$) is subtracted.
- **Invariant Effect**:
  Under translational speeds $> 2\text{ m/s}$ and angular rates $> 0.5\text{ rad/s}$, geometric blurring and artificial wall thickening are eliminated. Distance residuals to true static CAD/ESDF boundaries converge to the hardware sensor noise baseline.

### 2.2 Geometric Filtering & Ground Removal (`GroundFilter`)
Immediately following deskew, points are partitioned on GPU:
- **Robot Self-Exclusion Boundary**: Discards points satisfying $x^2 + y^2 \le r_{\text{robot}}^2$ and $z \in [z_{\min}, z_{\max}]$ (bumpers, wheels, upper linkages).
- **Maximum Range Boundary**: Discards points with $\|\mathbf{p}\| > d_{\max,\text{range}}$.
- **Ground Separation**: Points satisfying $z \le z_{\text{ground}}$ are classified as ground and omitted from the obstacle buffer $N_{\text{3D}}$.
- **Invariant Effect**:
  Floor seams, small field surface steps, and chassis self-reflections cannot contaminate obstacle data or corrupt horizontal/vertical ESDF alignment gradients.

### 2.3 2D LiDAR Deskew and Cross-Sensor Time Alignment
Planar LiDAR (Lakibeam 1, 30 Hz) sweeps beams sequentially:
- **Per-Beam Deskew (`deskewLaserScan`)**:
  Compensates rotational and translational motion for each beam angle at $t_j = t_{\text{start}} + j \cdot \Delta t_{\text{scan}}$ to the scan end.
- **Cross-Sensor Time Synchronization (`alignPointsToTimestamp`)**:
  For time difference $\Delta t_{\text{sync}} = t_{\text{ref}} - t_{\text{2D}}$ ($| \Delta t_{\text{sync}} | < 0.20\text{ s}$), points are projected to $t_{\text{ref}}$ via IMU rotation and velocity integration, then appended after 3D obstacles in `d_obstacle_`.
- **Invariant Effect**:
  Points from distinct sensors with disparate update rates form a single rigid body evaluated simultaneously on GPU without inter-sensor latency skew.

---

## 3. Motion Prediction & Dynamic Search Window Geometry

### 3.1 Search Center Prediction (`search_base_pose`)
The search center at $t_{\text{ref}}$ is predicted from previous accepted pose $P_{k-1} = (x_{k-1}, y_{k-1}, \psi_{k-1})$:
$$\psi_{\text{pred}} = \psi_{k-1} + \Delta \psi_{\text{imu}}(t_{k-1} \to t_{\text{ref}})$$
$$\mathbf{p}_{\text{pred}} = \mathbf{p}_{k-1} + \mathbf{v}_{\text{ema}} \cdot (t_{\text{ref}} - t_{k-1})$$
where $\mathbf{v}_{\text{ema}}$ is the Exponential Moving Average ($\alpha = 0.6$) velocity:
$$\mathbf{v}_k = \alpha \, \mathbf{v}_{\text{meas}} + (1 - \alpha) \, \mathbf{v}_{k-1}$$
- **Invariant Effect**:
  EMA suppresses discrete measurement noise while tracking acceleration dynamics, ensuring the true pose stays centered within the local search volume.

### 3.2 Adaptive Window Scaling under Velocity & Turn Rate
The candidate grid bounds expand adaptively based on state dynamics:
1. **Translational Velocity Scaling**:
   When $\|\mathbf{v}\| \ge 2.0\text{ m/s}$, the XY search radius expands linearly up to 1.5x:
   $$\text{scale}_{xy} = \min\left(1.5,\; 1.0 + 0.5 \times \frac{\|\mathbf{v}\| - 2.0}{v_{\max} - 2.0}\right), \qquad r_{xy} = r_{xy,\text{base}} \times \text{scale}_{xy}$$
2. **Dynamic Yaw Expansion (`dynamic_yaw_expansion`)**:
   When $|\omega_z| \ge \omega_{\text{thresh}}$ ($0.5\text{ rad/s}$), the yaw radius expands:
   $$r_{\psi} = \min\left(r_{\psi,\max},\; r_{\psi,\text{base}} + (|\omega_z| - \omega_{\text{thresh}}) \cdot k_{\text{factor}}\right)$$
- **Invariant Effect**:
  During stationary or low-speed motion, candidate grid size is minimized for lowest GPU latency. During abrupt maneuvers, the discretized volume expands to bound the motion uncertainty ellipsoid, preventing search-window starvation.

---

## 4. GPU 3D Texture ESDF Matching & Multi-Resolution Search

### 4.1 $\mathcal{O}(1)$ ESDF Query & Robust Outlier Clipping
For each candidate pose $P_k = (r_x, r_y, r_{\psi})$, CUDA kernel `evaluateESDFKernel` transforms points into map frame:
$$w_{x,i} = \cos(r_{\psi}) l_{x,i} - \sin(r_{\psi}) l_{y,i} + r_x$$
$$w_{y,i} = \sin(r_{\psi}) l_{x,i} + \cos(r_{\psi}) l_{y,i} + r_y$$
$$w_{z,i} = l_{z,i}$$

Distance $d_i$ is fetched from GPU 3D Texture memory via hardware trilinear interpolation. Residual cost $c_i$ is truncated:
$$c_i = \begin{cases} \min(d_i, d_{\max}) & \text{if } (w_{x,i}, w_{y,i}) \in \text{FieldBounds} \\ d_{\max} & \text{otherwise} \end{cases}$$
where $d_{\max} = \text{max\_dist\_thresh}$.

- **Invariant Effect**:
  For points where $d_i > d_{\max}$ (dynamic obstacles such as robots or persons), the gradient contribution $\frac{\partial c_i}{\partial P}$ is zero, imposing only a bounded constant penalty $d_{\max}$. The basin of attraction of static field boundaries remains undisturbed.

### 4.2 Multi-Resolution Hierarchical Convergence
Search proceeds in three parallelized stages using `cub::DeviceReduce::ArgMin`:
1. **Coarse Level**: Evaluates candidate grid across $\pm r_{xy}$ ($s_{xy} = 0.05\text{ m}$) and $\pm r_{\psi}$ ($s_{\psi} = 0.03\text{ rad}$).
2. **Fine Level 1 (1/5x)**: Evaluates a $5 \times 5 \times 5$ local grid centered at the coarse winner ($s_{xy} = 0.01\text{ m}$, $s_{\psi} = 0.006\text{ rad}$).
3. **Fine Level 2 (1/25x)**: Evaluates a $5 \times 5 \times 5$ grid centered at Level 1 winner ($s_{xy} = 0.002\text{ m}$, $s_{\psi} = 0.0012\text{ rad} \approx 0.07^\circ$).

- **Invariant Effect (Shannon-Nyquist Spatial Compliance)**:
  Because coarse grid step $s_{xy} = 0.05\text{ m}$ is strictly smaller than the attraction basin radius $d_{\max} = 0.25\text{ m}$ ($s_{xy} \ll d_{\max}$), the global minimum cannot fall into an undetected inter-voxel notch. The pipeline achieves 2 mm and $0.07^\circ$ precision in constant, bounded computation time.

---

## 5. 3D/2D Hybrid Fusion & Anti-Oscillation Mechanics

### 5.1 Independent Residual Formulation & FoV Scaling
Mean distance residuals for 3D ($N_{\text{3D}}$) and 2D ($N_{\text{2D}}$) points are computed independently:
$$C_{\text{3D}}(P_k) = \frac{1}{N_{\text{3D}}} \sum_{i=0}^{N_{\text{3D}}-1} c_i, \qquad C_{\text{2D}}(P_k) = \frac{1}{N_{\text{2D}}} \sum_{j=N_{\text{3D}}}^{N_{\text{3D}}+N_{\text{2D}}-1} c_j$$
$$C_{\text{total}}(P_k) = (1 - w_{\text{2D}}) \, C_{\text{3D}}(P_k) + w_{\text{2D}} \, C_{\text{2D}}(P_k)$$
$$w_{\text{2D}} = \text{weight\_2d\_lidar} \times s_{\text{fov}}, \qquad s_{\text{fov}} = \min\left(1.0,\; \frac{N_{\text{2D}}}{\text{target\_2d\_points}}\right)$$

- **Invariant Effect**:
  When 2D LiDAR is partially occluded, $s_{\text{fov}} \to 0$ smoothly and continuously dampens 2D influence without discontinuous weight switching.

```
                    ┌─────────────────────────┐
                    │ Step A: Combined (3D+2D)│
                    │ w_2d = weight * s_fov   │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          │ pass_3d == true AND                       │ !pass_3d OR
          │ cost_2d < 2.0 * cost_threshold_2d         │ cost_2d >= 2.0 * cost_threshold_2d
          ▼                                           ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│ NORMAL ACCEPTANCE       │               │ Guard 1: Trigger 3D-Only│
│ (No mode switch)        │               │ w_2d = 0.0 (Full Window)│
└─────────────────────────┘               └───────────┬─────────────┘
                                                      │
                                ┌─────────────────────┴─────────────────────┐
                                │ pass_3d == true                           │ !pass_3d AND N_3d >= min_3d
                                ▼                                           ▼
                      ┌───────────────────┐                       ┌─────────────────────────┐
                      │ 3D-ONLY ACCEPTED  │                       │ Guard 2: 2D-Only        │
                      │ (2D isolated)     │                       │ Tight Continuity Window │
                      └───────────────────┘                       │ (+/-2*step_xy/yaw)      │
                                                                  └─────────────────────────┘
```

### 5.2 Guard 1: Hysteresis Deadband on 2D Detachment
If minor extrinsic misalignment exists, fluctuations of $C_{\text{2D}}$ near threshold can cause high-frequency chattering between Combined and 3D-only modes.

The system applies a **hysteresis deadband**:
- **3D-Only Retry Trigger**:
  `!pass_3d` (2D pulled 3D pose out of acceptance)
  **OR** $C_{\text{2D}} \ge 2.0 \times \text{cost\_threshold\_2d}$ (severe 2D corruption / occlusion spike).
- **Invariant Effect**:
  When $C_{\text{2D}} \in [\text{cost\_threshold\_2d},\; 2.0 \times \text{cost\_threshold\_2d})$ and 3D passes, Combined mode persists and passes via `pass_3d`. Schmitt-trigger hysteresis eliminates mode toggling and limit-cycle oscillation at the threshold boundary.

### 5.3 Guard 2: Continuity Bounded Window for 2D Fallback
If 3D is occluded and fails (`!pass_3d`), 2D-only fallback ($w_{\text{2D}} = 1.0$) is attempted.

Because 2D LiDAR observes only a single plane ($z \approx 0.08\text{ m}$), an unrestricted search window can jump to parallel repeating walls, resulting in a 20-30 cm false lock. When 3D returns in the next frame, it snaps back, inducing severe position oscillation.

When $N_{\text{3D}} \ge \text{min\_3d\_points}$, 2D-only search is physically bounded to **2 grid steps from the predicted pose**:
$$r_{xy,\text{2D-only}} \le 2 \, s_{xy}, \qquad r_{\psi,\text{2D-only}} \le 2 \, s_{\psi}$$
- **Invariant Effect**:
  Maximum single-frame displacement is strictly bounded to $2 s_{xy}$ (`0.10 m`) and $2 s_{\psi}$ (`0.06 rad`). 2D fallback cannot jump to distant aliased walls, structurally eliminating teleportation and oscillation upon 3D recovery.

---

## 6. Match Acceptance & Safety Fail-Safe Architecture

### 6.1 Dual Acceptance Criteria (`isMatchAccepted`)
A match is accepted (`match_accepted = true`) only under one of two mutually rigorous criteria:

1. **Primary Cost Criterion (`pass_cost`)**:
   - $N_{\text{3D}} \ge \text{min\_3d\_points}$ AND $C_{\text{3D}} < \text{cost\_threshold}$
   - OR $N_{\text{2D}} \ge \text{min\_2d\_points}$ AND $C_{\text{2D}} < \text{cost\_threshold\_2d}$
2. **Consensus Inlier Criterion (`pass_inliers`)**:
   In crowded environments, static walls may be perfectly aligned while high-distance dynamic obstacles drive mean cost $C_{\text{3D}}$ above threshold. The frame is accepted if all 4 conditions hold simultaneously:
   - $N_{\text{inlier}} \ge \text{min\_inlier\_points}$ (sufficient static wall returns)
   - $C_{\text{inlier}} < \text{max\_inlier\_accept\_cost}$ (wall points exhibit tight surface alignment)
   - $N_{\text{inlier}} / N_{\text{3D}} \ge \text{inlier\_ratio\_threshold}$ (sufficient fraction of total points represents static field)
   - $C_{\text{3D}} < \text{cost\_threshold}$ (global cost remains within bounding limit)

- **Invariant Effect**:
  Prevents false rejection in crowded fields while guaranteeing consensus to prevent partial-wall false locks.

### 6.2 Zero Extrapolation on Failure (Safety TF Halting)
When match acceptance fails, TF publication behavior enforces system safety:
- **Transient Failure (1–2 frames)**: Short-term interpolation ($\le 0.10\text{ s}$) via IMU and velocity prediction maintains continuity through momentary drops.
- **Persistent Failure ($\ge 3$ frames or Lost)**:
  **Immediately halts `map -> base_link` TF and `/platform_constraint` publication**, and publishes `/robot/operation/localization_success = false`.
- **Invariant Effect**:
  Continuing dead-reckoning extrapolation under tracking failure inevitably accumulates drift and risks high-speed collision. Halting TF triggers the downstream path follower's `pose_age` watchdog timeout, actively executing a safe hardware emergency stop.

### 6.3 3D NMS Top-K Global Recovery (`GlobalSearcher`)
When `lost_frame_count` exceeds `lost_count_thresh`, the node transitions to `is_lost_ = true` and triggers asynchronous global recovery:
1. **Coarse Field Evaluation**: Evaluates downsampled obstacle cloud across field bounds at coarse resolution.
2. **3D Non-Maximum Suppression**: Extracts `top_k_candidates` local minima separated by $\ge 0.60\text{ m}$ or $\ge 0.50\text{ rad}$.
3. **Full-Resolution Refinement**: Runs multi-level `matchObstacleCloud` around each peak, accepting the global minimum.
- **Invariant Effect**:
  Resolves initial pose uncertainty and deep tracking failure across symmetric field geometry without getting trapped in suboptimal local minima.

---

## 7. Invariant Parameter Mechanics & Boundary Conditions

| Parameter Name | Mechanism & Mathematical Role | Invariant Effect if Over-Dimensioned | Invariant Effect if Under-Dimensioned |
|---|---|---|---|
| `esdf.resolution` | Spatial discretization of 3D ESDF voxels [m] | Wall boundaries blur; pose resolution degrades | GPU memory scales as $\mathcal{O}(N^3)$, risking VRAM exhaustion |
| `esdf.max_dist` | Distance truncation ceiling of ESDF field [m] | Dynamic objects far from walls distort alignment gradient | Attraction basin narrows; coarse search risks skipping true basin |
| `scan_accumulation.window_s` | 3D accumulation window duration [s] | Acceleration induces non-linear distortion beyond deskew limits | Point density drops; ESDF mean cost variance increases |
| `filter_params.ground_z_thresh` | Height threshold for floor separation [m] | Low static boundaries are discarded, weakening pose constraints | Seams and floor irregularities enter obstacle cloud, inflating cost |
| `filter_params.robot_radius` | Geometric self-filter radius [m] | Legitimate nearby walls and obstacles are eliminated | Chassis self-reflections enter matcher, pulling pose toward robot origin |
| `velocity_estimation.enable` | Linear velocity extrapolation | (Disabled) Search center falls behind at speed, exiting search window | (Enabled) EMA smoothing preserves tracking while bounding jitter |
| `matching_params.dynamic_yaw_expansion` | Turn-rate proportional yaw window expansion | (Disabled) Search window starved during fast turns, causing tracking loss | (Enabled) Preserves search coverage during spins while minimizing idle GPU load |
| `matching_params.dynamic_yaw_omega_thresh` | Angular rate threshold for yaw expansion [rad/s] | Delayed expansion during angular acceleration, risking starvation | Minor sensor noise triggers expansion, wasting GPU compute cycles |
| `matching_params.max_dist_thresh` | Point residual clipping limit $d_{\max}$ [m] | Dynamic obstacle penalties become unbounded, rejecting good matches | Valid wall points with slight noise hit ceiling, degrading cost sensitivity |
| `matching_params.weight_2d_lidar` | Base 2D residual weight $w_{\text{2D,base}}$ | 2D extrinsic errors and robot pitch bias the full 6-DoF/3-DoF pose | High-frequency horizontal wall constraints are lost; drift increases |
| `matching_params.cost_threshold` | Acceptance threshold for 3D residual cost [m] | Degraded or misaligned poses accepted; spatial jumping risk increases | Sensor noise trips rejection, causing unnecessary TF halting |
| `matching_params.cost_threshold_2d` | Acceptance threshold for 2D residual cost [m] | Misaligned 2D planar locks accepted as valid matches | Guard 1 triggers frequently, causing unnecessary 3D-only retries |
| `matching_params.min_inlier_ratio` | Minimum inlier ratio for consensus acceptance | Overly strict in clutter; valid geometric matches rejected | Partial single-wall matches accepted despite unconstrained degrees of freedom |
| `global_search.lost_count_thresh` | Rejection frame count before global recovery | Robot remains lost for extended periods without initiating recovery | Transient occlusions trigger expensive global search prematurely |
