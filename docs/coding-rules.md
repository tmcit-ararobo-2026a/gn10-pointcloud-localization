# Coding Rules

[English](coding-rules.md) | [日本語 (Japanese)](coding-rules_jp.md)

These coding standards apply to all source code, configuration files, and documentation in this repository.

## Table of Contents

1. [Naming Conventions](#1-naming-conventions)
2. [Implementation Constraints](#2-implementation-constraints)
3. [Comment & Documentation Standards](#3-comment--documentation-standards)

## 1. Naming Conventions

### 1.1 General Principles

- Ensure every identifier accurately reflects its behavior and role.
- When loop iterators span nested or non-trivial loops, prefer descriptive names such as `point_index`, `voxel_index`, or `sample_index` over single-letter names.
- Include physical units in variable and parameter names where applicable (e.g., `distance_m`, `window_sec`, `stamp_ns`).
- Choose unambiguous names and avoid words with conflicting interpretations.
- Repository names must use lowercase letters and hyphens (`kebab-case`).
- File names must use lowercase letters and underscores (`snake_case`).

### 1.2 Identifier Styles

| Target | Convention | Example |
| :--- | :--- | :--- |
| Classes / Structs | `PascalCase` | `PoseSolver`, `ESDFMap`, `YawHistory` |
| Functions / Variables | `snake_case` / `camelCase` (match module convention) | `get_angle_at()`, `target_velocity` |
| Constants / Enumerators | `ALL_CAPS` | `CYLINDER`, `VISUAL_BOX` |
| Private Member Variables | Trailing underscore `_` | `last_known_pose_`, `is_lost_` |

### 1.3 Abbreviations

- Ad-hoc abbreviations (`pos`, `cnt`, `mgr`) are discouraged in new interfaces.
- Standard domain acronyms are permitted: `ESDF`, `SDF`, `IMU`, `TF`, `PCD`, `NMS`, `CUDA`, `GPU`, `FOV`, `EMA`, `PID`.
- Keep acronyms capitalized in class and type names (e.g., `ESDFMap`, `IMUSample`).

## 2. Implementation Constraints

### 2.1 Common C++ Rules

- Follow the Google C++ Style Guide as the baseline standard.
- Avoid `do-while` and `goto` statements.
- Use `constexpr` for compile-time constants.

### 2.2 Embedded Systems (STM32 / ESP32)

- Dynamic heap allocation (`new`, `malloc`, `std::vector`, `std::string`) is prohibited; use `std::array` or fixed-size buffers.
- Keep interrupt/hardware callbacks minimal.
- Do not use `try-catch` exceptions on microcontrollers.

### 2.3 ROS 2 / Linux Host & Jetson GPU

- Use RAII and smart pointers (`std::unique_ptr`, `std::shared_ptr`) for host resource management, and manage CUDA device/pinned buffers deterministically in constructors and destructors.
- Prevent global namespace pollution.
- **Logging:** Use ROS 2 loggers (`RCLCPP_INFO`, `RCLCPP_WARN`, `RCLCPP_ERROR`) inside ROS 2 nodes rather than `std::cout`.

## 3. Comment & Documentation Standards

- **All source code, header, CUDA kernel, launch, test, and YAML config comments in `gn10-pointcloud-localization` MUST be written in English.**
- Provide Doxygen-style comments (`@brief`, `@param`, `@return`) on public functions and kernels so the localization algorithm and parameter effects are clear directly from the code.
- Separate logical stages of multi-step functions with concise single-line summary comments.
- **Bilingual Documentation:** Important documents (`README.md` / `README_jp.md`, `docs/parameter_tuning_guide.md` / `docs/parameter_tuning_guide_jp.md`, `docs/coding-rules.md` / `docs/coding-rules_jp.md`, and PlantUML `.pu` diagrams) must be maintained in both English and Japanese in separate files and kept synchronized whenever algorithms or parameters change.
