Below is a polished *README.md* formatted properly for GitHub, based on the information you provided.
No emojis are included.

---

# Orbit Estimation Dashboard — EKF vs RCO

## 1. Overview

This project presents an interactive *Streamlit-based dashboard* for comparative orbit estimation using two different reconstruction approaches:

* *Extended Kalman-like Filter (EKF)*
* *Robust Constrained Optimizer (RCO)* using IRLS with Huber Loss

The dashboard simulates orbital motion, injects sensor noise, outliers, and optional mid-orbit maneuvers, and evaluates how effectively each filtering method reconstructs the true trajectory. The interface provides interactive visualizations, diagnostic tools, and export options for detailed analysis.

---

## 2. Key Features

* Orbit simulation with configurable time steps and orbital dynamics
* Measurement noise, outlier injection, and optional maneuver modeling
* EKF and RCO estimation pipelines with adjustable tuning parameters
* Real-time interactive plots and diagnostics
* 3D orbit visualization
* Step-wise position error graphs
* Runtime comparison bar charts
* Huber weight diagnostics for RCO
* Result export options

  * RMSE summary CSV
  * Trajectory arrays (NPZ)
  * PNG figure downloads
* Built-in NPZ viewer to inspect saved results

---

## 3. Demo

To launch the dashboard:


streamlit run streamlit_orbit_gui.py


Make sure required dependencies are installed before running.

---

## 4. Sidebar Controls and Inputs

Users can freely configure:

| Category   | Parameters                                 |
| ---------- | ------------------------------------------ |
| Simulation | Time steps, Δt                             |
| Noise      | Measurement σ, outlier fraction, outlier σ |
| Maneuver   | Enable/disable, displacement magnitude     |
| RCO        | λ smoothing, Huber delta, max iterations   |
| EKF        | P0, Q, R scaling                           |
| Runtime    | Number of timing repetitions               |
| Misc       | Random seed                                |

---

## 5. Output Metrics

The application automatically computes and displays:

* RMSE comparisons of EKF vs RCO under:

  * Outlier conditions
  * Maneuver conditions
* Average runtime for both filters
* Interactive and downloadable visualizations

---

## 6. Recommended Folder Structure
<img width="221" height="110" alt="image" src="https://github.com/user-attachments/assets/f6110672-13a1-4d34-8f56-615911609c1e" />
---

## 7. Installation

Required packages:


pip install streamlit numpy matplotlib


(Optional built-in imports for development: csv, io, datetime, os)

---

## 8. How It Works (Conceptual Overview)

| Component       | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| Orbit generator | Produces synthetic near-circular orbit with slight Z-axis variation    |
| Noise model     | Adds Gaussian measurement noise and optional outliers                  |
| Maneuver model  | Injects abrupt mid-trajectory displacement                             |
| EKF             | Recursive Bayesian estimator for position tracking                     |
| RCO             | Iteratively Reweighted Least Squares + Huber Loss to suppress outliers |

---

## 9. Code Origin

All logic is implemented within:


streamlit_orbit_gui.py


This includes:

* Orbit simulation
* Measurement model
* EKF implementation
* RCO implementation (Huber + IRLS)
* Streamlit UI and visualization pipeline

---

## 10. Purpose

This dashboard provides a clear, practical environment to study the robustness of orbit estimation algorithms in the presence of:

* Gaussian noise
* Strong measurement outliers
* Sudden trajectory changes (maneuvers)

It is useful for:

* Aerospace and orbital tracking research
* Sensor fusion projects
* Robust estimation studies
* Optimization and filtering comparisons

---

## 11. License

This project may be used and modified for educational and research purposes.
If you use this work in publications or academic submissions, attribution is appreciated.

---
