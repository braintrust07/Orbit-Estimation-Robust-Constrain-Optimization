Orbit Estimation Dashboard — EKF vs RCO

This project is an interactive Streamlit-based dashboard that compares two different orbit estimation approaches:

* Extended Kalman-like Filter (EKF)
* Robust Constrained Optimizer (RCO) using IRLS + Huber Loss

The app simulates orbital motion, injects sensor noise, outliers, and optional mid-orbit maneuvers, and evaluates how well each filtering method reconstructs the true trajectory.

Key Features

Orbit simulation with configurable time steps & dynamics

Measurement noise, outliers, and maneuver injection

EKF and RCO estimation pipelines with adjustable tuning parameters

Interactive visualizations

3D orbit plots

Step-wise position error plots

Runtime comparison bar chart

Huber weight diagnostics

Data export

Summary metrics CSV

Orbit arrays NPZ

PNG figure downloads

Built-in NPZ Viewer to inspect saved results

Demo

Run the dashboard with:

streamlit run streamlit_orbit_gui.py


Make sure the required libraries are installed (see below).

Sidebar Controls & Inputs

You can freely configure:

Category	Parameters
Simulation	Time steps, Δt
Noise	Measurement σ, outlier fraction, outlier σ
Maneuver	Enable/disable, displacement
RCO	λ smoothing, Huber delta, max iterations
EKF	P0, Q, R scaling
Runtime	Number of timing repetitions
Misc	Random seed
Output Metrics

The app automatically computes and displays:

RMSE of EKF vs RCO for outlier and maneuver cases

Average runtime for both filters

Interactive plots and downloadable visualizations

Folder Structure (recommended)
├─ streamlit_orbit_gui.py
├─ README.md
├─ /results (optional for saved visualizations)
└─ /samples (optional for NPZ examples)

Installation
pip install streamlit numpy matplotlib


(Optional for development: csv, io, datetime, os — these are built-ins)

How It Works (Conceptual)
Component	Description
Orbit generator	Creates a synthetic near-circular orbit with a small Z variation
Noise model	Adds Gaussian measurement noise; optional outliers
Maneuver model	Abrupt mid-trajectory displacement
EKF	Conventional recursive Bayesian position filter
RCO	Iteratively reweighted least squares + Huber to suppress outliers
Code Origin

The application is implemented fully in streamlit_orbit_gui.py and includes:

Orbit simulation

Measurement model

EKF implementation

RCO implementation (IRLS + Huber)

Streamlit UI and visualization logic

Code reference: streamlit_orbit_gui.py 

streamlit_orbit_gui

Purpose

This dashboard provides a clear, practical way to analyze and visualize the robustness of filtering methods in the presence of:

Gaussian noise

Severe outliers

Orbital maneuvers / sudden trajectory changes

It can be useful for:

Aerospace research

Sensor fusion & tracking projects

Robust estimation studies

ML/optimization comparisons

License

Feel free to use and modify this project for research or educational purposes.
If you share or publish work based on it, giving credit would be appreciated ❤️.

Contributions

Suggestions and improvements are welcome!
Open a pull request or create an issue if you'd like to collaborate.
