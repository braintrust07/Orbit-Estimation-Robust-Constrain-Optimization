import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import time
import io
import csv
from datetime import datetime
import os

# =============================
# Core functions (adapted from supplied script)
# =============================

def generate_truth_orbit(n_steps=200, dt=10.0):
    t = np.arange(0, n_steps * dt, dt)
    theta = 0.001 * t
    r = 7000e3
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 1000 * np.sin(0.1 * theta)
    return np.stack((x, y, z), axis=1)


def generate_measurements(truth,
                          add_outliers=True,
                          add_maneuver=False,
                          outlier_fraction=0.10,
                          outlier_sigma=2000.0,
                          meas_sigma=100.0,
                          maneuver_delta_pos=2000.0,
                          rng_seed=None):
    if rng_seed is not None:
        np.random.seed(int(rng_seed))
    n = len(truth)

    meas_out = truth + np.random.normal(0, meas_sigma, truth.shape)
    if add_outliers:
        outlier_idx = np.random.choice(n, size=int(max(1, outlier_fraction * n)), replace=False)
        meas_out[outlier_idx] += np.random.normal(0, outlier_sigma, (len(outlier_idx), 3))

    truth_man = truth.copy()
    if add_maneuver:
        man_start = n // 2
        ramp = np.linspace(0, maneuver_delta_pos, n - man_start)
        radii = np.linalg.norm(truth_man[man_start:, :2], axis=1, keepdims=True)
        radii[radii == 0] = 1.0
        radial_unit = truth_man[man_start:, :2] / radii
        displacement_xy = (radial_unit * ramp[:, None])
        truth_man[man_start:, 0] += displacement_xy[:, 0]
        truth_man[man_start:, 1] += displacement_xy[:, 1]

    meas_man = truth_man + np.random.normal(0, meas_sigma, truth_man.shape)

    return meas_out, meas_man, truth_man


def RMSE(truth, est):
    return np.sqrt(np.mean(np.sum((truth - est) ** 2, axis=1)))


# -----------------------------
# EKF (simple position filter) - identical logic to original
# -----------------------------

def run_EKF(meas, P0_scale=100.0, Q_scale=10.0, R_scale=200.0):
    est = np.zeros_like(meas)
    x = meas[0].copy()
    P = np.eye(3) * P0_scale
    Q = np.eye(3) * Q_scale
    R = np.eye(3) * R_scale
    for i in range(len(meas)):
        x_pred = x
        P_pred = P + Q
        z = meas[i]
        K = P_pred @ np.linalg.inv(P_pred + R)
        x = x_pred + K @ (z - x_pred)
        P = (np.eye(3) - K) @ P_pred
        est[i] = x
    return est


# -----------------------------
# RCO (IRLS smoothing with Huber)
# -----------------------------

def run_RCO(meas,
            lambda_smooth=10.0,
            huber_delta=1e3,
            max_iters=50,
            tol=1e-6,
            return_weights=False):
    meas = np.asarray(meas, dtype=float)
    n, d = meas.shape

    # Build 2nd-difference matrix
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    L = D.T @ D
    I_n = np.eye(n)

    x = meas.copy()
    prev_obj = np.nan
    final_w = np.ones(n)

    for it in range(max_iters):
        residual = x - meas
        res_norm = np.linalg.norm(residual, axis=1)

        # Huber weights
        w = np.ones(n)
        large = res_norm > huber_delta
        if np.any(large):
            w[large] = huber_delta / (res_norm[large] + 1e-9)
        w = np.clip(w, 1e-4, 1.0)
        final_w = w.copy()
        W = np.diag(w)

        # Regularized least squares
        A = W + lambda_smooth * L + 1e-9 * I_n
        x_new = np.zeros_like(x)
        for j in range(d):
            b = W @ meas[:, j]
            x_new[:, j] = np.linalg.solve(A, b)

        # Objective for convergence check
        r = np.linalg.norm(x_new - meas, axis=1)
        huber = np.where(r <= huber_delta, 0.5 * r ** 2, huber_delta * (r - 0.5 * huber_delta))
        obj = np.sum(huber) + 0.5 * lambda_smooth * np.sum((D @ x_new) ** 2)

        if np.isfinite(prev_obj):
            relchg = abs(prev_obj - obj) / (1.0 + abs(prev_obj))
            if relchg < tol:
                x = x_new
                break
        prev_obj = obj
        x = x_new

    if return_weights:
        return x, final_w
    return x


# =============================
# Utility: timing wrapper
# =============================

def timed_run(func, *args, repeats=1):
    times = []
    result = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    return result, float(np.mean(times))


# =============================
# Plot helpers: matplotlib figures returned for use in Streamlit
# =============================

def fig_orbit(truth, ekf, rco, title="Orbit Comparison", zoom=False):
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color='k', linestyle='-', label='Truth')
    ax.plot(ekf[:, 0], ekf[:, 1], ekf[:, 2], color='r', linestyle='--', label='EKF')
    ax.plot(rco[:, 0], rco[:, 1], rco[:, 2], color='b', linestyle='-', label='RCO')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    if zoom:
        res = np.linalg.norm(truth - ekf, axis=1)
        idx = np.argmax(res)
        start = max(0, idx - 25)
        end = min(len(truth), idx + 25)
        ax.set_xlim(truth[start:end, 0].min(), truth[start:end, 0].max())
        ax.set_ylim(truth[start:end, 1].min(), truth[start:end, 1].max())
        ax.set_zlim(truth[start:end, 2].min(), truth[start:end, 2].max())
    fig.tight_layout()
    return fig


def fig_error(truth, ekf, rco, title="Position Error"):
    err_ekf = np.sqrt(np.sum((truth - ekf) ** 2, axis=1))
    err_rco = np.sqrt(np.sum((truth - rco) ** 2, axis=1))
    fig = Figure(figsize=(8, 3.5))
    ax = fig.subplots()
    ax.plot(err_ekf, linestyle='--', label='EKF Error')
    ax.plot(err_rco, linestyle='-', label='RCO Error')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position Error (m)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_weights(weights, title="RCO Weights (Huber)"):
    fig = Figure(figsize=(8, 2.5))
    ax = fig.subplots()
    ax.plot(weights, '-o', markersize=3)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Weight (1=trusted, 0=outlier)')
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def fig_runtime_bar(times_dict):
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    labels = list(times_dict.keys())
    vals = [times_dict[k] for k in labels]
    bars = ax.bar(labels, vals)
    ax.set_ylabel('Avg Execution Time (s)')
    ax.set_title('Runtime Comparison')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1e-6, f"{v:.4f}s", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    return fig


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Orbit Estimation — EKF vs RCO", layout="wide")

st.title("Orbit Estimation Dashboard — EKF vs RCO")
st.markdown(
    """
    This interactive app lets you compare a simple Extended Kalman-like filter (EKF)
    against a Robust Constrained Optimizer (RCO) implemented via IRLS + Huber loss.

    Customize measurement noise, outliers, maneuvers and smoothing parameters then run both filters
    to inspect trajectories, per-step errors, RCO weights and runtime comparison.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Scenario settings")
    n_steps = st.slider("Number of time steps", min_value=50, max_value=2000, value=200, step=10)
    dt = st.number_input("Time step dt (s)", value=10.0, format="%.3f")

    st.subheader("Measurement noise & outliers")
    meas_sigma = st.number_input("Measurement sigma (m)", min_value=1.0, value=100.0, step=1.0)
    add_outliers = st.checkbox("Add outliers", value=True)
    outlier_fraction = st.slider("Outlier fraction", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
    outlier_sigma = st.number_input("Outlier sigma (m)", min_value=10.0, value=2000.0, step=10.0)

    st.subheader("Maneuver")
    add_maneuver = st.checkbox("Add maneuver (mid trajectory)", value=False)
    maneuver_delta = st.number_input("Maneuver total radial displacement (m)", min_value=0.0, value=2000.0, step=100.0)

    st.subheader("RCO parameters")
    lambda_smooth = st.number_input("Smoothing lambda", min_value=0.0, value=10.0, step=0.1, format="%.4f")
    huber_delta = st.number_input("Huber delta (m)", min_value=0.1, value=1e3, step=10.0, format="%.4f")
    max_iters = st.number_input("RCO max IRLS iterations", min_value=1, max_value=1000, value=50)

    st.subheader("EKF tuning")
    P0_scale = st.number_input("EKF P0 scale", min_value=1.0, value=100.0)
    Q_scale = st.number_input("EKF Q scale", min_value=0.0, value=10.0)
    R_scale = st.number_input("EKF R scale", min_value=0.0, value=200.0)

    st.subheader("Misc")
    rng_seed = st.text_input("Random seed (leave blank for random)")
    repeats = st.slider("Timing repeats (average)", min_value=1, max_value=10, value=3)

    run_btn = st.button("Run Filters", type='primary')

# Container to display outputs
out_col1, out_col2 = st.columns([1.2, 1])

if run_btn:
    with st.spinner("Generating data and running filters..."):
        truth = generate_truth_orbit(n_steps=n_steps, dt=dt)
        meas_out, meas_man, truth_man = generate_measurements(
            truth,
            add_outliers=add_outliers,
            add_maneuver=add_maneuver,
            outlier_fraction=outlier_fraction,
            outlier_sigma=outlier_sigma,
            meas_sigma=meas_sigma,
            maneuver_delta_pos=maneuver_delta,
            rng_seed=(None if rng_seed.strip() == '' else rng_seed),
        )

        # Decide which truth to compare to for maneuver scenario
        compare_truth_out = truth
        compare_truth_man = truth_man if add_maneuver else truth

        # Run EKF & RCO for outlier scenario (measurements without maneuver)
        ekf_out, t_ekf_out = timed_run(run_EKF, meas_out, P0_scale, Q_scale, R_scale, repeats=repeats)
        rco_out, t_rco_out = timed_run(
            run_RCO, meas_out, lambda_smooth, huber_delta, int(max_iters), 1e-6, False, repeats=repeats
        )

        # Run EKF & RCO for maneuver scenario (measurements from truth_man)
        ekf_man, t_ekf_man = timed_run(run_EKF, meas_man, P0_scale, Q_scale, R_scale, repeats=repeats)
        rco_man, t_rco_man = timed_run(
            run_RCO, meas_man, lambda_smooth, huber_delta, int(max_iters), 1e-6, False, repeats=repeats
        )

        # Compute RMSE
        rmse_ekf_out = RMSE(compare_truth_out, ekf_out)
        rmse_rco_out = RMSE(compare_truth_out, rco_out)
        rmse_ekf_man = RMSE(compare_truth_man, ekf_man)
        rmse_rco_man = RMSE(compare_truth_man, rco_man)

    # Display summary metrics
    with out_col1:
        st.subheader("Summary Metrics")
        st.metric(label="Outliers: EKF RMSE (m)", value=f"{rmse_ekf_out:.2f}")
        st.metric(label="Outliers: RCO RMSE (m)", value=f"{rmse_rco_out:.2f}")
        st.metric(label="Maneuver: EKF RMSE (m)", value=f"{rmse_ekf_man:.2f}")
        st.metric(label="Maneuver: RCO RMSE (m)", value=f"{rmse_rco_man:.2f}")

        st.markdown("**Average runtimes (s)**")
        st.write(f"EKF (Outliers): {t_ekf_out:.5f}s — RCO (Outliers): {t_rco_out:.5f}s")
        st.write(f"EKF (Maneuver): {t_ekf_man:.5f}s — RCO (Maneuver): {t_rco_man:.5f}s")

        # Offer CSV download of summary
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["Scenario", "Filter", "RMSE (m)", "Avg Time (s)"])
        writer.writerow(["Outliers", "EKF", f"{rmse_ekf_out:.6f}", f"{t_ekf_out:.6f}"])
        writer.writerow(["Outliers", "RCO", f"{rmse_rco_out:.6f}", f"{t_rco_out:.6f}"])
        writer.writerow(["Maneuver", "EKF", f"{rmse_ekf_man:.6f}", f"{t_ekf_man:.6f}"])
        writer.writerow(["Maneuver", "RCO", f"{rmse_rco_man:.6f}", f"{t_rco_man:.6f}"])
        csv_bytes = csv_buf.getvalue().encode('utf-8')
        st.download_button(label="Download results CSV", data=csv_bytes, file_name=f"results_summary_{now}.csv")

    # Plots column
    with out_col2:
        st.subheader("Plots")
        tab1, tab4, tab2, tab3 = st.tabs(["Orbit (Outliers)","Orbit(Manuevers)", "Errors (Outliers)", "Runtime & Weights"]) 

        with tab1:
            st.write("**Full trajectory — Outliers scenario**")
            fig1 = fig_orbit(compare_truth_out, ekf_out, rco_out, title="Orbit Estimation (Outlier Scenario)")
            st.pyplot(fig1)
            st.write("**Zoom around largest EKF residual**")
            fig1z = fig_orbit(compare_truth_out, ekf_out, rco_out, title="Zoomed Orbit (Outliers)", zoom=True)
            st.pyplot(fig1z)
        with tab4:
            # Maneuver orbit plot
            st.write("**Full trajectory — Maneuver scenario**")
            fig1m = fig_orbit(compare_truth_man, ekf_man, rco_man,
            title="Orbit Estimation (Maneuver Scenario)")
            st.pyplot(fig1m)
            st.write("**Zoom around largest EKF residual (Maneuver)**")
            fig1mz = fig_orbit(compare_truth_man, ekf_man, rco_man,
            title="Zoomed Orbit (Maneuver)", zoom=True)
            st.pyplot(fig1mz)


        with tab2:
            st.write("**Per-step position error (Outliers)**")
            fig_err = fig_error(compare_truth_out, ekf_out, rco_out, title="Position Error (Outlier Scenario)")
            st.pyplot(fig_err)

            st.write("**Maneuver scenario — per-step error**")
            fig_err_man = fig_error(compare_truth_man, ekf_man, rco_man, title="Position Error (Maneuver)")
            st.pyplot(fig_err_man)

        with tab3:
            st.write("**Runtime comparison**")
            times = {
                'EKF-Outliers': t_ekf_out,
                'RCO-Outliers': t_rco_out,
                'EKF-Maneuver': t_ekf_man,
                'RCO-Maneuver': t_rco_man,
            }
            fig_rt = fig_runtime_bar(times)
            st.pyplot(fig_rt)

            st.write("**RCO Huber weights (Outliers)**")
            _, weights_out = run_RCO(meas_out, lambda_smooth, huber_delta, int(max_iters), 1e-6, return_weights=True)
            fig_w = fig_weights(weights_out, title="RCO Weights (Outliers)")
            st.pyplot(fig_w)

            if add_maneuver:
                st.write("**RCO Huber weights (Maneuver)**")
                _, weights_man = run_RCO(meas_man, lambda_smooth, huber_delta, int(max_iters), 1e-6, return_weights=True)
                st.pyplot(fig_weights(weights_man, title="RCO Weights (Maneuver)"))

    # Full-width diagnostics and download of plotted images
    st.markdown("---")
    st.header("Diagnostics & Downloads")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Save figure images")
        buf = io.BytesIO()
        fig1.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        st.download_button("Download Orbit (Outliers) PNG", data=buf, file_name=f"orbit_outliers_{now}.png")

    with col_b:
        st.caption("Raw measurement / estimate arrays")
        # Create NPZ for download
        npz_buf = io.BytesIO()
        np.savez_compressed(npz_buf,
                            truth=compare_truth_out,
                            meas_out=meas_out,
                            ekf_out=ekf_out,
                            rco_out=rco_out,
                            truth_man=compare_truth_man,
                            meas_man=meas_man,
                            ekf_man=ekf_man,
                            rco_man=rco_man)
        npz_buf.seek(0)
        st.download_button("Download NPZ (arrays)", data=npz_buf, file_name=f"orbit_arrays_{now}.npz")

    st.success("Run complete — results displayed above.")

else:
    st.info("Adjust parameters in the sidebar and press **Run Filters** to begin.")

# =============================
# NPZ VIEWER (Added at bottom)
# =============================

st.markdown("---")
st.header("📁 NPZ Viewer (Load & Inspect Saved Arrays)")

uploaded_npz = st.file_uploader("Upload an NPZ file", type=["npz"])

if uploaded_npz is not None:
    try:
        npz_data = np.load(uploaded_npz)

        st.subheader("Available Arrays")
        st.write(list(npz_data.files))

        selected_array = st.selectbox(
            "Select array to preview",
            npz_data.files,
        )

        if selected_array:
            arr = npz_data[selected_array]

            st.subheader(f"Array: {selected_array}")
            st.write(f"Shape: {arr.shape}")
            st.write(f"Dtype: {arr.dtype}")

            # Show first few rows
            st.write("Preview:")
            st.dataframe(arr[:50])

            # Plot if 2D with 3 columns (x,y,z)
            if arr.ndim == 2 and arr.shape[1] == 3:
                st.write("3D Plot Preview:")

                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2])
                ax.set_title(selected_array)
                st.pyplot(fig)

            # Download CSV version
            csv_buf = io.StringIO()
            np.savetxt(csv_buf, arr, delimiter=",")
            st.download_button(
                label="Download this array as CSV",
                data=csv_buf.getvalue(),
                file_name=f"{selected_array}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Failed to load NPZ file: {e}")

st.caption("The above results validate the performance of the filtering algorithms through a detailed comparison of truth, measurements, EKF outputs, and RCO estimates using both maneuvering and outlier scenarios. The included NPZ-based diagnostics ensure transparency, reproducibility, and complete inspection of the raw estimation data.")
