import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Histogram Fitter", layout="wide")
st.title("Histogram Distribution Fitter")

def get_param_names(dist):
    shapes = dist.shapes
    params = []
    if shapes:
        params += [s.strip() for s in shapes.split(",")]
    params += ["loc", "scale"]
    return params

st.header("1. Data Input")
tab_manual, tab_csv = st.tabs(["Manual Entry", "CSV Upload"])

data = None

with tab_manual:
    manual_text = st.text_area(
        "Enter numeric values:",
        "1.2, 2.1, 2.5, 3.2, 3.3"
    )
    try:
        cleaned = manual_text.replace(",", " ").replace("\n", " ")
        data = np.array([float(x) for x in cleaned.split() if x])
    except:
        st.error("Invalid number format.")
        data = None

with tab_csv:
    uploaded = st.file_uploader("Upload CSV (numeric column)")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found")
            col = numeric_cols[0]
            data = df[col].dropna().values
            st.success(f"Loaded {len(data)} points from: {col}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            data = None

if data is None or len(data) == 0:
    st.warning("Please enter or upload valid data to continue.")
    st.stop()

DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta,
    "Lognormal": stats.lognorm,
    "Chi-square": stats.chi2,
    "Student t": stats.t,
    "Uniform": stats.uniform,
    "Rayleigh": stats.rayleigh,
}

st.header("2. Fit a Distribution")
dist_name = st.selectbox("Select distribution:", list(DISTRIBUTIONS.keys()))
dist = DISTRIBUTIONS[dist_name]

params = dist.fit(data)
param_names = get_param_names(dist)

tab_auto, tab_manual_fit = st.tabs(["Automatic Fit", "Manual Fit"])

with tab_auto:
    st.subheader("Fitted Parameters")
    cols = st.columns(len(params))
    for c, name, val in zip(cols, param_names, params):
        c.metric(name, f"{val:.4f}")

    x = np.linspace(min(data), max(data), 500)
    shape_params = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf_vals = dist.pdf(x, *shape_params, loc=loc, scale=scale)

    st.subheader("Histogram with Fitted PDF")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=25, density=True, alpha=0.4, color="#2ecc71", edgecolor="black")
    ax.plot(x, pdf_vals, color="#1e8449", linewidth=2)
    ax.set_title("Fitted PDF", fontsize=14)
    st.pyplot(fig)

    hist_vals, bin_edges = np.histogram(data, bins=25, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fitted_vals = dist.pdf(centers, *shape_params, loc=loc, scale=scale)

    mse = np.mean((hist_vals - fitted_vals) ** 2)
    max_err = np.max(np.abs(hist_vals - fitted_vals))

    st.subheader("Fit Quality")
    st.write(f"MSE: {mse:.6f}")
    st.write(f"Max Error: {max_err:.6f}")

with tab_manual_fit:
    st.subheader("Manual Parameter Adjustment")

    shape_sliders = []
    shapes = dist.shapes.split(",") if dist.shapes else []

    cols = st.columns(len(shapes)) if shapes else []
    for col, name in zip(cols, shapes):
        val = col.slider(name.strip(), 0.01, 10.0, 1.0)
        shape_sliders.append(val)

    loc_val = st.slider("loc", -10.0, 10.0, 0.0)
    scale_val = st.slider("scale", 0.1, 10.0, 1.0)

    x = np.linspace(min(data), max(data), 500)

    try:
        pdf_manual = dist.pdf(x, *shape_sliders, loc=loc_val, scale=scale_val)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(data, bins=25, density=True, alpha=0.4, color="#2ecc71", edgecolor="black")
        ax2.plot(x, pdf_manual, color="#1e8449", linewidth=2)
        ax2.set_title("Manual PDF", fontsize=14)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error computing PDF: {e}")






            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            