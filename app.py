import os
import glob
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & DARK PACS THEME
# ==========================================
st.set_page_config(
    page_title="KneeMRI & 3D PTS Analysis by Rianna-Maria Tanase",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #00f2fe; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8b949e; }
    .stSelectbox, .stSlider, .stFileUploader { background-color: #161b22; border-radius: 8px; }
    .risk-banner-normal { background-color: #0d381e; border: 1px solid #2ea043; color: #56d364; padding: 12px; border-radius: 6px; }
    .risk-banner-warning { background-color: #3b2300; border: 1px solid #d29922; color: #e3b341; padding: 12px; border-radius: 6px; }
    .risk-banner-danger { background-color: #4c1d1d; border: 1px solid #f85149; color: #f85149; padding: 12px; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KNEEMRI DATA LOADERS
# ==========================================
@st.cache_data
def load_metadata():
    if os.path.exists("metadata.csv"):
        return pd.read_csv("metadata.csv")
    return None

@st.cache_data
def load_mri_volume(file_path):
    """Loads and normalizes 12-bit pickled MRI volume arrays."""
    try:
        with open(file_path, 'rb') as f:
            volume = pickle.load(f)
        volume = np.array(volume, dtype=np.float32)
        # Normalize to 8-bit range for visualization
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5) * 255.0
        return volume.astype(np.uint8)
    except Exception as e:
        st.error(f"Error loading volume file `{file_path}`: {e}")
        return None

def get_available_mri_files():
    """Dynamically finds all .pck files in sample_data/, volumetric_data/, or root directory."""
    files = {}
    search_paths = ["sample_data/*.pck", "volumetric_data/*.pck", "*.pck"]
    
    for path_pattern in search_paths:
        for file in glob.glob(path_pattern):
            filename = os.path.basename(file)
            rec_id = os.path.splitext(filename)[0]
            if rec_id not in files:
                files[rec_id] = file
                
    return files

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🦴 KneeMRI PACS Workstation")
st.sidebar.markdown("---")

df_meta = load_metadata()
available_files = get_available_mri_files()

st.sidebar.subheader("📁 Record Selection")

if available_files:
    # Sort IDs logically
    record_ids = sorted(list(available_files.keys()))
    selected_id = st.sidebar.selectbox("Select Patient Record ID:", record_ids)
    vol_path = available_files[selected_id]
else:
    st.sidebar.warning("No `.pck` files detected. Ensure `example.pck` or `sample_data/` folder exists.")
    vol_path = "example.pck"

# Display patient diagnosis metadata if available
if df_meta is not None and available_files:
    vol_col = df_meta.columns[0]
    matched_meta = df_meta[df_meta[vol_col].astype(str) == str(selected_id)]
    if not matched_meta.empty:
        st.sidebar.markdown("**Record Metadata:**")
        st.sidebar.dataframe(matched_meta.T, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Surgical Parameters")

preset_case = st.sidebar.selectbox(
    "Clinical Presets:",
    ("Custom Adjustment", "Normal Variant (6.5°)", "Borderline Risk (11.0°)", "Severe High Risk (15.5°)")
)

if preset_case == "Normal Variant (6.5°)":
    pts_angle = 6.5
elif preset_case == "Borderline Risk (11.0°)":
    pts_angle = 11.0
elif preset_case == "Severe High Risk (15.5°)":
    pts_angle = 15.5
else:
    pts_angle = st.sidebar.slider("Posterior Tibial Slope (°):", 0.0, 20.0, 9.5, 0.5)

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("KneeMRI & 3D PTS Diagnostic Engine")
st.caption("Proton density weighted MRI volume inspection & Posterior Tibial Slope risk estimation. Made by Rianna-Maria Tanase")

# Metrics Header
col1, col2, col3, col4 = st.columns(4)

if pts_angle < 10.0:
    risk_tier = "LOW"
    shear_force = "Normal (~120 N)"
elif 10.0 <= pts_angle <= 12.0:
    risk_tier = "MODERATE"
    shear_force = "Elevated (~210 N)"
else:
    risk_tier = "HIGH RISK"
    shear_force = "Critical (>340 N)"

col1.metric("Calculated PTS Angle", f"{pts_angle:.1f}°")
col2.metric("ACL Strain Risk Tier", risk_tier)
col3.metric("Est. Anterior Shear Force", shear_force)
col4.metric("Dataset Source", "CHC Rijeka (1.5T)")

st.markdown("---")

# Main Content Layout
main_col, side_col = st.columns([3, 1])

with main_col:
    if os.path.exists(vol_path):
        volume = load_mri_volume(vol_path)
        if volume is not None:
            num_slices = volume.shape[0] if volume.ndim == 3 else 1
            slice_idx = st.slider("Navigate Sagittal MRI Slices:", 0, num_slices - 1, num_slices // 2)

            img_slice = volume[slice_idx] if volume.ndim == 3 else volume

            # Plotly Interactive MRI Heatmap Display
            fig_mri = go.Figure(data=go.Heatmap(
                z=img_slice,
                colorscale='gray',
                showscale=False
            ))

            # Overlay Slope Measurement Line
            h, w = img_slice.shape
            rad = np.radians(pts_angle)
            center_x, center_y = w // 2, h // 2
            dx = w * 0.3
            dy = dx * np.tan(rad)

            fig_mri.add_trace(go.Scatter(
                x=[center_x - dx, center_x + dx],
                y=[center_y - dy, center_y + dy],
                mode='lines',
                line=dict(color='#f85149', width=3, dash='dash'),
                name=f'Slope ({pts_angle}°)'
            ))

            fig_mri.update_layout(
                title=f"Sagittal View - Slice {slice_idx + 1}/{num_slices} ({os.path.basename(vol_path)})",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, autorange='reversed'),
                margin=dict(l=0, r=0, b=0, t=40),
                height=580
            )

            st.plotly_chart(fig_mri, use_container_width=True)
    else:
        st.error(f"MRI data file `{vol_path}` not found. Place `.pck` files in root or `sample_data/` directory.")

with side_col:
    st.subheader("📋 Risk Diagnostic Engine")

    if pts_angle < 10.0:
        st.markdown(
            f"""<div class='risk-banner-normal'>
            <h4>🟢 Low Risk ({pts_angle:.1f}°)</h4>
            <p>Anatomical slope within normal limits (&lt;10°). Minimal secondary anterior shearing strain on ACL graft during weight-bearing.</p>
            </div>""", unsafe_allow_html=True
        )
    elif 10.0 <= pts_angle <= 12.0:
        st.markdown(
            f"""<div class='risk-banner-warning'>
            <h4>🟡 Moderate Risk ({pts_angle:.1f}°)</h4>
            <p>Borderline slope elevation (10°–12°). Increased tibial displacement forces during knee flexion. Monitor graft tension.</p>
            </div>""", unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class='risk-banner-danger'>
            <h4>🔴 High Risk ({pts_angle:.1f}°)</h4>
            <p>High slope detected (&gt;12°). Significantly elevated risk for primary/secondary ACL graft revision. Consider slope-correcting anterior osteotomy.</p>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Biomechanical Estimates:**")
    st.write(f"- **PTS Angle:** `{pts_angle}°`")
    st.write(f"- **Tibial Drift Ratio:** `{np.tan(np.radians(pts_angle)):.3f}`")
    st.write(f"- **ACL Load Increase:** `{max(0, (pts_angle - 8.0) * 12.5):.1f}%`")

    st.markdown("---")
    st.download_button(
        label="📄 Export Clinical Report",
        data=f"PTS Angle: {pts_angle} deg\nRisk Tier: {risk_tier}\nShear Force: {shear_force}",
        file_name="pts_mri_report.txt"
    )
