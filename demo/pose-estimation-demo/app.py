"""Streamlit entry point for the pose estimation demo."""
import streamlit as st

st.set_page_config(page_title="Pose Estimation Demo", layout="wide")
st.title("Vision-Based Precision Pose Estimation")
st.caption("Deterministic research demo dashboard")
st.info("Use the experiment API or scripts to generate telemetry. This lightweight dashboard is safe to run without model checkpoints.")
