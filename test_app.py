"""
Minimal Test Version - Deposit Rate Predictor
Use this to test if Streamlit Cloud deployment works
"""

import streamlit as st

# Test if plotly can be imported
try:
    import plotly.graph_objects as go
    plotly_status = "✅ Plotly imported successfully!"
    plotly_error = None
except Exception as e:
    plotly_status = "❌ Plotly import failed"
    plotly_error = str(e)

try:
    import pandas as pd
    pandas_status = "✅ Pandas imported successfully!"
except:
    pandas_status = "❌ Pandas import failed"

try:
    import numpy as np
    numpy_status = "✅ Numpy imported successfully!"
except:
    numpy_status = "❌ Numpy import failed"

# Display
st.title("🏦 Deposit Rate Predictor - Test Version")

st.header("Dependency Check")

st.write(plotly_status)
if plotly_error:
    st.error(plotly_error)

st.write(pandas_status)
st.write(numpy_status)

st.markdown("---")

if plotly_error is None:
    st.header("Quick Test")
    
    # Simple prediction demo
    fed_change = st.slider("Fed Rate Change (bps)", -100, 100, 25)
    
    if st.button("Test Prediction"):
        st.success(f"Test successful! Fed change: {fed_change} bps")
        
        # Simple chart
        fig = go.Figure(data=[
            go.Bar(x=['Credit Union', 'Bank'], 
                   y=[fed_change * 1.2, fed_change * 1.0],
                   marker_color=['green', 'blue'])
        ])
        fig.update_layout(title="Sample Prediction")
        st.plotly_chart(fig)
        
        st.info("If you see this chart, deployment is working! ✅")
else:
    st.error("Cannot proceed - plotly is not installed correctly")
    st.info("""
    **Troubleshooting:**
    1. Check that requirements.txt exists in root directory
    2. Verify it contains: streamlit, plotly, pandas, numpy
    3. Delete and redeploy the app
    """)

st.markdown("---")
st.caption("Test version - If this works, replace with full dashboard")