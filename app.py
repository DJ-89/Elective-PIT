import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Earthquake Risk Prediction System",
    page_icon=" earthquak",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #FF6B35;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-low {
        color: #4ECDC4;
        font-weight: bold;
        font-size: 1.2em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"> earthquak Earthquake Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown("---")

# Load the trained model and other artifacts
@st.cache_resource
def load_model():
    model = joblib.load('risk_area_identifier.pkl')
    scaler = joblib.load('scaler_risk_identifier.pkl')
    dbscan = joblib.load('dbscan_zone_identifier.pkl')
    threshold = joblib.load('threshold_risk_identifier.pkl')
    feature_cols = joblib.load('feature_cols_risk_identifier.pkl')
    zone_risk_lookup = joblib.load('zone_risk_lookup.pkl')
    return model, scaler, dbscan, threshold, feature_cols, zone_risk_lookup

try:
    model, scaler, dbscan, threshold, feature_cols, zone_risk_lookup = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.header(" earthquak Input Parameters")

# Input fields for earthquake data
col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input("Latitude (°)", value=12.8797, format="%.4f", 
                            help="Latitude in degrees (Philippines range: 4° to 21°)")
longitude = col2.number_input("Longitude (°)", value=121.7740, format="%.4f", 
                             help="Longitude in degrees (Philippines range: 116° to 127°)")
depth = st.sidebar.slider("Depth (km)", min_value=0.0, max_value=100.0, value=10.0, 
                         help="Depth of the earthquake in kilometers")

# Calculate additional features
latitude_abs = abs(latitude)
longitude_abs = abs(longitude)
distance_from_center = np.sqrt(latitude**2 + longitude**2)
depth_log = np.log1p(depth)
depth_normalized = depth / 100.0  # Assuming max depth is around 100km
lat_long_interact = latitude * longitude
lat_depth = latitude * depth
long_depth = longitude * depth

# Predict seismic zone using DBSCAN (simplified approach)
# Since we can't predict the exact zone without the full dataset, 
# we'll use the zone risk lookup or assign a default zone
try:
    # For simplicity, we'll use the mean values from the training data to assign a zone risk
    # In a real scenario, we would need to run the full DBSCAN clustering
    # Here we'll use the mean zone risk from the lookup table
    if not zone_risk_lookup.empty:
        zone_risk_score = zone_risk_lookup['zone_risk_score'].mean()
    else:
        zone_risk_score = 0.5  # Default value
    seismic_zone = -1  # Default zone for new locations
except:
    zone_risk_score = 0.5
    seismic_zone = -1

# Prepare input data for prediction
input_data = np.array([[latitude, longitude, depth, seismic_zone, zone_risk_score,
                       latitude_abs, longitude_abs, distance_from_center,
                       depth_log, depth_normalized, lat_long_interact, 
                       lat_depth, long_depth]])

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make prediction
prediction_proba = model.predict_proba(input_scaled)[0, 1]
prediction_binary = 1 if prediction_proba >= threshold else 0

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="sub-header">📍 Location Details</div>', unsafe_allow_html=True)
    
    # Create a map visualization
    df_map = pd.DataFrame({
        'lat': [latitude],
        'lon': [longitude],
        'depth': [depth],
        'risk_level': ['High Risk' if prediction_binary == 1 else 'Low Risk']
    })
    
    fig_map = px.scatter_mapbox(
        df_map, 
        lat='lat', 
        lon='lon', 
        color='risk_level',
        color_discrete_map={'High Risk': '#FF6B35', 'Low Risk': '#4ECDC4'},
        size='depth',
        size_max=15,
        zoom=6,
        height=400,
        title='Earthquake Risk Location'
    )
    
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.markdown('<div class="sub-header">📊 Risk Assessment</div>', unsafe_allow_html=True)
    
    # Display risk metrics
    st.metric("Risk Probability", f"{prediction_proba:.3f}")
    st.metric("Risk Threshold", f"{threshold:.3f}")
    
    if prediction_binary == 1:
        st.markdown('<div class="risk-high">⚠️ HIGH RISK AREA</div>', unsafe_allow_html=True)
        st.warning("This location is classified as a high-risk area based on historical seismic patterns.")
    else:
        st.markdown('<div class="risk-low">✅ LOW RISK AREA</div>', unsafe_allow_html=True)
        st.info("This location is classified as a low-risk area based on historical seismic patterns.")

# Detailed results section
st.markdown("---")
st.markdown('<div class="sub-header">📈 Detailed Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Input Parameters**")
    st.write(f"**Latitude:** {latitude}°")
    st.write(f"**Longitude:** {longitude}°")
    st.write(f"**Depth:** {depth} km")
    st.write(f"**Distance from center:** {distance_from_center:.2f}")

with col2:
    st.markdown("**Calculated Features**")
    st.write(f"**Latitude (abs):** {latitude_abs:.2f}")
    st.write(f"**Longitude (abs):** {longitude_abs:.2f}")
    st.write(f"**Depth (log):** {depth_log:.2f}")
    st.write(f"**Lat-Long interaction:** {lat_long_interact:.2f}")

with col3:
    st.markdown("**Risk Metrics**")
    st.write(f"**Risk Probability:** {prediction_proba:.3f}")
    st.write(f"**Zone Risk Score:** {zone_risk_score:.3f}")
    st.write(f"**Predicted Risk:** {'High' if prediction_binary == 1 else 'Low'}")
    st.write(f"**Model Threshold:** {threshold:.3f}")

# Additional visualizations
st.markdown("---")
st.markdown('<div class="sub-header">🔍 Risk Visualization</div>', unsafe_allow_html=True)

# Create a probability distribution plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(['Low Risk Probability', 'High Risk Probability'], 
       [1-prediction_proba, prediction_proba], 
       color=['#4ECDC4', '#FF6B35'])
ax.set_ylabel('Probability')
ax.set_ylim(0, 1)
ax.set_title('Risk Probability Distribution')

# Add value labels on bars
for i, v in enumerate([1-prediction_proba, prediction_proba]):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

st.pyplot(fig)
plt.clf()

# Information section
st.markdown("---")
st.markdown('<div class="sub-header">ℹ️ About This Model</div>', unsafe_allow_html=True)

st.info("""
This earthquake risk prediction model was trained on historical seismic data to identify areas with higher probability of significant seismic activity. The model considers:

- **Geographic coordinates** (latitude and longitude)
- **Depth of seismic activity**
- **Spatial clustering patterns** (using DBSCAN)
- **Historical risk indicators** based on past earthquake data

⚠️ **Important Disclaimers:**
- This model provides risk assessments based on historical patterns and should not be used as the sole basis for critical decisions
- Results are predictions only and actual seismic activity may vary
- Always follow official earthquake preparedness guidelines from PHIVOLCS and local authorities
- The model was trained on limited sample data for demonstration purposes
""")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; padding: 1rem;'>"
           " earthquak Earthquake Risk Prediction System | For Educational/Research Purposes Only</div>", 
           unsafe_allow_html=True)