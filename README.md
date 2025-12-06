# Earthquake Risk Prediction System

This is a Streamlit-based web application that predicts earthquake risk levels based on geographical coordinates and depth data using a trained XGBoost model.

## Features

- Interactive earthquake risk prediction based on latitude, longitude, and depth
- Visual risk assessment with color-coded indicators
- Geographic mapping of risk locations
- Detailed feature analysis and probability visualization
- Historical seismic pattern analysis

## Model Information

The model was trained using:
- **Algorithm**: XGBoost Classifier
- **Features**: Latitude, Longitude, Depth, Seismic Zones, and engineered features
- **Target**: High-risk area classification based on historical patterns (shallow quakes ≤15km and magnitude ≥4.0)

## Files Included

- `app.py`: Main Streamlit application
- `model.py`: Model training script
- `risk_area_identifier.pkl`: Trained XGBoost model
- `scaler_risk_identifier.pkl`: Feature scaler
- `dbscan_zone_identifier.pkl`: DBSCAN clustering model
- `threshold_risk_identifier.pkl`: Classification threshold
- `feature_cols_risk_identifier.pkl`: Feature column names
- `zone_risk_lookup.pkl`: Zone risk lookup table
- `requirements.txt`: Python dependencies
- `phivolcs_earthquake_data.csv`: Sample training data

## Deployment Options

### 1. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository and deploy

### 3. Deploy to Heroku

1. Create a Heroku account and install the Heroku CLI
2. Create a new app in your Heroku dashboard
3. Connect to your GitHub repository or deploy using the CLI

### 4. Deploy to other platforms

The application can also be deployed to:
- AWS (using EC2 or Elastic Beanstalk)
- Google Cloud Platform
- Azure
- Railway
- Vercel (with proper configuration)

## Important Disclaimers

⚠️ **This model provides risk assessments based on historical patterns and should not be used as the sole basis for critical decisions**

- Results are predictions only and actual seismic activity may vary
- Always follow official earthquake preparedness guidelines from PHIVOLCS and local authorities
- The model was trained on sample data for demonstration purposes
- For actual seismic risk assessment, consult with geological experts and official sources

## Model Performance

- AUC-ROC Score: ~0.67 (on sample data)
- Cross-validation F1 score: ~0.50 (on sample data)
- Note: Performance may vary with real-world data

## Architecture

The system uses a multi-step approach:
1. **DBSCAN Clustering**: Groups locations into seismic zones
2. **Feature Engineering**: Creates additional predictive features
3. **XGBoost Classification**: Predicts risk levels based on engineered features
4. **Threshold Optimization**: Balances precision and recall for optimal predictions