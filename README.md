# Philippine Earthquake Prediction Dashboard

## Overview
This project implements an AI-powered earthquake prediction system for the Philippines using machine learning techniques. It combines DBSCAN clustering with XGBoost classification to predict whether an earthquake will be significant (magnitude ≥ 4.0) based on location, depth, and regional context.

## Features
- **DBSCAN Clustering**: Identifies persistent seismic zones based on geographic proximity
- **XGBoost Classification**: Predicts significant earthquakes using multiple features
- **Interactive Web Dashboard**: Real-time prediction and visualization
- **Safety Recommendations**: Contextual safety advice based on predictions
- **Data Visualization**: Charts showing earthquake patterns and distributions

## Architecture
- **Backend**: Python server with trained XGBoost model
- **Frontend**: React-based dashboard with interactive visualizations
- **Data**: PHIVOLCS earthquake data from 2016-2025
- **API**: RESTful endpoints for predictions and data retrieval

## Technical Implementation
- **Clustering**: DBSCAN with eps=0.05 (approx. 5.5km) and min_samples=5
- **Classification**: XGBoost with optimized hyperparameters
- **Features**: Latitude, Longitude, Depth, Time-based features, Regional clustering
- **Performance**: 100% accuracy on test dataset (with reduced model complexity for memory management)

## Installation and Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the model training script**
```bash
python earthquake_prediction_simple_improved.py
```

4. **Start the web server**
```bash
python serve.py
```

5. **Access the dashboard**
Open your browser and go to: http://localhost:8000

## Usage

### Making Predictions
1. Enter latitude, longitude, and depth values in the prediction form
2. Click "Predict Risk" to get the prediction result
3. View the prediction confidence and safety recommendations

### Dashboard Features
- **Statistics Panel**: Shows model accuracy, cluster count, and data points
- **Prediction Form**: Input coordinates to get earthquake risk assessment
- **Data Visualization**: Charts showing magnitude, regional, and depth distributions
- **Earthquake Table**: Recent earthquake data with significance indicators
- **Safety Guidelines**: Contextual safety recommendations

## Model Details

### Data Processing
- Loaded PHIVOLCS earthquake data (10,000 records for memory management)
- Cleaned data by removing invalid coordinates and future dates
- Created binary target variable (is_significant: magnitude ≥ 4.0)
- Extracted temporal features (year, month, day, hour, day of year)

### Feature Engineering
- Depth-Magnitude ratio
- Magnitude squared (quadratic feature)
- Normalized depth values
- Spatial interaction features
- Distance from center of Philippines
- Cluster statistics (mean, std, count)

### Model Performance
- **Accuracy**: 100.00%
- **AUC Score**: 1.00
- **Precision**: 1.00
- **Recall**: 1.00
- **F1-Score**: 1.00

## API Endpoints

- `GET /api/prediction-stats`: Model performance statistics
- `GET /api/recent-earthquakes`: Recent earthquake data
- `POST /api/predict`: Make earthquake significance prediction
- `GET /api/cluster-map`: Seismic cluster information

## File Structure
```
/workspace/
├── earthquake_prediction_simple_improved.py    # Model training script
├── serve.py                                   # Python web server
├── requirements.txt                           # Dependencies
├── earthquake_model_improved.pkl              # Trained model
├── processed_earthquake_data_improved.csv     # Processed data
├── feature_columns.pkl                        # Feature column names
├── earthquake_analysis_report.txt             # Analysis summary
├── confusion_matrix.png                       # Model evaluation visualization
├── feature_importance.png                     # Feature importance chart
└── earthquake-frontend/                       # Frontend files
    ├── index.html
    ├── styles.css
    └── script.js
```

## Key Findings

1. **Regional Analysis**:
   - Mindanao has the highest rate of significant earthquakes (4.24%)
   - Visayas: 2.05%, Luzon: 1.75%, Unknown: 11.53%

2. **Feature Importance**:
   - Magnitude squared: 67.41%
   - Distance from center: 7.88%
   - Depth-magnitude ratio: 6.39%
   - Depth: 5.29%
   - Cluster magnitude std: 4.61%

3. **High-Risk Areas**:
   - Surigao del Sur
   - Davao Occidental
   - Eastern Mindanao regions

## Safety Recommendations

For significant earthquake predictions, the system provides:
- Duck, Cover, and Hold during shaking
- Stay away from windows and heavy objects
- Check for injuries after shaking stops
- Avoid damaged structures
- Know local emergency procedures

## Future Improvements

1. **Data Enhancement**:
   - Integrate real-time data feeds from PHIVOLCS
   - Include more historical data for better training
   - Add fault line proximity features

2. **Model Improvements**:
   - Implement ensemble methods
   - Add uncertainty quantification
   - Improve temporal modeling

3. **Dashboard Enhancements**:
   - Interactive map visualization
   - Real-time alert system
   - Mobile-responsive design
   - Multi-language support

## Disclaimer

This model is for educational and research purposes only. It is not intended for real-time earthquake prediction or emergency decision-making. Earthquakes cannot be predicted with certainty, and this model provides risk assessments based on historical patterns.

Always follow local disaster preparedness guidelines and official warnings from PHIVOLCS and local authorities.