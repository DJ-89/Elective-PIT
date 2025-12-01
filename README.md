# Philippine Earthquake Prediction Dashboard

## Overview
This interactive dashboard provides earthquake risk assessment for the Philippines using machine learning algorithms. The system combines DBSCAN clustering and XGBoost classification to predict significant seismic events (magnitude ≥ 4.0) based on location, depth, and regional factors.

## Features
- 📍 **Real-time Prediction**: Enter coordinates to get immediate risk assessment
- 📊 **Data Visualization**: Interactive charts showing magnitude, regional, and depth distributions
- 🗺️ **Regional Analysis**: Breakdown of seismic activity across Luzon, Visayas, and Mindanao
- 🛡️ **Safety Recommendations**: Contextual advice based on prediction results
- 📈 **Performance Metrics**: Live display of model accuracy and statistics

## Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript with Chart.js for visualizations
- **Backend**: Node.js HTTP server
- **ML Model**: XGBoost classifier with DBSCAN clustering
- **API**: RESTful endpoints for data and predictions

## Installation & Setup

### Prerequisites
- Node.js (v14 or higher)

### Quick Start
1. Clone or download the repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the server:
   ```bash
   npm start
   ```
5. Open your browser and go to `http://localhost:8000`

### Alternative Start
You can also run the server directly:
```bash
node server.js
```

## Usage

### Making Predictions
1. Enter latitude and longitude coordinates
2. Input depth in kilometers
3. Click "Predict Risk Level"
4. View results with confidence percentage
5. See safety recommendations if needed

### Dashboard Sections
- **Prediction Panel**: Input coordinates for risk assessment
- **Statistics**: Model performance metrics
- **Visualizations**: Charts showing seismic patterns
- **Data Table**: Recent earthquake information

## API Endpoints

### GET /api/prediction-stats
Returns model performance statistics:
```json
{
  "accuracy": 99.5,
  "clusters": 2,
  "data_points": 10000,
  "significant_events": 1247
}
```

### GET /api/recent-earthquakes
Returns recent earthquake data:
```json
[
  {
    "date": "2025-01-15",
    "location": "Surigao Del Sur",
    "lat": 9.0,
    "lon": 125.8,
    "depth": 35.2,
    "magnitude": 4.5,
    "significant": true
  }
]
```

### POST /api/predict
Makes a prediction based on coordinates:
```json
{
  "latitude": 9.0,
  "longitude": 125.8,
  "depth": 35.2
}
```

Response:
```json
{
  "is_significant": true,
  "confidence": 0.85,
  "cluster_id": 1,
  "location": {
    "latitude": 9.0,
    "longitude": 125.8,
    "depth": 35.2
  }
}
```

## Machine Learning Model

### Clustering
- **Algorithm**: DBSCAN (Density-Based Spatial Clustering)
- **Parameters**: eps=0.05, min_samples=5
- **Purpose**: Identify persistent seismic zones in the Philippines

### Classification
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Target**: Binary classification (significant vs non-significant)
- **Features**: Latitude, longitude, depth, cluster assignment
- **Performance**: 99.5% accuracy

### Key Insights
- Mindanao has the highest rate of significant earthquakes (5.47%)
- Top predictive features: Latitude, longitude, depth, regional clustering
- High-risk areas: Surigao del Sur, Davao Occidental, Eastern Mindanao

## Safety Recommendations

The dashboard provides contextual safety advice based on prediction results:

### For Significant Predictions:
- 🚨 Drop, Cover, and Hold - Take cover under a sturdy desk or table
- 🚪 Keep exit routes clear and know your evacuation plan
- 📱 Stay updated with PHIVOLCS alerts and local emergency services
- 📦 Prepare emergency kit with water, food, flashlight, and first aid
- 🏗️ Check building safety and retrofit if necessary

### For Non-Significant Predictions:
- ✅ Remain calm - This is a minor seismic event
- 📊 Continue monitoring for updates from PHIVOLCS
- 🏠 Check for minor damages in your immediate area
- 📞 Inform local authorities if you notice any issues

## Data Sources
- **Primary**: PHIVOLCS (Philippine Institute of Volcanology and Seismology)
- **Dataset**: Historical earthquake records from 2016-2025
- **Records**: Over 100,000 earthquake events processed

## Project Structure
```
├── index.html          # Main dashboard page
├── server.js           # Node.js server implementation
├── package.json        # Project dependencies and scripts
└── README.md           # This documentation
```

## Important Note
This system provides risk assessment based on historical data patterns. Actual earthquake prediction remains scientifically impossible. The dashboard should be used as a supplementary tool alongside official PHIVOLCS information and local emergency services.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- PHIVOLCS for providing earthquake data
- Machine learning community for algorithms and techniques
- Disaster preparedness organizations for safety guidelines