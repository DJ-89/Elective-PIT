import React, { useState, useEffect } from 'react';
import './App.css';
import EarthquakeChart from './components/EarthquakeChart';
import PredictionForm from './components/PredictionForm';
import StatsPanel from './components/StatsPanel';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    accuracy: 0,
    clusterCount: 0,
    dataPoints: 0
  });

  useEffect(() => {
    // Fetch initial stats
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/prediction-stats');
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      setPredictionResult(result);
    } catch (err) {
      setError('Error making prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>Philippine Earthquake Prediction Dashboard</h1>
        <p>Real-time seismic risk assessment using machine learning</p>
      </header>

      <main className="app-main">
        <StatsPanel stats={stats} />
        
        <PredictionForm 
          onPredict={handlePrediction} 
          loading={loading}
          result={predictionResult}
          error={error}
        />
        
        <EarthquakeChart />
      </main>

      <footer className="app-footer">
        <p>Powered by machine learning and Philippine seismic data</p>
      </footer>
    </div>
  );
}

export default App;