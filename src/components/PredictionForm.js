import React, { useState } from 'react';

const PredictionForm = ({ onPredict, loading, result, error }) => {
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(formData);
  };

  return (
    <div className="prediction-form">
      <h2>Earthquake Risk Assessment</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="latitude">Latitude:</label>
          <input
            type="number"
            id="latitude"
            name="latitude"
            value={formData.latitude}
            onChange={handleChange}
            step="any"
            required
            placeholder="Enter latitude (e.g., 14.5995)"
          />
        </div>
        <div className="form-group">
          <label htmlFor="longitude">Longitude:</label>
          <input
            type="number"
            id="longitude"
            name="longitude"
            value={formData.longitude}
            onChange={handleChange}
            step="any"
            required
            placeholder="Enter longitude (e.g., 120.9842)"
          />
        </div>
        <button 
          type="submit" 
          className="btn"
          disabled={loading}
        >
          {loading ? (
            <span className="loading">Predicting...</span>
          ) : (
            'Predict Risk'
          )}
        </button>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {result && (
        <div className={`prediction-result ${result.significant ? 'significant' : 'not-significant'}`}>
          <h3>Prediction Result</h3>
          <p><strong>Significant Earthquake:</strong> {result.significant ? 'Yes' : 'No'}</p>
          <p><strong>Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
          <p><strong>Cluster ID:</strong> {result.cluster_id}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          {result.safety_recommendations && (
            <div>
              <h4>Safety Recommendations:</h4>
              <p>{result.safety_recommendations}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PredictionForm;