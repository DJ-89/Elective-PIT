import React from 'react';

const StatsPanel = ({ stats }) => {
  return (
    <div className="stats-panel">
      <div className="stat-card">
        <h3>Model Accuracy</h3>
        <div className="stat-value">{(stats.accuracy * 100).toFixed(2)}%</div>
      </div>
      <div className="stat-card">
        <h3>Seismic Clusters</h3>
        <div className="stat-value">{stats.clusterCount}</div>
      </div>
      <div className="stat-card">
        <h3>Data Points</h3>
        <div className="stat-value">{stats.dataPoints.toLocaleString()}</div>
      </div>
    </div>
  );
};

export default StatsPanel;