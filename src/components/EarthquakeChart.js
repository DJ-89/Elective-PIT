import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

const EarthquakeChart = () => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    // Destroy previous chart instance if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    
    // Sample data - in a real app, you would fetch this from your API
    const data = {
      labels: ['Magnitude 1-3', 'Magnitude 4-5', 'Magnitude 6-7', 'Magnitude 8+'],
      datasets: [
        {
          label: 'Earthquake Distribution',
          data: [45, 35, 15, 5], // Sample data
          backgroundColor: [
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(255, 99, 132, 0.6)',
            'rgba(153, 102, 255, 0.6)'
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(153, 102, 255, 1)'
          ],
          borderWidth: 1
        }
      ]
    };

    const config = {
      type: 'bar',
      data: data,
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Earthquake Magnitude Distribution'
          }
        }
      }
    };

    chartInstance.current = new Chart(ctx, config);

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  return (
    <div className="chart-container">
      <h2>Seismic Data Visualization</h2>
      <canvas ref={chartRef} id="earthquakeChart"></canvas>
    </div>
  );
};

export default EarthquakeChart;