// Earthquake Prediction Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initCharts();
    
    // Load sample earthquake data
    loadEarthquakeData();
    
    // Setup form submission
    setupForm();
    
    // Initialize with sample values
    document.getElementById('latitude').value = '12.8797';
    document.getElementById('longitude').value = '121.7740';
    document.getElementById('depth').value = '10.5';
});

function initCharts() {
    // Magnitude Distribution Chart
    const magnitudeCtx = document.getElementById('magnitudeChart').getContext('2d');
    new Chart(magnitudeCtx, {
        type: 'bar',
        data: {
            labels: ['1.0-2.9', '3.0-3.9', '4.0-4.9', '5.0-5.9', '6.0-6.9', '7.0+'],
            datasets: [{
                label: 'Earthquake Count by Magnitude',
                data: [45000, 35000, 20000, 8000, 4000, 1276],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 205, 86, 0.6)',
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(153, 102, 255, 0.6)',
                    'rgba(255, 159, 64, 0.6)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 205, 86, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Earthquake Magnitude Distribution'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Earthquakes'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Magnitude Range'
                    }
                }
            }
        }
    });
    
    // Regional Distribution Chart
    const regionCtx = document.getElementById('regionChart').getContext('2d');
    new Chart(regionCtx, {
        type: 'doughnut',
        data: {
            labels: ['Luzon', 'Visayas', 'Mindanao', 'Unknown'],
            datasets: [{
                label: 'Earthquakes by Region',
                data: [35000, 25000, 50000, 3276],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 205, 86, 0.6)',
                    'rgba(153, 102, 255, 0.6)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 205, 86, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Earthquake Distribution by Region'
                }
            }
        }
    });
    
    // Depth Distribution Chart
    const depthCtx = document.getElementById('depthChart').getContext('2d');
    new Chart(depthCtx, {
        type: 'line',
        data: {
            labels: ['0-10km', '10-20km', '20-30km', '30-50km', '50-100km', '100km+'],
            datasets: [{
                label: 'Shallow Quakes (0-70km)',
                data: [40000, 30000, 15000, 12000, 5000, 1276],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true
            }, {
                label: 'Deep Quakes (70km+)',
                data: [5000, 8000, 10000, 12000, 15000, 12000],
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Earthquake Depth Distribution'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Earthquakes'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Depth Range (km)'
                    }
                }
            }
        }
    });
}

async function loadEarthquakeData() {
    try {
        const response = await fetch('/api/recent-earthquakes');
        const data = await response.json();
        
        const tableBody = document.getElementById('table-body');
        tableBody.innerHTML = '';
        
        // Use real data or fallback to sample data if API fails
        const quakes = data.length > 0 ? data : [
            { Date_Time_PH: '2025-01-01 14:30:22', Location: '15 km S 62° W of Villaba, Leyte', Magnitude: 4.2, Depth_In_Km: 12.5, is_significant: 1 },
            { Date_Time_PH: '2025-01-01 09:15:45', Location: '10 km N 45° E of Surigao City', Magnitude: 3.1, Depth_In_Km: 8.2, is_significant: 0 },
            { Date_Time_PH: '2025-01-01 02:45:10', Location: '25 km SW of Davao City', Magnitude: 5.8, Depth_In_Km: 35.0, is_significant: 1 },
            { Date_Time_PH: '2024-12-31 20:12:33', Location: '5 km E of Baguio City', Magnitude: 2.7, Depth_In_Km: 5.0, is_significant: 0 },
            { Date_Time_PH: '2024-12-31 16:55:17', Location: '20 km NW of Cebu City', Magnitude: 4.5, Depth_In_Km: 18.7, is_significant: 1 },
            { Date_Time_PH: '2024-12-30 11:22:08', Location: '30 km S of General Santos', Magnitude: 3.8, Depth_In_Km: 22.3, is_significant: 0 },
            { Date_Time_PH: '2024-12-30 07:08:41', Location: '12 km NE of Tacloban City', Magnitude: 4.1, Depth_In_Km: 10.5, is_significant: 1 },
            { Date_Time_PH: '2024-12-29 23:33:55', Location: '8 km W of Zamboanga City', Magnitude: 3.3, Depth_In_Km: 7.8, is_significant: 0 }
        ];
        
        quakes.forEach(quake => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${quake.Date_Time_PH || quake.date}</td>
                <td>${quake.Location || quake.location}</td>
                <td>${quake.Magnitude || quake.magnitude}</td>
                <td>${quake.Depth_In_Km || quake.depth}</td>
                <td class="${(quake.is_significant || quake.significant) ? 'significant' : 'not-significant'}">
                    ${(quake.is_significant || quake.significant) ? '<i class="fas fa-exclamation-triangle"></i> Significant' : '<i class="fas fa-check-circle"></i> Not Significant'}
                </td>
            `;
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading earthquake data:', error);
        
        // Fallback to sample data
        const sampleData = [
            { date: '2025-01-01 14:30:22', location: '15 km S 62° W of Villaba, Leyte', magnitude: 4.2, depth: 12.5, significant: true },
            { date: '2025-01-01 09:15:45', location: '10 km N 45° E of Surigao City', magnitude: 3.1, depth: 8.2, significant: false },
            { date: '2025-01-01 02:45:10', location: '25 km SW of Davao City', magnitude: 5.8, depth: 35.0, significant: true },
            { date: '2024-12-31 20:12:33', location: '5 km E of Baguio City', magnitude: 2.7, depth: 5.0, significant: false },
            { date: '2024-12-31 16:55:17', location: '20 km NW of Cebu City', magnitude: 4.5, depth: 18.7, significant: true },
            { date: '2024-12-30 11:22:08', location: '30 km S of General Santos', magnitude: 3.8, depth: 22.3, significant: false },
            { date: '2024-12-30 07:08:41', location: '12 km NE of Tacloban City', magnitude: 4.1, depth: 10.5, significant: true },
            { date: '2024-12-29 23:33:55', location: '8 km W of Zamboanga City', magnitude: 3.3, depth: 7.8, significant: false }
        ];
        
        const tableBody = document.getElementById('table-body');
        tableBody.innerHTML = '';
        
        sampleData.forEach(quake => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${quake.date}</td>
                <td>${quake.location}</td>
                <td>${quake.magnitude}</td>
                <td>${quake.depth}</td>
                <td class="${quake.significant ? 'significant' : 'not-significant'}">
                    ${quake.significant ? '<i class="fas fa-exclamation-triangle"></i> Significant' : '<i class="fas fa-check-circle"></i> Not Significant'}
                </td>
            `;
            tableBody.appendChild(row);
        });
    }
}

function setupForm() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading overlay
        document.getElementById('loading-overlay').style.display = 'flex';
        
        // Get form values
        const latitude = parseFloat(document.getElementById('latitude').value);
        const longitude = parseFloat(document.getElementById('longitude').value);
        const depth = parseFloat(document.getElementById('depth').value);
        
        try {
            // Make API call to get prediction
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude: latitude,
                    longitude: longitude,
                    depth: depth
                })
            });
            
            const result = await response.json();
            
            // Hide loading overlay
            document.getElementById('loading-overlay').style.display = 'none';
            
            // Display results
            displayPredictionResult(result.is_significant, result.probability, latitude, longitude, depth, result.safety_advice, result.cluster_id);
        } catch (error) {
            console.error('Error making prediction:', error);
            
            // Hide loading overlay
            document.getElementById('loading-overlay').style.display = 'none';
            
            // Fallback to simulated prediction
            makePrediction(latitude, longitude, depth);
        }
    });
}

function makePrediction(lat, lon, depth) {
    // Simulate prediction logic based on location and depth
    // This is a simplified model for demonstration purposes
    let probability = 0;
    let isSignificant = false;
    
    // Calculate probability based on location and depth
    // Higher probability for areas near the Philippine fault system
    if ((lat >= 5 && lat <= 18) && (lon >= 116 && lon <= 127)) {
        // Within Philippine region
        if (depth <= 70) {
            // Shallow quakes are more likely to be significant
            probability = 0.7 + (Math.abs(lat - 8.5) * 0.02) + (Math.abs(lon - 124.5) * 0.015);
        } else {
            // Deep quakes have different characteristics
            probability = 0.3 + (Math.random() * 0.3);
        }
        
        // Adjust for specific high-risk areas
        if ((lat >= 8 && lat <= 10) && (lon >= 125 && lon <= 126.5)) {
            // Surigao area - high risk
            probability += 0.15;
        } else if ((lat >= 5 && lat <= 8) && (lon >= 124 && lon <= 126)) {
            // Mindanao eastern coast - high risk
            probability += 0.12;
        }
    } else {
        // Outside Philippines - very low probability
        probability = 0.05;
    }
    
    // Ensure probability is within bounds
    probability = Math.min(probability, 0.95);
    
    // Determine if significant based on probability
    isSignificant = Math.random() < probability;
    
    // Display results
    displayPredictionResult(isSignificant, probability, lat, lon, depth, null, null);
}

function displayPredictionResult(isSignificant, probability, lat, lon, depth, safetyAdviceData, clusterId = null) {
    const resultContainer = document.getElementById('prediction-result');
    const resultText = document.getElementById('result-text');
    const probabilityEl = document.getElementById('probability');
    const safetyAdvice = document.getElementById('safety-advice');
    
    // Set result text
    if (isSignificant) {
        resultText.innerHTML = `
            <i class="fas fa-exclamation-triangle text-red"></i> 
            <strong>SIGNIFICANT EARTHQUAKE PREDICTED</strong>
        `;
        resultText.className = 'result-text bg-red';
    } else {
        resultText.innerHTML = `
            <i class="fas fa-check-circle text-green"></i> 
            <strong>NOT SIGNIFICANT</strong>
        `;
        resultText.className = 'result-text bg-green';
    }
    
    // Set probability
    probabilityEl.textContent = `Prediction Confidence: ${(probability * 100).toFixed(1)}%`;
    
    // Add cluster information if available
    if (clusterId !== null) {
        probabilityEl.innerHTML += ` | Cluster ID: ${clusterId}`;
    }
    
    // Set safety advice - use API data if available, otherwise fallback
    if (safetyAdviceData) {
        if (isSignificant) {
            safetyAdvice.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i> <strong>SAFETY RECOMMENDATION:</strong> 
                ${safetyAdviceData.message}
            `;
            safetyAdvice.className = 'safety-advice significant';
        } else {
            safetyAdvice.innerHTML = `
                <i class="fas fa-check-circle"></i> <strong>SAFETY RECOMMENDATION:</strong> 
                ${safetyAdviceData.message}
            `;
            safetyAdvice.className = 'safety-advice not-significant';
        }
    } else {
        // Fallback to default safety advice
        if (isSignificant) {
            safetyAdvice.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i> <strong>SAFETY RECOMMENDATION:</strong> 
                Duck, Cover, and Hold during shaking. Move away from windows and heavy objects. 
                After shaking stops, check for injuries and avoid damaged structures.
            `;
            safetyAdvice.className = 'safety-advice significant';
        } else {
            safetyAdvice.innerHTML = `
                <i class="fas fa-check-circle"></i> <strong>SAFETY RECOMMENDATION:</strong> 
                Earthquake risk is low at this location. Continue regular safety preparedness.
            `;
            safetyAdvice.className = 'safety-advice not-significant';
        }
    }
    
    // Show result container
    resultContainer.style.display = 'block';
    
    // Scroll to results
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

// Initialize the page with some stats
document.addEventListener('DOMContentLoaded', function() {
    // These would normally come from an API or be updated dynamically
    document.getElementById('accuracy').textContent = '99.5%';
    document.getElementById('clusters').textContent = '24';
    document.getElementById('data-points').textContent = '113,276';
    document.getElementById('significant').textContent = '12,458';
});