const http = require('http');
const fs = require('fs');
const path = require('path');

// Simple HTTP server to serve the earthquake prediction dashboard
const server = http.createServer((req, res) => {
    console.log(`Request received: ${req.method} ${req.url}`);

    // Serve the main HTML file
    if (req.url === '/' || req.url === '/index.html') {
        fs.readFile(path.join(__dirname, 'index.html'), (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading index.html');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    }
    // API endpoint for prediction stats
    else if (req.url === '/api/prediction-stats') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            accuracy: 99.5,
            clusters: 2,
            data_points: 10000,
            significant_events: 1247
        }));
    }
    // API endpoint for recent earthquakes
    else if (req.url === '/api/recent-earthquakes') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify([
            { date: '2025-01-15', location: 'Surigao Del Sur', lat: 9.0, lon: 125.8, depth: 35.2, magnitude: 4.5, significant: true },
            { date: '2025-01-14', location: 'Davao Occidental', lat: 5.6, lon: 125.2, depth: 22.1, magnitude: 3.8, significant: false },
            { date: '2025-01-13', location: 'Cebu', lat: 10.3, lon: 123.9, depth: 15.5, magnitude: 4.2, significant: true },
            { date: '2025-01-12', location: 'Leyte', lat: 11.2, lon: 124.8, depth: 45.0, magnitude: 3.9, significant: false },
            { date: '2025-01-11', location: 'Bataan', lat: 14.5, lon: 120.5, depth: 10.3, magnitude: 4.1, significant: true }
        ]));
    }
    // API endpoint for predictions
    else if (req.url === '/api/predict' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const { latitude, longitude, depth } = data;

                // Mock prediction logic based on location
                let isSignificant = false;
                let confidence = 0;
                let clusterId = 0;

                // Simple mock logic - in reality this would call your ML model
                if (latitude < 12 && longitude > 124) { // Mindanao region
                    isSignificant = Math.random() > 0.3; // Higher chance in Mindanao
                    confidence = isSignificant ? 0.85 : 0.75;
                    clusterId = 1;
                } else if (latitude > 15) { // Northern Luzon
                    isSignificant = Math.random() > 0.6; // Lower chance
                    confidence = isSignificant ? 0.65 : 0.85;
                    clusterId = 0;
                } else { // Central areas
                    isSignificant = Math.random() > 0.5; // Medium chance
                    confidence = 0.70;
                    clusterId = 0;
                }

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    is_significant: isSignificant,
                    confidence: confidence,
                    cluster_id: clusterId,
                    location: { latitude, longitude, depth }
                }));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid request data' }));
            }
        });
    }
    // 404 for other routes
    else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
    console.log(`Philippine Earthquake Prediction Dashboard server running on http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop the server');
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down server...');
    server.close(() => {
        console.log('Server closed.');
        process.exit(0);
    });
});