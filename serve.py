#!/usr/bin/env python3
"""
Earthquake Prediction Dashboard Server
This server serves the React frontend and provides API endpoints for the earthquake prediction model.
"""

import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
import threading
import time
import random

# Mock earthquake prediction function
def predict_earthquake_significance(latitude, longitude, depth):
    """
    Mock prediction function that simulates the ML model
    In a real application, this would load the trained model and make predictions
    """
    # Simulate model processing time
    time.sleep(0.5)
    
    # Create a more sophisticated mock prediction based on location
    # Areas near the Philippines (Ring of Fire) have higher probability
    philippines_region = (5 <= latitude <= 20) and (110 <= longitude <= 130)
    
    if philippines_region:
        # Higher probability in the Philippines region
        is_significant = random.random() > 0.2  # 80% chance of being significant
        base_confidence = 0.8
    else:
        # Lower probability elsewhere
        is_significant = random.random() > 0.8  # 20% chance of being significant
        base_confidence = 0.7
    
    # Adjust confidence based on depth (shallow quakes are more likely to be significant)
    depth_factor = 1.0 if depth < 70 else 0.8  # Shallow quakes more significant
    
    # Add some randomness to confidence
    confidence = min(0.99, max(0.7, base_confidence * depth_factor + random.uniform(-0.1, 0.1)))
    
    return {
        "is_significant": is_significant,
        "confidence": round(confidence * 100, 2),
        "probability": round(confidence, 2)
    }

# Mock data for recent earthquakes
MOCK_EARTHQUAKES = [
    {"id": 1, "latitude": 14.5995, "longitude": 120.9842, "depth": 10, "magnitude": 4.5, "date": "2023-01-15", "isSignificant": True},
    {"id": 2, "latitude": 10.3157, "longitude": 123.9547, "depth": 25, "magnitude": 3.2, "date": "2023-01-20", "isSignificant": False},
    {"id": 3, "latitude": 15.2121, "longitude": 120.5528, "depth": 5, "magnitude": 5.1, "date": "2023-01-25", "isSignificant": True},
    {"id": 4, "latitude": 12.8797, "longitude": 121.1696, "depth": 35, "magnitude": 2.8, "date": "2023-02-01", "isSignificant": False},
    {"id": 5, "latitude": 18.1096, "longitude": 121.4968, "depth": 15, "magnitude": 4.2, "date": "2023-02-05", "isSignificant": True},
    {"id": 6, "latitude": 13.4125, "longitude": 123.6087, "depth": 50, "magnitude": 3.8, "date": "2023-02-10", "isSignificant": False},
    {"id": 7, "latitude": 9.2347, "longitude": 125.7453, "depth": 75, "magnitude": 4.9, "date": "2023-02-15", "isSignificant": True},
    {"id": 8, "latitude": 17.5995, "longitude": 121.1234, "depth": 5, "magnitude": 2.5, "date": "2023-02-20", "isSignificant": False},
]

class EarthquakeServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/prediction-stats':
            # Return model statistics
            stats = {
                "model_accuracy": 99.5,
                "predictions_made": 1248,
                "data_points": 10234,
                "last_updated": "2023-12-01"
            }
            self.send_api_response(stats)
        
        elif parsed_path.path == '/api/recent-earthquakes':
            # Return recent earthquake data
            self.send_api_response(MOCK_EARTHQUAKES)
        
        elif parsed_path.path.startswith('/api/predict'):
            # Handle prediction request
            params = parse_qs(parsed_path.query)
            
            try:
                latitude = float(params.get('latitude', [14.5995])[0])
                longitude = float(params.get('longitude', [120.9842])[0])
                depth = float(params.get('depth', [10])[0])
                
                result = predict_earthquake_significance(latitude, longitude, depth)
                
                self.send_api_response({
                    "prediction": result["is_significant"],
                    "confidence": result["confidence"],
                    "probability": result["probability"],
                    "message": "Prediction successful"
                })
                
            except ValueError:
                self.send_error_response("Invalid parameters", 400)
        
        else:
            # Serve static files (React app)
            if self.path == '/' or self.path.startswith('/?'):
                self.path = '/index.html'
            
            # Check if file exists in the frontend build directory
            frontend_path = os.path.join(os.path.dirname(__file__), 'earthquake-frontend', self.path.lstrip('/'))
            
            # If the file exists in the frontend directory, serve it
            if os.path.exists(frontend_path):
                self.path = f'earthquake-frontend{self.path}'
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
            else:
                # If not found, serve index.html for client-side routing
                self.path = '/earthquake-frontend/index.html'
                return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/api/predict':
            try:
                # Get content length and read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse JSON data
                data = json.loads(post_data.decode('utf-8'))
                
                latitude = float(data.get('latitude', 14.5995))
                longitude = float(data.get('longitude', 120.9842))
                depth = float(data.get('depth', 10))
                
                result = predict_earthquake_significance(latitude, longitude, depth)
                
                self.send_api_response({
                    "prediction": result["is_significant"],
                    "confidence": result["confidence"],
                    "probability": result["probability"],
                    "message": "Prediction successful"
                })
                
            except json.JSONDecodeError:
                self.send_error_response("Invalid JSON", 400)
            except ValueError:
                self.send_error_response("Invalid parameters", 400)
            except Exception as e:
                self.send_error_response(f"Server error: {str(e)}", 500)
        else:
            self.send_error(404)

    def send_api_response(self, data):
        """Send JSON API response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, message, status_code=400):
        """Send error response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

def start_server():
    PORT = 8000
    
    # Change to the workspace directory to serve files properly
    os.chdir('/workspace')
    
    with socketserver.TCPServer(("", PORT), EarthquakeServerHandler) as httpd:
        print(f"🌍 Earthquake Prediction Dashboard server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    start_server()