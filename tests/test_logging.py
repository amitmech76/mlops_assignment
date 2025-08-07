#!/usr/bin/env python3
"""
Test script for logging functionality
Makes multiple prediction requests to test logging and metrics
"""

import requests
import time
import json
from datetime import datetime

# API configuration
BASE_URL = "http://localhost:8000"

def test_housing_prediction():
    """Test California Housing prediction"""
    housing_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = requests.post(f"{BASE_URL}/predict/housing", json=housing_data)
    return response.json() if response.status_code == 200 else None

def test_iris_prediction():
    """Test Iris classification"""
    iris_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = requests.post(f"{BASE_URL}/predict/iris", json=iris_data)
    return response.json() if response.status_code == 200 else None

def test_invalid_request():
    """Test invalid request to trigger error logging"""
    invalid_data = {
        "MedInc": "invalid",
        "HouseAge": 41.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/housing", json=invalid_data)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_metrics():
    """Get current metrics"""
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_recent_logs():
    """Get recent logs"""
    try:
        response = requests.get(f"{BASE_URL}/logs/recent?limit=5")
        return response.json() if response.status_code == 200 else None
    except:
        return None

def main():
    """Main test function"""
    print("🧪 Testing Logging and Metrics Functionality")
    print("=" * 50)
    
    # Check if API is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code != 200:
            print("❌ API is not running. Please start the API first.")
            return
        print("✅ API is running")
    except:
        print("❌ Cannot connect to API. Please start the API first.")
        return
    
    print("\n📊 Initial Metrics:")
    initial_metrics = get_metrics()
    if initial_metrics:
        overall = initial_metrics.get('overall_stats', {})
        print(f"   Total Predictions: {overall.get('total_predictions', 0)}")
        print(f"   Successful: {overall.get('successful_predictions', 0)}")
        print(f"   Error Rate: {overall.get('error_rate_percent', 0)}%")
    
    print("\n🚀 Making Test Predictions...")
    
    # Make multiple predictions
    for i in range(5):
        print(f"\n--- Test {i+1} ---")
        
        # Housing prediction
        print("🏠 Testing Housing Prediction...")
        housing_result = test_housing_prediction()
        if housing_result:
            print(f"   ✅ Prediction: {housing_result.get('prediction', 'N/A')}")
        else:
            print("   ❌ Housing prediction failed")
        
        time.sleep(0.5)
        
        # Iris prediction
        print("🌸 Testing Iris Prediction...")
        iris_result = test_iris_prediction()
        if iris_result:
            print(f"   ✅ Prediction: {iris_result.get('prediction', 'N/A')}")
            if iris_result.get('confidence'):
                print(f"   📊 Confidence: {iris_result.get('confidence'):.2f}")
        else:
            print("   ❌ Iris prediction failed")
        
        time.sleep(0.5)
    
    # Test invalid request
    print("\n⚠️  Testing Invalid Request...")
    invalid_result = test_invalid_request()
    if invalid_result:
        print("   ❌ Invalid request should have failed")
    else:
        print("   ✅ Invalid request properly rejected")
    
    print("\n📈 Final Metrics:")
    final_metrics = get_metrics()
    if final_metrics:
        overall = final_metrics.get('overall_stats', {})
        print(f"   Total Predictions: {overall.get('total_predictions', 0)}")
        print(f"   Successful: {overall.get('successful_predictions', 0)}")
        print(f"   Error Rate: {overall.get('error_rate_percent', 0)}%")
        print(f"   Avg Processing Time: {overall.get('avg_processing_time_ms', 0)}ms")
        
        # Model-specific stats
        model_stats = final_metrics.get('model_stats', {})
        for model, stats in model_stats.items():
            print(f"   📋 {model.title()}: {stats.get('total_predictions', 0)} predictions")
    
    print("\n📝 Recent Logs:")
    logs = get_recent_logs()
    if logs and logs.get('recent_predictions'):
        for pred in logs['recent_predictions'][:3]:
            status = "✅" if pred['status'] == 'success' else "❌"
            print(f"   {status} {pred['model_name']}: {pred['prediction']} ({pred['processing_time_ms']:.1f}ms)")
    else:
        print("   📭 No recent logs available")
    
    print("\n🎉 Logging test completed!")
    print("💡 Check the logs directory for detailed logs and SQLite database")

if __name__ == "__main__":
    main() 