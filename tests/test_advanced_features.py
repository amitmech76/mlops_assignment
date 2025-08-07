#!/usr/bin/env python3
"""
Test script for advanced features:
- Input validation
- Model retraining
- Data drift detection
- Prometheus metrics
"""

import requests
import time
import json
import pandas as pd
from datetime import datetime

# API configuration
BASE_URL = "http://localhost:8000"

def test_input_validation():
    """Test input validation with various scenarios"""
    print("🧪 Testing Input Validation")
    print("=" * 40)
    
    # Test valid housing data
    print("\n✅ Testing Valid Housing Data:")
    valid_housing = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = requests.post(f"{BASE_URL}/predict/housing", json=valid_housing)
    if response.status_code == 200:
        print("   ✅ Valid housing data accepted")
    else:
        print(f"   ❌ Valid housing data rejected: {response.text}")
    
    # Test invalid housing data
    print("\n❌ Testing Invalid Housing Data:")
    invalid_housing = {
        "MedInc": 100.0,  # Too high
        "HouseAge": 150.0,  # Too old
        "AveRooms": 0.5,   # Too few rooms
        "AveBedrms": 0.1,  # Too few bedrooms
        "Population": 50000,  # Too high
        "AveOccup": 100.0,  # Too high
        "Latitude": 50.0,   # Outside California
        "Longitude": -100.0  # Outside California
    }
    
    response = requests.post(f"{BASE_URL}/predict/housing", json=invalid_housing)
    if response.status_code == 422:
        print("   ✅ Invalid housing data properly rejected")
        print(f"   📝 Validation errors: {response.json()}")
    else:
        print(f"   ❌ Invalid housing data should have been rejected: {response.status_code}")
    
    # Test valid iris data
    print("\n✅ Testing Valid Iris Data:")
    valid_iris = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = requests.post(f"{BASE_URL}/predict/iris", json=valid_iris)
    if response.status_code == 200:
        print("   ✅ Valid iris data accepted")
    else:
        print(f"   ❌ Valid iris data rejected: {response.text}")
    
    # Test invalid iris data
    print("\n❌ Testing Invalid Iris Data:")
    invalid_iris = {
        "sepal_length": 15.0,  # Too long
        "sepal_width": 20.0,   # Too wide
        "petal_length": 0.1,   # Too short
        "petal_width": 25.0    # Too wide
    }
    
    response = requests.post(f"{BASE_URL}/predict/iris", json=invalid_iris)
    if response.status_code == 422:
        print("   ✅ Invalid iris data properly rejected")
        print(f"   📝 Validation errors: {response.json()}")
    else:
        print(f"   ❌ Invalid iris data should have been rejected: {response.status_code}")

def test_model_retraining():
    """Test model retraining functionality"""
    print("\n🔄 Testing Model Retraining")
    print("=" * 40)
    
    # Test housing model retraining
    print("\n🏠 Testing Housing Model Retraining:")
    retraining_request = {
        "model_name": "housing",
        "trigger_type": "manual",
        "data_path": None,  # Use default data
        "parameters": {"n_estimators": 100}
    }
    
    response = requests.post(f"{BASE_URL}/retrain", json=retraining_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Retraining started: {result['job_id']}")
        print(f"   📊 Status: {result['status']}")
        print(f"   ⏱️  Estimated duration: {result['estimated_duration']} minutes")
        
        # Check job status
        job_id = result['job_id']
        time.sleep(2)  # Wait a bit
        
        status_response = requests.get(f"{BASE_URL}/retrain/status/{job_id}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   📈 Job status: {status['status']}")
        else:
            print(f"   ❌ Could not get job status: {status_response.text}")
    else:
        print(f"   ❌ Retraining failed: {response.text}")
    
    # Test iris model retraining
    print("\n🌸 Testing Iris Model Retraining:")
    retraining_request = {
        "model_name": "iris",
        "trigger_type": "manual",
        "data_path": None,
        "parameters": {"max_iter": 200}
    }
    
    response = requests.post(f"{BASE_URL}/retrain", json=retraining_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Retraining started: {result['job_id']}")
        print(f"   📊 Status: {result['status']}")
    else:
        print(f"   ❌ Retraining failed: {response.text}")
    
    # Get all retraining jobs
    print("\n📋 Getting All Retraining Jobs:")
    response = requests.get(f"{BASE_URL}/retrain/jobs")
    if response.status_code == 200:
        jobs = response.json()['jobs']
        print(f"   📊 Total jobs: {len(jobs)}")
        for job in jobs[:3]:  # Show first 3 jobs
            print(f"   - {job['job_id']}: {job['status']}")
    else:
        print(f"   ❌ Could not get jobs: {response.text}")

def test_data_drift_detection():
    """Test data drift detection"""
    print("\n📊 Testing Data Drift Detection")
    print("=" * 40)
    
    # Create sample reference and current data
    print("\n📈 Creating Sample Data for Drift Detection:")
    
    # Reference data (original distribution)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.uniform(0, 10, 1000)
    })
    
    # Current data (slightly drifted)
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),  # Shifted mean
        'feature2': np.random.normal(5, 2, 1000),      # Same
        'feature3': np.random.uniform(2, 12, 1000)     # Shifted range
    })
    
    # Save data files
    reference_data.to_csv('reference_data.csv', index=False)
    current_data.to_csv('current_data.csv', index=False)
    
    print("   ✅ Sample data created")
    print(f"   📁 Reference data: {len(reference_data)} samples")
    print(f"   📁 Current data: {len(current_data)} samples")
    
    # Test drift detection
    print("\n🔍 Testing Drift Detection:")
    drift_request = {
        "model_name": "housing",
        "reference_data_path": "reference_data.csv",
        "current_data_path": "current_data.csv",
        "threshold": 0.05
    }
    
    response = requests.post(f"{BASE_URL}/drift/detect", json=drift_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   📊 Drift detected: {result['drift_detected']}")
        print(f"   📈 Drift score: {result['drift_score']:.4f}")
        print(f"   📋 Features with drift: {result['features_with_drift']}")
        print(f"   💡 Recommendation: {result['recommendation']}")
    else:
        print(f"   ❌ Drift detection failed: {response.text}")

def test_prometheus_metrics():
    """Test Prometheus metrics endpoint"""
    print("\n📊 Testing Prometheus Metrics")
    print("=" * 40)
    
    # Get Prometheus metrics
    response = requests.get(f"{BASE_URL}/metrics/prometheus")
    if response.status_code == 200:
        metrics = response.text
        print("   ✅ Prometheus metrics endpoint working")
        
        # Check for specific metrics
        metric_lines = metrics.split('\n')
        api_metrics = [line for line in metric_lines if 'ml_api_' in line]
        model_metrics = [line for line in metric_lines if 'ml_model_' in line]
        
        print(f"   📈 API metrics found: {len(api_metrics)}")
        print(f"   🤖 Model metrics found: {len(model_metrics)}")
        
        # Show some example metrics
        print("\n   📋 Example metrics:")
        for metric in api_metrics[:3]:
            print(f"   - {metric}")
        for metric in model_metrics[:3]:
            print(f"   - {metric}")
    else:
        print(f"   ❌ Prometheus metrics failed: {response.status_code}")

def test_advanced_health_check():
    """Test advanced health check"""
    print("\n🏥 Testing Advanced Health Check")
    print("=" * 40)
    
    response = requests.get(f"{BASE_URL}/health/advanced")
    if response.status_code == 200:
        health = response.json()
        print(f"   📊 Overall status: {health['status']}")
        print(f"   🤖 Models loaded: {health['models_loaded']}")
        print(f"   💾 Database connection: {health['database_connection']}")
        print(f"   📈 Prometheus connection: {health['prometheus_connection']}")
        print(f"   ⏰ Last updated: {health['last_updated']}")
    else:
        print(f"   ❌ Advanced health check failed: {response.status_code}")

def generate_load():
    """Generate load for metrics testing"""
    print("\n🚀 Generating Load for Metrics Testing")
    print("=" * 40)
    
    # Make multiple requests to generate metrics
    for i in range(10):
        # Housing prediction
        housing_data = {
            "MedInc": 8.3252 + i * 0.1,
            "HouseAge": 41.0 + i,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0 + i * 10,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
        
        response = requests.post(f"{BASE_URL}/predict/housing", json=housing_data)
        if response.status_code == 200:
            print(f"   ✅ Housing prediction {i+1}/10")
        
        # Iris prediction
        iris_data = {
            "sepal_length": 5.1 + i * 0.1,
            "sepal_width": 3.5,
            "petal_length": 1.4 + i * 0.05,
            "petal_width": 0.2
        }
        
        response = requests.post(f"{BASE_URL}/predict/iris", json=iris_data)
        if response.status_code == 200:
            print(f"   ✅ Iris prediction {i+1}/10")
        
        time.sleep(0.1)  # Small delay

def main():
    """Main test function"""
    print("🧪 Testing Advanced ML API Features")
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
    
    # Run all tests
    test_input_validation()
    test_model_retraining()
    test_data_drift_detection()
    test_prometheus_metrics()
    test_advanced_health_check()
    generate_load()
    
    print("\n🎉 Advanced features test completed!")
    print("\n📊 Monitoring URLs:")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000 (admin/admin)")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Metrics: http://localhost:8000/metrics/prometheus")

if __name__ == "__main__":
    import numpy as np
    main() 