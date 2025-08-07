import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_models_info():
    """Test the models info endpoint"""
    response = requests.get(f"{BASE_URL}/models/info")
    print("Models Info:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_housing_prediction():
    """Test California Housing prediction"""
    # Sample data for California Housing prediction
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
    print("California Housing Prediction:")
    print(f"Input: {housing_data}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_iris_prediction():
    """Test Iris classification"""
    # Sample data for Iris classification
    iris_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = requests.post(f"{BASE_URL}/predict/iris", json=iris_data)
    print("Iris Classification:")
    print(f"Input: {iris_data}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print("Testing ML Model Prediction API")
    print("=" * 40)
    
    try:
        # Test health endpoint
        test_health()
        
        # Test models info
        test_models_info()
        
        # Test housing prediction
        test_housing_prediction()
        
        # Test iris prediction
        test_iris_prediction()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 