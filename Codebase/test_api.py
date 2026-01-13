"""
Simple script to test the API endpoints
Run this after starting the API server
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*60)
    print("Testing /metrics endpoint...")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_features():
    """Test features endpoint"""
    print("\n" + "="*60)
    print("Testing /features endpoint...")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/features")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_prediction():
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict endpoint...")
    print("="*60)

    # Sample customer data
    payload = {
        "customer_age": 45,
        "dependent_count": 2,
        "months_on_book": 36,
        "total_relationship_count": 3,
        "months_inactive_12_mon": 1,
        "contacts_count_12_mon": 2,
        "credit_limit": 10000.0,
        "total_revolving_bal": 1500.0,
        "avg_open_to_buy": 8500.0,
        "total_amt_chng_q4_q1": 0.8,
        "total_trans_amt": 5000.0,
        "total_trans_ct": 50,
        "total_ct_chng_q4_q1": 0.7,
        "avg_utilization_ratio": 0.3
    }

    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("STARTING API TESTS")
    print("🧪 "*20)

    print("\nMake sure the API server is running:")
    print("  uvicorn api:app --reload --port 8000\n")

    results = {
        "Health Check": test_health(),
        "Metrics": test_metrics(),
        "Features": test_features(),
        "Prediction": test_prediction()
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the API server and model files.")

    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
