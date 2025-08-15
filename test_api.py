#!/usr/bin/env python3
"""
Enhanced test script for the ONNX-based Job Title Normalization API
Uses test data from test-data.csv and includes comprehensive testing
"""

import requests
import json
import time
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# API configuration
BASE_URL = "http://localhost:8080"

def load_test_data():
    """Load test data from CSV file"""
    test_titles = []
    try:
        with open('test-data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('title') and len(row['title'].strip()) > 0:
                    # Clean up the title (remove HTML artifacts and job board text)
                    title = row['title'].strip()
                    # Remove common job board artifacts
                    title = title.replace('Job in ', '').replace('Job Application for ', '')
                    title = title.replace(' | Monster.com', '').replace(' var MONS_LOG_VARS = {"JobID":', '')
                    title = title.replace('body { margin:px; overflow: visible !important; } #ejb_header { color: #; font-family: Verdana', '')
                    title = title.replace('function wrap(EL', '')
                    
                    # Only keep titles that are reasonable length and don't contain obvious artifacts
                    if len(title) > 5 and len(title) < 200 and not title.startswith('Please apply'):
                        test_titles.append(title)
        
        print(f"Loaded {len(test_titles)} test titles from CSV")
        return test_titles
    except FileNotFoundError:
        print("test-data.csv not found, using default test titles")
        return [
            "Senior Software Engineer",
            "Data Scientist", 
            "Product Manager",
            "DevOps Engineer",
            "Machine Learning Engineer"
        ]

# Load test data
TEST_TITLES = load_test_data()

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_normalization():
    """Test single title normalization"""
    print("\nTesting single title normalization...")
    try:
        payload = {
            "title": "Senior Software Engineer",
            "use_cache": True
        }
        response = requests.post(f"{BASE_URL}/normalize", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Original: {payload['title']}")
            print(f"Normalized: {result['normalized_title']}")
            print(f"Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"Cached: {result['cached']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_normalization():
    """Test batch title normalization with CSV data"""
    print("\nTesting batch title normalization...")
    try:
        # Use a subset of CSV data for testing
        test_subset = random.sample(TEST_TITLES, min(50, len(TEST_TITLES)))
        payload = {
            "titles": test_subset,
            "use_cache": True
        }
        response = requests.post(f"{BASE_URL}/normalize/batch", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total processing time: {result['total_time_ms']:.2f}ms")
            print(f"Average processing time: {result['avg_time_ms']:.2f}ms")
            print(f"Processed {len(result['results'])} titles")
            print("\nSample results:")
            for i, item in enumerate(result['results'][:5]):  # Show first 5 results
                print(f"  {i+1}. {item['title'][:50]}... -> {item['normalized_title']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_csv_data_sample():
    """Test a sample of titles from the CSV file"""
    print("\nTesting CSV data sample...")
    try:
        # Test 10 random titles from CSV
        sample_titles = random.sample(TEST_TITLES, min(10, len(TEST_TITLES)))
        
        results = []
        total_time = 0
        
        for title in sample_titles:
            start_time = time.time()
            payload = {"title": title, "use_cache": True}
            response = requests.post(f"{BASE_URL}/normalize", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                processing_time = (end_time - start_time) * 1000
                total_time += processing_time
                results.append({
                    'title': title[:60] + '...' if len(title) > 60 else title,
                    'normalized': result['normalized_title'],
                    'time_ms': processing_time,
                    'cached': result['cached']
                })
            else:
                print(f"Failed to process: {title[:50]}...")
        
        print(f"Processed {len(results)} titles in {total_time:.2f}ms")
        print("Sample results:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result['title']}")
            print(f"     -> {result['normalized']} ({result['time_ms']:.2f}ms, cached: {result['cached']})")
        
        return len(results) > 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            metrics = response.json()
            print(f"Metrics: {json.dumps(metrics, indent=2)}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_caching():
    """Test caching functionality"""
    print("\nTesting caching functionality...")
    try:
        title = "Software Engineer"
        
        # First request (should not be cached)
        print(f"First request for: {title}")
        payload = {"title": title, "use_cache": True}
        response1 = requests.post(f"{BASE_URL}/normalize", json=payload)
        result1 = response1.json()
        print(f"  Processing time: {result1['processing_time_ms']:.2f}ms")
        print(f"  Cached: {result1['cached']}")
        
        # Second request (should be cached)
        print(f"Second request for: {title}")
        response2 = requests.post(f"{BASE_URL}/normalize", json=payload)
        result2 = response2.json()
        print(f"  Processing time: {result2['processing_time_ms']:.2f}ms")
        print(f"  Cached: {result2['cached']}")
        
        # Verify caching worked
        if result1['cached'] == False and result2['cached'] == True:
            print("  Caching working correctly")
            return True
        else:
            print("  Caching not working as expected")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    try:
        # Test empty title
        payload = {"title": "", "use_cache": True}
        response = requests.post(f"{BASE_URL}/normalize", json=payload)
        print(f"Empty title - Status: {response.status_code}")
        
        # Test very long title
        long_title = "A" * 300
        payload = {"title": long_title, "use_cache": True}
        response = requests.post(f"{BASE_URL}/normalize", json=payload)
        print(f"Long title - Status: {response.status_code}")
        
        # Test invalid JSON
        response = requests.post(f"{BASE_URL}/normalize", data="invalid json")
        print(f"Invalid JSON - Status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Enhanced Job Title Normalization API Test Suite ===\n")
    print(f"Using {len(TEST_TITLES)} test titles from CSV\n")
    
    tests = [
        ("Health Check", test_health),
        ("Single Normalization", test_single_normalization),
        ("Batch Normalization", test_batch_normalization),
        ("CSV Data Sample", test_csv_data_sample),
        ("Metrics", test_metrics),
        ("Caching", test_caching),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        print("-" * 50)
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
