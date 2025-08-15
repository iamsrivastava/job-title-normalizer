#!/usr/bin/env python3
"""
Enhanced load testing script for the Job Title Normalization API
Monitors token input/output, provides per-second metrics, and generates Excel reports
"""

import requests
import json
import time
import csv
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import threading

# API configuration
BASE_URL = "http://localhost:8080"

class TokenMonitor:
    """Monitor token input/output and processing metrics"""
    
    def __init__(self):
        self.input_tokens = defaultdict(int)  # timestamp -> token count
        self.output_tokens = defaultdict(int)  # timestamp -> token count
        self.processing_times = defaultdict(list)  # timestamp -> list of processing times
        self.request_counts = defaultdict(int)  # timestamp -> request count
        self.cache_hits = defaultdict(int)  # timestamp -> cache hit count
        self.cache_misses = defaultdict(int)  # timestamp -> cache miss count
        self.errors = defaultdict(int)  # timestamp -> error count
        self.lock = threading.Lock()
        
        # Per-second tracking
        self.secondly_metrics = defaultdict(lambda: {
            'input_tokens': 0,
            'output_tokens': 0,
            'requests': 0,
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        })
    
    def record_request(self, timestamp, input_tokens, output_tokens, processing_time, cached, success):
        """Record metrics for a single request"""
        with self.lock:
            # Round timestamp to nearest second for aggregation
            second_key = int(timestamp)
            
            self.secondly_metrics[second_key]['input_tokens'] += input_tokens
            self.secondly_metrics[second_key]['output_tokens'] += output_tokens
            self.secondly_metrics[second_key]['requests'] += 1
            self.secondly_metrics[second_key]['processing_times'].append(processing_time)
            
            if cached:
                self.secondly_metrics[second_key]['cache_hits'] += 1
            else:
                self.secondly_metrics[second_key]['cache_misses'] += 1
                
            if not success:
                self.secondly_metrics[second_key]['errors'] += 1
    
    def get_current_metrics(self):
        """Get current second metrics"""
        current_second = int(time.time())
        return self.secondly_metrics.get(current_second, {
            'input_tokens': 0,
            'output_tokens': 0,
            'requests': 0,
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        })
    
    def get_summary_stats(self):
        """Get summary statistics across all recorded data"""
        all_processing_times = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_requests = 0
        total_cache_hits = 0
        total_cache_misses = 0
        total_errors = 0
        
        for metrics in self.secondly_metrics.values():
            all_processing_times.extend(metrics['processing_times'])
            total_input_tokens += metrics['input_tokens']
            total_output_tokens += metrics['output_tokens']
            total_requests += metrics['requests']
            total_cache_hits += metrics['cache_hits']
            total_cache_misses += metrics['cache_misses']
            total_errors += metrics['errors']
        
        return {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_requests': total_requests,
            'total_cache_hits': total_cache_hits,
            'total_cache_misses': total_cache_misses,
            'total_errors': total_errors,
            'processing_times': all_processing_times,
            'avg_input_tokens_per_request': total_input_tokens / total_requests if total_requests > 0 else 0,
            'avg_output_tokens_per_request': total_output_tokens / total_requests if total_requests > 0 else 0,
            'cache_hit_rate': total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0
        }
    
    def export_to_excel(self, filename, test_name, additional_data=None):
        """Export metrics to Excel with multiple sheets"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Per-second metrics
            secondly_data = []
            for second, metrics in sorted(self.secondly_metrics.items()):
                secondly_data.append({
                    'Timestamp': datetime.fromtimestamp(second).strftime('%Y-%m-%d %H:%M:%S'),
                    'Unix_Timestamp': second,
                    'Input_Tokens': metrics['input_tokens'],
                    'Output_Tokens': metrics['output_tokens'],
                    'Requests': metrics['requests'],
                    'Avg_Processing_Time_ms': statistics.mean(metrics['processing_times']) if metrics['processing_times'] else 0,
                    'Min_Processing_Time_ms': min(metrics['processing_times']) if metrics['processing_times'] else 0,
                    'Max_Processing_Time_ms': max(metrics['processing_times']) if metrics['processing_times'] else 0,
                    'Cache_Hits': metrics['cache_hits'],
                    'Cache_Misses': metrics['cache_misses'],
                    'Errors': metrics['errors'],
                    'Tokens_per_Second': metrics['input_tokens'] + metrics['output_tokens'],
                    'Requests_per_Second': metrics['requests']
                })
            
            df_secondly = pd.DataFrame(secondly_data)
            df_secondly.to_excel(writer, sheet_name='Per_Second_Metrics', index=False)
            
            # Sheet 2: Summary statistics
            summary_stats = self.get_summary_stats()
            summary_data = {
                'Metric': [
                    'Test Name',
                    'Total Input Tokens',
                    'Total Output Tokens',
                    'Total Requests',
                    'Total Cache Hits',
                    'Total Cache Misses',
                    'Total Errors',
                    'Average Input Tokens per Request',
                    'Average Output Tokens per Request',
                    'Cache Hit Rate (%)',
                    'Average Processing Time (ms)',
                    'Min Processing Time (ms)',
                    'Max Processing Time (ms)',
                    'Standard Deviation (ms)',
                    '95th Percentile (ms)',
                    '99th Percentile (ms)'
                ],
                'Value': [
                    test_name,
                    summary_stats['total_input_tokens'],
                    summary_stats['total_output_tokens'],
                    summary_stats['total_requests'],
                    summary_stats['total_cache_hits'],
                    summary_stats['total_cache_misses'],
                    summary_stats['total_errors'],
                    f"{summary_stats['avg_input_tokens_per_request']:.2f}",
                    f"{summary_stats['avg_output_tokens_per_request']:.2f}",
                    f"{summary_stats['cache_hit_rate'] * 100:.2f}",
                    f"{statistics.mean(summary_stats['processing_times']):.2f}" if summary_stats['processing_times'] else "N/A",
                    f"{min(summary_stats['processing_times']):.2f}" if summary_stats['processing_times'] else "N/A",
                    f"{max(summary_stats['processing_times']):.2f}" if summary_stats['processing_times'] else "N/A",
                    f"{statistics.stdev(summary_stats['processing_times']):.2f}" if len(summary_stats['processing_times']) > 1 else "N/A",
                    f"{sorted(summary_stats['processing_times'])[int(len(summary_stats['processing_times']) * 0.95)]:.2f}" if summary_stats['processing_times'] else "N/A",
                    f"{sorted(summary_stats['processing_times'])[int(len(summary_stats['processing_times']) * 0.99)]:.2f}" if summary_stats['processing_times'] else "N/A"
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Sheet 3: Additional test data if provided
            if additional_data:
                df_additional = pd.DataFrame(additional_data)
                df_additional.to_excel(writer, sheet_name='Test_Details', index=False)
            
            # Sheet 4: Raw data for analysis
            raw_data = []
            for second, metrics in sorted(self.secondly_metrics.items()):
                for i, proc_time in enumerate(metrics['processing_times']):
                    raw_data.append({
                        'Timestamp': datetime.fromtimestamp(second).strftime('%Y-%m-%d %H:%M:%S'),
                        'Unix_Timestamp': second,
                        'Request_Index': i,
                        'Processing_Time_ms': proc_time,
                        'Input_Tokens': metrics['input_tokens'] // max(1, metrics['requests']),
                        'Output_Tokens': metrics['output_tokens'] // max(1, metrics['requests'])
                    })
            
            if raw_data:
                df_raw = pd.DataFrame(raw_data)
                df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        print(f"Excel report exported to: {filename}")

class LoadTester:
    def __init__(self, csv_file='test-data.csv'):
        self.csv_file = csv_file
        self.test_titles = self.load_test_data()
        self.results = []
        self.token_monitor = TokenMonitor()
        self.monitoring_active = False
        
    def load_test_data(self):
        """Load test data from CSV file"""
        test_titles = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
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
            print(f"CSV file {self.csv_file} not found, using default test titles")
            return [
                "Senior Software Engineer", "Data Scientist", "Product Manager",
                "DevOps Engineer", "Machine Learning Engineer", "Software Engineer",
                "Project Manager", "Business Analyst", "UX Designer", "QA Engineer"
            ]
    
    def estimate_tokens(self, text):
        """Estimate token count for text (rough approximation)"""
        # Simple token estimation: split by whitespace and punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return len(tokens)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Token monitoring started...")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        print("Token monitoring stopped.")
    
    def _monitor_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            current_metrics = self.token_monitor.get_current_metrics()
            if current_metrics['requests'] > 0:
                print(f"\r[LIVE] Tokens: {current_metrics['input_tokens']}→{current_metrics['output_tokens']} | "
                      f"Req/s: {current_metrics['requests']} | "
                      f"Cache: {current_metrics['cache_hits']}/{current_metrics['cache_hits'] + current_metrics['cache_misses']} | "
                      f"Avg Time: {statistics.mean(current_metrics['processing_times']):.1f}ms", end='', flush=True)
            time.sleep(1)
    
    def single_request(self, title, use_cache=True):
        """Make a single API request with token monitoring"""
        try:
            payload = {"title": title, "use_cache": use_cache}
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/normalize", json=payload)
            end_time = time.time()
            
            # Estimate tokens
            input_tokens = self.estimate_tokens(title)
            output_tokens = 0
            
            if response.status_code == 200:
                result = response.json()
                output_tokens = self.estimate_tokens(result['normalized_title'])
                
                # Record metrics
                processing_time = (end_time - start_time) * 1000
                self.token_monitor.record_request(
                    start_time, input_tokens, output_tokens, 
                    processing_time, result.get('cached', False), True
                )
                
                return {
                    'success': True,
                    'title': title,
                    'normalized': result['normalized_title'],
                    'processing_time': processing_time,
                    'api_time': result.get('processing_time_ms', 0),
                    'cached': result.get('cached', False),
                    'status_code': response.status_code,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
            else:
                # Record failed request
                processing_time = (end_time - start_time) * 1000
                self.token_monitor.record_request(
                    start_time, input_tokens, output_tokens, 
                    processing_time, False, False
                )
                
                return {
                    'success': False,
                    'title': title,
                    'error': response.text,
                    'status_code': response.status_code,
                    'processing_time': processing_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
        except Exception as e:
            return {
                'success': False,
                'title': title,
                'error': str(e),
                'status_code': 0,
                'processing_time': 0,
                'input_tokens': 0,
                'output_tokens': 0
            }
    
    def batch_request(self, titles, use_cache=True):
        """Make a batch API request with token monitoring"""
        try:
            payload = {"titles": titles, "use_cache": use_cache}
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/normalize/batch", json=payload)
            end_time = time.time()
            
            # Estimate total tokens
            total_input_tokens = sum(self.estimate_tokens(title) for title in titles)
            total_output_tokens = 0
            
            if response.status_code == 200:
                result = response.json()
                # Estimate output tokens from results
                for item in result.get('results', []):
                    total_output_tokens += self.estimate_tokens(item.get('normalized_title', ''))
                
                # Record metrics for batch
                processing_time = (end_time - start_time) * 1000
                self.token_monitor.record_request(
                    start_time, total_input_tokens, total_output_tokens, 
                    processing_time, False, True  # Batch requests are not cached individually
                )
                
                return {
                    'success': True,
                    'batch_size': len(titles),
                    'total_time': processing_time,
                    'api_total_time': result.get('total_time_ms', 0),
                    'api_avg_time': result.get('avg_time_ms', 0),
                    'status_code': response.status_code,
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens
                }
            else:
                # Record failed batch
                processing_time = (end_time - start_time) * 1000
                self.token_monitor.record_request(
                    start_time, total_input_tokens, total_output_tokens, 
                    processing_time, False, False
                )
                
                return {
                    'success': False,
                    'batch_size': len(titles),
                    'error': response.text,
                    'status_code': response.status_code,
                    'total_time': processing_time,
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens
                }
        except Exception as e:
            return {
                'success': False,
                'batch_size': len(titles),
                'error': str(e),
                'status_code': 0,
                'total_time': 0,
                'input_tokens': 0,
                'output_tokens': 0
            }
    
    def concurrent_load_test(self, num_requests, num_threads, use_cache=True):
        """Run concurrent load test with token monitoring"""
        print(f"\n=== Concurrent Load Test ===")
        print(f"Requests: {num_requests}, Threads: {num_threads}, Cache: {use_cache}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Select random titles for the test
        test_titles = random.sample(self.test_titles, min(num_requests, len(self.test_titles)))
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all requests
            future_to_title = {
                executor.submit(self.single_request, title, use_cache): title 
                for title in test_titles
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_title):
                result = future.result()
                results.append(result)
                
                # Progress indicator
                if len(results) % 10 == 0:
                    print(f"\n  Completed {len(results)}/{num_requests} requests...")
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            processing_times = [r['processing_time'] for r in successful_requests]
            api_times = [r['api_time'] for r in successful_requests if 'api_time' in r]
            input_tokens = [r['input_tokens'] for r in successful_requests]
            output_tokens = [r['output_tokens'] for r in successful_requests]
            
            print(f"\nResults:")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Successful requests: {len(successful_requests)}")
            print(f"  Failed requests: {len(failed_requests)}")
            print(f"  Success rate: {len(successful_requests)/len(results)*100:.1f}%")
            print(f"  Requests per second: {len(results)/(total_time/1000):.2f}")
            print(f"  Total input tokens: {sum(input_tokens)}")
            print(f"  Total output tokens: {sum(output_tokens)}")
            print(f"  Average input tokens per request: {statistics.mean(input_tokens):.1f}")
            print(f"  Average output tokens per request: {statistics.mean(output_tokens):.1f}")
            
            if processing_times:
                print(f"  Processing times (ms):")
                print(f"    Min: {min(processing_times):.2f}")
                print(f"    Max: {max(processing_times):.2f}")
                print(f"    Mean: {statistics.mean(processing_times):.2f}")
                print(f"    Median: {statistics.median(processing_times):.2f}")
                print(f"    Std Dev: {statistics.stdev(processing_times):.2f}")
        
        if failed_requests:
            print(f"\nFailed requests:")
            for req in failed_requests[:5]:  # Show first 5 failures
                print(f"  {req['title'][:50]}... - {req['error']}")
        
        return results
    
    def batch_load_test(self, num_batches, batch_size, use_cache=True):
        """Run batch load test with token monitoring"""
        print(f"\n=== Batch Load Test ===")
        print(f"Batches: {num_batches}, Batch size: {batch_size}, Cache: {use_cache}")
        
        # Start monitoring
        self.start_monitoring()
        
        results = []
        total_titles = 0
        
        for i in range(num_batches):
            # Select random titles for this batch
            batch_titles = random.sample(self.test_titles, min(batch_size, len(self.test_titles)))
            total_titles += len(batch_titles)
            
            result = self.batch_request(batch_titles, use_cache)
            results.append(result)
            
            if result['success']:
                print(f"  Batch {i+1}: {len(batch_titles)} titles in {result['total_time']:.2f}ms")
                print(f"    Tokens: {result['input_tokens']}→{result['output_tokens']}")
            else:
                print(f"  Batch {i+1}: Failed - {result['error']}")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Analyze batch results
        successful_batches = [r for r in results if r['success']]
        failed_batches = [r for r in results if not r['success']]
        
        if successful_batches:
            total_times = [r['total_time'] for r in successful_batches]
            api_times = [r['api_total_time'] for r in successful_batches if 'api_total_time' in r]
            input_tokens = [r['input_tokens'] for r in successful_batches]
            output_tokens = [r['output_tokens'] for r in successful_batches]
            
            print(f"\nBatch Results:")
            print(f"  Total titles processed: {total_titles}")
            print(f"  Successful batches: {len(successful_batches)}")
            print(f"  Failed batches: {len(failed_batches)}")
            print(f"  Success rate: {len(successful_batches)/len(results)*100:.1f}%")
            print(f"  Total input tokens: {sum(input_tokens)}")
            print(f"  Total output tokens: {sum(output_tokens)}")
            
            if total_times:
                print(f"  Total processing times (ms):")
                print(f"    Min: {min(total_times):.2f}")
                print(f"    Max: {max(total_times):.2f}")
                print(f"    Mean: {statistics.mean(total_times):.2f}")
                print(f"    Median: {statistics.median(total_times):.2f}")
        
        return results
    
    def stress_test(self, duration_seconds=60, requests_per_second=10, use_cache=True):
        """Run stress test with token monitoring"""
        print(f"\n=== Stress Test ===")
        print(f"Duration: {duration_seconds}s, Rate: {requests_per_second} req/s, Cache: {use_cache}")
        
        # Start monitoring
        self.start_monitoring()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        results = []
        request_count = 0
        
        with ThreadPoolExecutor(max_workers=requests_per_second * 2) as executor:
            while time.time() < end_time:
                # Submit batch of requests
                batch_size = min(requests_per_second, 10)  # Submit in small batches
                batch_titles = random.sample(self.test_titles, batch_size)
                
                futures = [
                    executor.submit(self.single_request, title, use_cache)
                    for title in batch_titles
                ]
                
                # Wait for batch to complete
                batch_results = [future.result() for future in futures]
                results.extend(batch_results)
                request_count += len(batch_results)
                
                # Progress indicator
                elapsed = time.time() - start_time
                if elapsed > 0:
                    current_rate = request_count / elapsed
                    print(f"\n  Elapsed: {elapsed:.1f}s, Requests: {request_count}, Rate: {current_rate:.1f} req/s")
                
                # Small delay to maintain rate
                time.sleep(1.0 / requests_per_second)
        
        total_time = (time.time() - start_time) * 1000
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Analyze stress test results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"\nStress Test Results:")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful_requests)}")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Success rate: {len(successful_requests)/len(results)*100:.1f}%")
        print(f"  Average rate: {len(results)/(total_time/1000):.2f} req/s")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive load test suite with Excel export"""
        print("=== Job Title Normalization API Load Test Suite ===")
        print(f"Test data: {len(self.test_titles)} titles from {self.csv_file}")
        print(f"API endpoint: {BASE_URL}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_start_time = time.time()
        
        # Test 1: Light load
        light_results = self.concurrent_load_test(50, 5, use_cache=True)
        
        # Test 2: Medium load
        medium_results = self.concurrent_load_test(200, 10, use_cache=True)
        
        # Test 3: Heavy load
        heavy_results = self.concurrent_load_test(500, 20, use_cache=True)
        
        # Test 4: Batch processing
        batch_results = self.batch_load_test(10, 50, use_cache=True)
        
        # Test 5: Stress test
        stress_results = self.stress_test(duration_seconds=30, requests_per_second=20, use_cache=True)
        
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        
        print("\n=== Load Test Suite Complete ===")
        print(f"Total test time: {total_test_time:.2f} seconds")
        
        # Generate Excel report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"load_test_report_{timestamp}.xlsx"
        
        # Prepare additional test data for Excel
        test_details = [
            {'Test_Phase': 'Light Load', 'Requests': 50, 'Threads': 5, 'Results': len(light_results)},
            {'Test_Phase': 'Medium Load', 'Requests': 200, 'Threads': 10, 'Results': len(medium_results)},
            {'Test_Phase': 'Heavy Load', 'Requests': 500, 'Threads': 20, 'Results': len(heavy_results)},
            {'Test_Phase': 'Batch Processing', 'Batches': 10, 'Batch_Size': 50, 'Results': len(batch_results)},
            {'Test_Phase': 'Stress Test', 'Duration_s': 30, 'Rate_req_s': 20, 'Results': len(stress_results)},
            {'Test_Phase': 'Total Test Time', 'Duration_s': total_test_time, 'Total_Requests': len(light_results) + len(medium_results) + len(heavy_results) + len(stress_results)}
        ]
        
        self.token_monitor.export_to_excel(excel_filename, "Comprehensive Load Test", test_details)
        
        # Print summary
        summary_stats = self.token_monitor.get_summary_stats()
        print(f"\n=== Token Summary ===")
        print(f"Total input tokens: {summary_stats['total_input_tokens']:,}")
        print(f"Total output tokens: {summary_stats['total_output_tokens']:,}")
        print(f"Total requests: {summary_stats['total_requests']:,}")
        print(f"Cache hit rate: {summary_stats['cache_hit_rate']*100:.1f}%")
        print(f"Average processing time: {statistics.mean(summary_stats['processing_times']):.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='Enhanced load test the Job Title Normalization API with token monitoring')
    parser.add_argument('--csv', default='test-data.csv', help='CSV file with test data')
    parser.add_argument('--concurrent', action='store_true', help='Run concurrent load test')
    parser.add_argument('--batch', action='store_true', help='Run batch load test')
    parser.add_argument('--stress', action='store_true', help='Run stress test')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests for concurrent test')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads for concurrent test')
    parser.add_argument('--batches', type=int, default=5, help='Number of batches for batch test')
    parser.add_argument('--batch-size', type=int, default=50, help='Size of each batch')
    parser.add_argument('--duration', type=int, default=60, help='Duration for stress test (seconds)')
    parser.add_argument('--rate', type=int, default=10, help='Requests per second for stress test')
    parser.add_argument('--excel', action='store_true', help='Generate Excel report for any test')
    
    args = parser.parse_args()
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"Error: API is not responding correctly (status: {response.status_code})")
            return
    except Exception as e:
        print(f"Error: Cannot connect to API at {BASE_URL}: {e}")
        return
    
    tester = LoadTester(args.csv)
    
    try:
        if args.concurrent:
            results = tester.concurrent_load_test(args.requests, args.threads)
            if args.excel:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_filename = f"concurrent_test_report_{timestamp}.xlsx"
                tester.token_monitor.export_to_excel(excel_filename, f"Concurrent Test ({args.requests} req, {args.threads} threads)")
        elif args.batch:
            results = tester.batch_load_test(args.batches, args.batch_size)
            if args.excel:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_filename = f"batch_test_report_{timestamp}.xlsx"
                tester.token_monitor.export_to_excel(excel_filename, f"Batch Test ({args.batches} batches, {args.batch_size} size)")
        elif args.stress:
            results = tester.stress_test(args.duration, args.rate)
            if args.excel:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_filename = f"stress_test_report_{timestamp}.xlsx"
                tester.token_monitor.export_to_excel(excel_filename, f"Stress Test ({args.duration}s, {args.rate} req/s)")
        elif args.comprehensive:
            tester.run_comprehensive_test()
        else:
            # Default: run comprehensive test
            tester.run_comprehensive_test()
    finally:
        # Ensure monitoring is stopped
        tester.stop_monitoring()

if __name__ == "__main__":
    main()
