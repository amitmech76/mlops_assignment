#!/usr/bin/env python3
"""
Simple monitoring dashboard for ML Prediction API
"""

import requests
import json
import time
import os
from datetime import datetime
import sqlite3
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
LOG_DB_PATH = "logs/predictions.db"

def get_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None

def get_metrics():
    """Get metrics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None

def get_recent_logs():
    """Get recent logs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/logs/recent?limit=10", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None

def get_direct_db_stats():
    """Get stats directly from SQLite database"""
    try:
        if not Path(LOG_DB_PATH).exists():
            return None
            
        with sqlite3.connect(LOG_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Get recent predictions
            cursor.execute("""
                SELECT timestamp, model_name, prediction_value, status, processing_time_ms
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent = cursor.fetchall()
            
            # Get model stats
            cursor.execute("""
                SELECT model_name, COUNT(*), AVG(processing_time_ms)
                FROM predictions 
                GROUP BY model_name
            """)
            model_stats = cursor.fetchall()
            
            return {
                "total_predictions": total_predictions,
                "recent_predictions": [
                    {
                        "timestamp": row[0],
                        "model": row[1],
                        "prediction": row[2],
                        "status": row[3],
                        "processing_time_ms": row[4]
                    }
                    for row in recent
                ],
                "model_stats": [
                    {
                        "model": row[0],
                        "count": row[1],
                        "avg_processing_time_ms": round(row[2] or 0, 2)
                    }
                    for row in model_stats
                ]
            }
    except Exception as e:
        return None

def display_dashboard():
    """Display monitoring dashboard"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print("           ML Prediction API - Monitoring Dashboard")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # API Health
    print("API Health Check:")
    health = get_api_health()
    if health:
        print(f"   PASS: Status: {health.get('status', 'unknown')}")
        print(f"   Models Loaded: {health.get('models_loaded', {})}")
    else:
        print("   FAIL: API not responding")
    print()
    
    # Metrics
    print("Metrics:")
    metrics = get_metrics()
    if metrics:
        overall = metrics.get('overall_stats', {})
        print(f"   Total Predictions: {overall.get('total_predictions', 0)}")
        print(f"   Successful: {overall.get('successful_predictions', 0)}")
        print(f"   Error Rate: {overall.get('error_rate_percent', 0)}%")
        print(f"   Avg Processing Time: {overall.get('avg_processing_time_ms', 0)}ms")
        
        # Model-specific stats
        model_stats = metrics.get('model_stats', {})
        for model, stats in model_stats.items():
            print(f"   {model.title()}: {stats.get('total_predictions', 0)} predictions")
    else:
        print("   FAIL: Unable to fetch metrics")
    print()
    
    # Recent Logs
    print("Recent Predictions:")
    logs = get_recent_logs()
    if logs and logs.get('recent_predictions'):
        for pred in logs['recent_predictions'][:3]:  # Show last 3
            status_text = "PASS" if pred['status'] == 'success' else "FAIL"
            print(f"   {status_text} {pred['model_name']}: {pred['prediction']} ({pred['processing_time_ms']:.1f}ms)")
    else:
        print("   No recent predictions")
    print()
    
    # Direct DB Stats (fallback)
    print("Database Stats:")
    db_stats = get_direct_db_stats()
    if db_stats:
        print(f"   Total Records: {db_stats['total_predictions']}")
        for stat in db_stats['model_stats']:
            print(f"   {stat['model']}: {stat['count']} predictions, {stat['avg_processing_time_ms']}ms avg")
    else:
        print("   FAIL: Database not accessible")
    print()
    
    print("=" * 60)
    print("Press Ctrl+C to exit")
    print("=" * 60)

def main():
    """Main monitoring loop"""
    print("Starting ML Prediction API Monitor...")
    print("Press Ctrl+C to exit")
    print()
    
    try:
        while True:
            display_dashboard()
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
