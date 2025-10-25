#!/usr/bin/env python3
"""
CGI script to run underwater quadrotor simulation from web interface
"""

import cgi
import cgitb
import json
import sys
import os
import subprocess
from pathlib import Path

# Enable error reporting
cgitb.enable()

# Add the parent directory to Python path to access run_simulation.py
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

def run_simulation():
    """Run the underwater quadrotor simulation"""
    try:
        # Import and run the simulation
        from run_simulation import run_simulation_with_output
        
        # Run the simulation
        result = run_simulation_with_output()
        
        if result:
            return {
                "success": True,
                "data": result
            }
        else:
            return {
                "success": False,
                "error": "Simulation failed to complete"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main CGI function"""
    # Set content type
    print("Content-Type: application/json")
    print()
    
    try:
        # Get the request data
        if os.environ.get('REQUEST_METHOD') == 'POST':
            # Read POST data
            content_length = int(os.environ.get('CONTENT_LENGTH', 0))
            if content_length > 0:
                post_data = sys.stdin.read(content_length)
                request_data = json.loads(post_data)
                
                if request_data.get('action') == 'run_simulation':
                    result = run_simulation()
                    print(json.dumps(result))
                else:
                    print(json.dumps({
                        "success": False,
                        "error": "Invalid action"
                    }))
            else:
                print(json.dumps({
                    "success": False,
                    "error": "No data received"
                }))
        else:
            print(json.dumps({
                "success": False,
                "error": "Only POST requests are supported"
            }))
            
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"CGI Error: {str(e)}"
        }))

if __name__ == "__main__":
    main()
