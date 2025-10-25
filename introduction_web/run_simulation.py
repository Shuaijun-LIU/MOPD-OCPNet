#!/usr/bin/env python3
"""
Web interface for running underwater quadrotor simulation
This script captures the simulation output and provides it to the web interface
"""

import sys
import os
import json
import subprocess
import time
import threading
from pathlib import Path

# Add the simu directory to Python path
simu_path = Path(__file__).parent.parent / "simu"
sys.path.insert(0, str(simu_path))

def run_simulation_with_output():
    """Run the underwater quadrotor simulation and capture output"""
    try:
        # Import the simulation module
        from underwater_quadrotor_simulator import UnderwaterQuadrotorSimulator, RealTimeVisualizer
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create simulator
        simulator = UnderwaterQuadrotorSimulator()
        
        # Run simulation for a shorter duration for web demo
        print("Starting underwater quadrotor simulation...")
        print("Mission Mode: Patrol")
        print("Target: Square patrol at 2 meters underwater depth")
        print("-" * 50)
        
        # Run simulation
        start_time = time.time()
        simulation_duration = 30.0  # 30 seconds for web demo
        
        while (time.time() - start_time) < simulation_duration:
            simulator.update_simulation()
            time.sleep(0.02)  # 50Hz
            
            # Print status every 5 seconds
            if int(time.time() - start_time) % 5 == 0 and int(time.time() - start_time) > 0:
                elapsed = time.time() - start_time
                print(f"Simulation running... {elapsed:.1f}s")
                print(f"Position: ({simulator.state.x:.2f}, {simulator.state.y:.2f}, {simulator.state.z:.2f}) m")
                print(f"Velocity: ({simulator.state.vx:.2f}, {simulator.state.vy:.2f}, {simulator.state.vz:.2f}) m/s")
                print(f"Attitude: ({simulator.state.roll*180/3.14159:.1f}°, {simulator.state.pitch*180/3.14159:.1f}°, {simulator.state.yaw*180/3.14159:.1f}°)")
                print(f"Motors: {simulator.state.motor1:.0f}, {simulator.state.motor2:.0f}, {simulator.state.motor3:.0f}, {simulator.state.motor4:.0f} RPM")
                print("-" * 30)
        
        print("Simulation completed!")
        
        # Generate summary data
        summary = {
            "duration": simulation_duration,
            "total_points": len(simulator.time_history),
            "final_position": {
                "x": simulator.state.x,
                "y": simulator.state.y,
                "z": simulator.state.z
            },
            "final_velocity": {
                "x": simulator.state.vx,
                "y": simulator.state.vy,
                "z": simulator.state.vz
            },
            "final_attitude": {
                "roll": simulator.state.roll,
                "pitch": simulator.state.pitch,
                "yaw": simulator.state.yaw
            },
            "final_motors": {
                "m1": simulator.state.motor1,
                "m2": simulator.state.motor2,
                "m3": simulator.state.motor3,
                "m4": simulator.state.motor4
            },
            "data_points": {
                "time": simulator.time_history[-100:],  # Last 100 points
                "position_x": [p[0] for p in simulator.position_history[-100:]],
                "position_y": [p[1] for p in simulator.position_history[-100:]],
                "position_z": [p[2] for p in simulator.position_history[-100:]],
                "attitude_roll": [a[0] for a in simulator.attitude_history[-100:]],
                "attitude_pitch": [a[1] for a in simulator.attitude_history[-100:]],
                "attitude_yaw": [a[2] for a in simulator.attitude_history[-100:]],
                "motor1": [m[0] for m in simulator.motor_history[-100:]],
                "motor2": [m[1] for m in simulator.motor_history[-100:]],
                "motor3": [m[2] for m in simulator.motor_history[-100:]],
                "motor4": [m[3] for m in simulator.motor_history[-100:]]
            }
        }
        
        return summary
        
    except ImportError as e:
        print(f"Error importing simulation module: {e}")
        print("Make sure the underwater_quadrotor_simulator.py file is in the simu directory")
        return None
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None

def main():
    """Main function for web interface"""
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Web interface mode
        summary = run_simulation_with_output()
        if summary:
            print("SIMULATION_COMPLETE")
            print(json.dumps(summary))
        else:
            print("SIMULATION_ERROR")
    else:
        # Command line mode
        summary = run_simulation_with_output()
        if summary:
            print("\nSimulation Summary:")
            print(f"Duration: {summary['duration']} seconds")
            print(f"Data points: {summary['total_points']}")
            print(f"Final position: ({summary['final_position']['x']:.2f}, {summary['final_position']['y']:.2f}, {summary['final_position']['z']:.2f}) m")

if __name__ == "__main__":
    main()
