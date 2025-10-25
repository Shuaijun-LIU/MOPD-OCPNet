#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-DOF Underwater Quadrotor UAV Simulator
Includes physics model, ocean environment effects, control system and real-time visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import time
import threading
from dataclasses import dataclass
from typing import Tuple, List
import random

# Set font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

@dataclass
class QuadrotorState:
    """Quadrotor state"""
    # Position (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Attitude (rad)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Linear velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Angular velocity (rad/s)
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    
    # Motor speeds (RPM)
    motor1: float = 0.0  # Front Left
    motor2: float = 0.0  # Front Right
    motor3: float = 0.0  # Back Left
    motor4: float = 0.0  # Back Right

@dataclass
class OceanEnvironment:
    """Ocean environment parameters"""
    # Current parameters
    current_speed: float = 0.5  # m/s
    current_direction: float = 0.0  # rad
    current_variation: float = 0.2  # Current variation amplitude
    
    # Depth-related parameters
    depth_factor: float = 1.0  # Depth influence factor
    
    # Turbulence parameters
    turbulence_intensity: float = 0.1
    turbulence_frequency: float = 0.5

class QuadrotorPhysics:
    """Quadrotor physics model"""
    
    def __init__(self):
        # Physical parameters
        self.mass = 2.0  # kg
        self.gravity = 9.81  # m/s²
        self.water_density = 1025.0  # kg/m³
        
        # Geometric parameters
        self.arm_length = 0.25  # m (rotor arm length)
        self.rotor_radius = 0.1  # m
        
        # Aerodynamic parameters
        self.thrust_coefficient = 1.0e-6  # Thrust coefficient
        self.drag_coefficient = 0.1  # Drag coefficient
        self.moment_coefficient = 1.0e-7  # Moment coefficient
        
        # Inertia matrix (simplified)
        self.Ixx = 0.1  # kg·m²
        self.Iyy = 0.1
        self.Izz = 0.2
        
        # Added mass (underwater)
        self.added_mass_factor = 0.5
        
    def calculate_thrust(self, motor_speed: float) -> float:
        """Calculate single rotor thrust"""
        return self.thrust_coefficient * motor_speed**2
    
    def calculate_drag(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate fluid drag force"""
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)
        
        drag_force = -0.5 * self.water_density * self.drag_coefficient * speed * velocity
        return drag_force
    
    def calculate_ocean_current(self, position: np.ndarray, time: float, ocean: OceanEnvironment) -> np.ndarray:
        """Calculate ocean current effects"""
        x, y, z = position
        
        # Base current
        base_current = np.array([
            ocean.current_speed * np.cos(ocean.current_direction),
            ocean.current_speed * np.sin(ocean.current_direction),
            0.0
        ])
        
        # Depth influence
        depth_factor = np.exp(-z / 10.0)  # Decay with depth
        
        # Turbulence effects
        turbulence = np.array([
            ocean.turbulence_intensity * np.sin(time * ocean.turbulence_frequency + x),
            ocean.turbulence_intensity * np.cos(time * ocean.turbulence_frequency + y),
            ocean.turbulence_intensity * 0.1 * np.sin(time * ocean.turbulence_frequency * 2)
        ])
        
        # Random variation
        random_variation = np.array([
            random.gauss(0, ocean.current_variation),
            random.gauss(0, ocean.current_variation),
            random.gauss(0, ocean.current_variation * 0.5)
        ])
        
        current = (base_current + turbulence + random_variation) * depth_factor
        return current
    
    def update_dynamics(self, state: QuadrotorState, control_inputs: np.ndarray, 
                       ocean: OceanEnvironment, dt: float) -> QuadrotorState:
        """Update dynamics equations"""
        
        # Extract control inputs [total thrust, roll torque, pitch torque, yaw torque]
        total_thrust, roll_torque, pitch_torque, yaw_torque = control_inputs
        
        # Calculate individual rotor thrusts
        thrusts = np.array([
            self.calculate_thrust(state.motor1),
            self.calculate_thrust(state.motor2),
            self.calculate_thrust(state.motor3),
            self.calculate_thrust(state.motor4)
        ])
        
        # Total thrust
        total_thrust_actual = np.sum(thrusts)
        
        # Position update
        position = np.array([state.x, state.y, state.z])
        velocity = np.array([state.vx, state.vy, state.vz])
        
        # Ocean current effects
        current_velocity = self.calculate_ocean_current(position, time.time(), ocean)
        relative_velocity = velocity - current_velocity
        
        # Drag force calculation
        drag_force = self.calculate_drag(relative_velocity)
        
        # Gravity
        gravity_force = np.array([0, 0, -self.mass * self.gravity])
        
        # Buoyancy (simplified)
        buoyancy_force = np.array([0, 0, self.mass * self.gravity * 0.1])
        
        # Thrust in body frame
        thrust_body = np.array([0, 0, total_thrust_actual])
        
        # Rotation matrix (ZYX order)
        cos_r, sin_r = np.cos(state.roll), np.sin(state.roll)
        cos_p, sin_p = np.cos(state.pitch), np.sin(state.pitch)
        cos_y, sin_y = np.cos(state.yaw), np.sin(state.yaw)
        
        R = np.array([
            [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
            [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
            [-sin_p, cos_p*sin_r, cos_p*cos_r]
        ])
        
        # Transform thrust to world frame
        thrust_world = R @ thrust_body
        
        # Total force
        total_force = thrust_world + gravity_force + buoyancy_force + drag_force
        
        # Consider added mass
        effective_mass = self.mass * (1 + self.added_mass_factor)
        
        # Acceleration
        acceleration = total_force / effective_mass
        
        # Angular velocity update
        angular_velocity = np.array([state.wx, state.wy, state.wz])
        
        # Angular acceleration
        angular_acceleration = np.array([
            roll_torque / self.Ixx,
            pitch_torque / self.Iyy,
            yaw_torque / self.Izz
        ])
        
        # Update state
        new_state = QuadrotorState()
        
        # Position and velocity
        new_state.x = state.x + state.vx * dt
        new_state.y = state.y + state.vy * dt
        new_state.z = state.z + state.vz * dt
        
        new_state.vx = state.vx + acceleration[0] * dt
        new_state.vy = state.vy + acceleration[1] * dt
        new_state.vz = state.vz + acceleration[2] * dt
        
        # Attitude and angular velocity
        new_state.roll = state.roll + state.wx * dt
        new_state.pitch = state.pitch + state.wy * dt
        new_state.yaw = state.yaw + state.wz * dt
        
        new_state.wx = state.wx + angular_acceleration[0] * dt
        new_state.wy = state.wy + angular_acceleration[1] * dt
        new_state.wz = state.wz + angular_acceleration[2] * dt
        
        # Keep motor speeds
        new_state.motor1 = state.motor1
        new_state.motor2 = state.motor2
        new_state.motor3 = state.motor3
        new_state.motor4 = state.motor4
        
        return new_state

class PIDController:
    """PID Controller"""
    
    def __init__(self, kp: float, ki: float, kd: float, max_output: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
    
    def update(self, error: float) -> float:
        """Update PID controller"""
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt < 1e-6:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Total output
        output = proportional + integral + derivative
        
        # Limit output
        output = np.clip(output, -self.max_output, self.max_output)
        
        # Update state
        self.previous_error = error
        self.previous_time = current_time
        
        return output

class FlightController:
    """Flight Controller"""
    
    def __init__(self):
        # PID controllers
        self.position_controller_x = PIDController(2.0, 0.1, 0.5, 5.0)
        self.position_controller_y = PIDController(2.0, 0.1, 0.5, 5.0)
        self.position_controller_z = PIDController(3.0, 0.2, 0.8, 10.0)
        
        self.attitude_controller_roll = PIDController(1.0, 0.05, 0.2, 2.0)
        self.attitude_controller_pitch = PIDController(1.0, 0.05, 0.2, 2.0)
        self.attitude_controller_yaw = PIDController(0.8, 0.02, 0.15, 1.0)
        
        # Target position
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -2.0  # 2 meters underwater
        
        # Target attitude
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
        # Mission status
        self.mission_mode = "hover"  # hover, patrol, emergency
        self.mission_start_time = time.time()
        
        # Physical parameters
        self.mass = 2.0  # kg
    
    def set_target(self, x: float, y: float, z: float, yaw: float = 0.0):
        """Set target position"""
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw
    
    def update_mission(self, current_time: float):
        """Update mission status"""
        elapsed_time = current_time - self.mission_start_time
        
        # Simple patrol mission
        if self.mission_mode == "patrol":
            if elapsed_time < 10:
                self.set_target(2.0, 0.0, -2.0, 0.0)
            elif elapsed_time < 20:
                self.set_target(2.0, 2.0, -2.0, np.pi/2)
            elif elapsed_time < 30:
                self.set_target(0.0, 2.0, -2.0, np.pi)
            elif elapsed_time < 40:
                self.set_target(0.0, 0.0, -2.0, 3*np.pi/2)
            else:
                self.mission_start_time = current_time  # Restart
    
    def calculate_control(self, state: QuadrotorState) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate control inputs"""
        current_time = time.time()
        self.update_mission(current_time)
        
        # Position control
        pos_error_x = self.target_x - state.x
        pos_error_y = self.target_y - state.y
        pos_error_z = self.target_z - state.z
        
        # Position controller output (desired attitude)
        desired_roll = self.position_controller_y.update(pos_error_y)
        desired_pitch = self.position_controller_x.update(pos_error_x)
        desired_yaw = self.target_yaw
        
        # Attitude control
        roll_error = desired_roll - state.roll
        pitch_error = desired_pitch - state.pitch
        yaw_error = desired_yaw - state.yaw
        
        # Angle normalization
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        roll_torque = self.attitude_controller_roll.update(roll_error)
        pitch_torque = self.attitude_controller_pitch.update(pitch_error)
        yaw_torque = self.attitude_controller_yaw.update(yaw_error)
        
        # Altitude control (total thrust)
        thrust_error = self.target_z - state.z
        base_thrust = self.mass * 9.81  # Gravity compensation
        thrust_adjustment = self.position_controller_z.update(thrust_error)
        total_thrust = base_thrust + thrust_adjustment
        
        # Control inputs
        control_inputs = np.array([total_thrust, roll_torque, pitch_torque, yaw_torque])
        
        # Calculate motor speeds
        motor_speeds = self.calculate_motor_speeds(total_thrust, roll_torque, pitch_torque, yaw_torque)
        
        return control_inputs, motor_speeds
    
    def calculate_motor_speeds(self, thrust: float, roll_torque: float, 
                              pitch_torque: float, yaw_torque: float) -> np.ndarray:
        """Calculate four motor speeds"""
        # Allocation matrix (simplified)
        # Front Left(1), Front Right(2), Back Left(3), Back Right(4)
        arm_length = 0.25
        thrust_coeff = 1.0e-6
        
        # Base thrust
        base_speed = np.sqrt(thrust / (4 * thrust_coeff))
        
        # Torque allocation
        roll_speed = roll_torque / (2 * arm_length * thrust_coeff * base_speed)
        pitch_speed = pitch_torque / (2 * arm_length * thrust_coeff * base_speed)
        yaw_speed = yaw_torque / (4 * thrust_coeff * base_speed)
        
        # Individual motor speeds
        motor1 = base_speed - roll_speed + pitch_speed - yaw_speed  # Front Left
        motor2 = base_speed + roll_speed + pitch_speed + yaw_speed  # Front Right
        motor3 = base_speed - roll_speed - pitch_speed + yaw_speed  # Back Left
        motor4 = base_speed + roll_speed - pitch_speed - yaw_speed  # Back Right
        
        # Limit speed range
        motor_speeds = np.array([motor1, motor2, motor3, motor4])
        motor_speeds = np.clip(motor_speeds, 0, 2000)  # Max 2000 RPM
        
        return motor_speeds

class UnderwaterQuadrotorSimulator:
    """Main Underwater Quadrotor Simulator Class"""
    
    def __init__(self):
        # Initialize components
        self.physics = QuadrotorPhysics()
        self.controller = FlightController()
        self.ocean = OceanEnvironment()
        
        # Initial state
        self.state = QuadrotorState()
        self.state.z = -1.0  # Initial depth 1 meter
        
        # Data recording
        self.time_history = []
        self.position_history = []
        self.attitude_history = []
        self.motor_history = []
        self.current_history = []
        
        # Simulation parameters
        self.dt = 0.02  # 50Hz
        self.running = False
        self.simulation_time = 0.0
        
        # Set mission
        self.controller.mission_mode = "patrol"
        
    def update_simulation(self):
        """Update simulation one step"""
        # Calculate control inputs
        control_inputs, motor_speeds = self.controller.calculate_control(self.state)
        
        # Update motor speeds
        self.state.motor1 = motor_speeds[0]
        self.state.motor2 = motor_speeds[1]
        self.state.motor3 = motor_speeds[2]
        self.state.motor4 = motor_speeds[3]
        
        # Update physics state
        self.state = self.physics.update_dynamics(self.state, control_inputs, self.ocean, self.dt)
        
        # Record data
        self.time_history.append(self.simulation_time)
        self.position_history.append([self.state.x, self.state.y, self.state.z])
        self.attitude_history.append([self.state.roll, self.state.pitch, self.state.yaw])
        self.motor_history.append([self.state.motor1, self.state.motor2, self.state.motor3, self.state.motor4])
        
        # Calculate current ocean current
        current = self.physics.calculate_ocean_current(
            np.array([self.state.x, self.state.y, self.state.z]), 
            time.time(), 
            self.ocean
        )
        self.current_history.append(current)
        
        self.simulation_time += self.dt
    
    def run_simulation(self, duration: float = 60.0):
        """Run simulation"""
        self.running = True
        start_time = time.time()
        
        print("Starting Underwater Quadrotor Simulation...")
        print("Mission Mode: Patrol")
        print("Target: Square patrol at 2 meters underwater depth")
        print("-" * 50)
        
        while self.running and (time.time() - start_time) < duration:
            self.update_simulation()
            time.sleep(self.dt)
        
        self.running = False
        print("Simulation completed!")

class RealTimeVisualizer:
    """Real-time Visualizer"""
    
    def __init__(self, simulator: UnderwaterQuadrotorSimulator):
        self.sim = simulator
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        
    def setup_plots(self):
        """Setup plot areas"""
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(2, 3, 1, projection='3d')
        self.ax_3d.set_title('3D Trajectory & Ocean Current', fontsize=12, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # Position plot
        self.ax_pos = self.fig.add_subplot(2, 3, 2)
        self.ax_pos.set_title('Position Changes', fontsize=12, fontweight='bold')
        self.ax_pos.set_xlabel('Time (s)')
        self.ax_pos.set_ylabel('Position (m)')
        
        # Attitude plot
        self.ax_att = self.fig.add_subplot(2, 3, 3)
        self.ax_att.set_title('Attitude Angles', fontsize=12, fontweight='bold')
        self.ax_att.set_xlabel('Time (s)')
        self.ax_att.set_ylabel('Angle (deg)')
        
        # Motor speed plot
        self.ax_motor = self.fig.add_subplot(2, 3, 4)
        self.ax_motor.set_title('Four Motor Speeds', fontsize=12, fontweight='bold')
        self.ax_motor.set_xlabel('Time (s)')
        self.ax_motor.set_ylabel('Speed (RPM)')
        
        # Velocity plot
        self.ax_vel = self.fig.add_subplot(2, 3, 5)
        self.ax_vel.set_title('Velocity', fontsize=12, fontweight='bold')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        
        # Status information
        self.ax_info = self.fig.add_subplot(2, 3, 6)
        self.ax_info.set_title('Real-time Status', fontsize=12, fontweight='bold')
        self.ax_info.axis('off')
        
        plt.tight_layout()
    
    def update_plots(self):
        """Update all plots"""
        if len(self.sim.time_history) < 2:
            return
        
        # Clear old plots
        for ax in [self.ax_3d, self.ax_pos, self.ax_att, self.ax_motor, self.ax_vel]:
            ax.clear()
        
        # Get data
        times = np.array(self.sim.time_history)
        positions = np.array(self.sim.position_history)
        attitudes = np.array(self.sim.attitude_history)
        motors = np.array(self.sim.motor_history)
        currents = np.array(self.sim.current_history)
        
        # Define beautiful color palette
        colors = {
            'primary': '#2E86AB',      # Ocean blue
            'secondary': '#A23B72',    # Deep pink
            'accent': '#F18F01',       # Golden orange
            'success': '#C73E1D',     # Deep red
            'info': '#7209B7',        # Purple
            'warning': '#F77F00',     # Orange
            'light': '#F1FAEE',       # Light green
            'dark': '#1D3557'         # Dark blue
        }
        
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(2, 3, 1, projection='3d')
        self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       color=colors['primary'], linewidth=3, alpha=0.8, label='Trajectory')
        self.ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                          c=colors['success'], s=150, alpha=0.9, label='Current Position')
        
        # Add ocean current arrow
        if len(currents) > 0:
            current = currents[-1]
            self.ax_3d.quiver(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                             current[0], current[1], current[2], 
                             color=colors['accent'], alpha=0.8, linewidth=3, 
                             label='Ocean Current')
        
        self.ax_3d.set_title('3D Trajectory & Ocean Current', fontsize=12, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.legend(fontsize=9)
        
        # Position plot
        self.ax_pos.plot(times, positions[:, 0], color=colors['success'], linewidth=2.5, label='X', alpha=0.8)
        self.ax_pos.plot(times, positions[:, 1], color=colors['info'], linewidth=2.5, label='Y', alpha=0.8)
        self.ax_pos.plot(times, positions[:, 2], color=colors['primary'], linewidth=2.5, label='Z', alpha=0.8)
        self.ax_pos.set_title('Position Changes', fontsize=12, fontweight='bold')
        self.ax_pos.set_xlabel('Time (s)')
        self.ax_pos.set_ylabel('Position (m)')
        self.ax_pos.legend(fontsize=9)
        self.ax_pos.grid(True, alpha=0.3)
        
        # Attitude plot
        self.ax_att.plot(times, np.degrees(attitudes[:, 0]), color=colors['success'], linewidth=2.5, label='Roll', alpha=0.8)
        self.ax_att.plot(times, np.degrees(attitudes[:, 1]), color=colors['info'], linewidth=2.5, label='Pitch', alpha=0.8)
        self.ax_att.plot(times, np.degrees(attitudes[:, 2]), color=colors['primary'], linewidth=2.5, label='Yaw', alpha=0.8)
        self.ax_att.set_title('Attitude Angles', fontsize=12, fontweight='bold')
        self.ax_att.set_xlabel('Time (s)')
        self.ax_att.set_ylabel('Angle (deg)')
        self.ax_att.legend(fontsize=9)
        self.ax_att.grid(True, alpha=0.3)
        
        # Motor speed plot
        self.ax_motor.plot(times, motors[:, 0], color=colors['success'], linewidth=2.5, label='Motor1(FL)', alpha=0.8)
        self.ax_motor.plot(times, motors[:, 1], color=colors['info'], linewidth=2.5, label='Motor2(FR)', alpha=0.8)
        self.ax_motor.plot(times, motors[:, 2], color=colors['primary'], linewidth=2.5, label='Motor3(BL)', alpha=0.8)
        self.ax_motor.plot(times, motors[:, 3], color=colors['accent'], linewidth=2.5, label='Motor4(BR)', alpha=0.8)
        self.ax_motor.set_title('Four Motor Speeds', fontsize=12, fontweight='bold')
        self.ax_motor.set_xlabel('Time (s)')
        self.ax_motor.set_ylabel('Speed (RPM)')
        self.ax_motor.legend(fontsize=9)
        self.ax_motor.grid(True, alpha=0.3)
        
        # Velocity plot
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0) / self.sim.dt
            vel_times = times[1:]
            self.ax_vel.plot(vel_times, velocities[:, 0], color=colors['success'], linewidth=2.5, label='Vx', alpha=0.8)
            self.ax_vel.plot(vel_times, velocities[:, 1], color=colors['info'], linewidth=2.5, label='Vy', alpha=0.8)
            self.ax_vel.plot(vel_times, velocities[:, 2], color=colors['primary'], linewidth=2.5, label='Vz', alpha=0.8)
            self.ax_vel.set_title('Velocity', fontsize=12, fontweight='bold')
            self.ax_vel.set_xlabel('Time (s)')
            self.ax_vel.set_ylabel('Velocity (m/s)')
            self.ax_vel.legend(fontsize=9)
            self.ax_vel.grid(True, alpha=0.3)
        
        # Status information
        self.ax_info.clear()
        self.ax_info.set_title('Real-time Status', fontsize=12, fontweight='bold')
        self.ax_info.axis('off')
        
        current_state = self.sim.state
        info_text = f"""
Current Position: ({current_state.x:.2f}, {current_state.y:.2f}, {current_state.z:.2f}) m
Current Velocity: ({current_state.vx:.2f}, {current_state.vy:.2f}, {current_state.vz:.2f}) m/s
Current Attitude: ({np.degrees(current_state.roll):.1f}, {np.degrees(current_state.pitch):.1f}, {np.degrees(current_state.yaw):.1f}) °

Motor Speeds (RPM):
Motor1(FL): {current_state.motor1:.0f}
Motor2(FR): {current_state.motor2:.0f}
Motor3(BL): {current_state.motor3:.0f}
Motor4(BR): {current_state.motor4:.0f}

Target Position: ({self.sim.controller.target_x:.2f}, {self.sim.controller.target_y:.2f}, {self.sim.controller.target_z:.2f}) m
Mission Mode: {self.sim.controller.mission_mode}
Simulation Time: {self.sim.simulation_time:.1f} s
        """
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['light'], alpha=0.8))
        
        plt.tight_layout()
    
    def animate(self, frame):
        """Animation update function"""
        if self.sim.running:
            self.sim.update_simulation()
            self.update_plots()
        return []

def main():
    """Main function"""
    print("=" * 60)
    print("6-DOF Underwater Quadrotor UAV Simulator")
    print("=" * 60)
    
    # Create simulator
    simulator = UnderwaterQuadrotorSimulator()
    
    # Create visualizer
    visualizer = RealTimeVisualizer(simulator)
    
    # Setup animation
    ani = animation.FuncAnimation(visualizer.fig, visualizer.animate, 
                                interval=50, blit=False, repeat=False, cache_frame_data=False)
    
    # Start simulation thread
    def run_simulation():
        simulator.run_simulation(duration=60.0)  # Run for 60 seconds
    
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    
    # Show interface
    plt.show()
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()
