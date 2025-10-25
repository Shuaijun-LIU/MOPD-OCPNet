# 6-DOF Underwater Quadrotor UAV Simulator

A complete underwater quadrotor drone simulation system featuring physics modeling, ocean environment effects, control systems, and real-time visualization.

## Features

- **6-DOF Physics Model**: Complete position and attitude dynamics
- **Four-Rotor Control**: Independent control of four motor speeds
- **Ocean Environment Effects**: Ocean currents, turbulence, depth-related drag
- **Real-time Visualization**: 3D trajectory, status monitoring, data charts
- **Mission Planning**: Support for hover, patrol, and other mission modes
- **Emergency Situations**: Random current variations, environmental disturbances

## Installation

```bash
pip install -r requirements.txt
```

## Running the Simulation

```bash
python underwater_quadrotor_simulator.py
```

## System Parameters

### Physical Parameters
- Mass: 2.0 kg
- Rotor arm length: 0.25 m
- Gravity: 9.81 m/s²
- Seawater density: 1025 kg/m³

### Control Parameters
- Position control: PID (Kp=2.0, Ki=0.1, Kd=0.5)
- Attitude control: PID (Kp=1.0, Ki=0.05, Kd=0.2)
- Altitude control: PID (Kp=3.0, Ki=0.2, Kd=0.8)

### Ocean Environment
- Current speed: 0.5 m/s
- Turbulence intensity: 0.1
- Depth influence: Exponential decay

## Real-time Output

The program displays in real-time:
1. **3D Trajectory Plot**: Drone movement trajectory and ocean current direction
2. **Position Changes**: X, Y, Z coordinates over time
3. **Attitude Angles**: Roll, pitch, yaw angles
4. **Four Motor Speeds**: Real-time RPM of all four motors
5. **Velocity**: Linear velocity components
6. **Status Information**: Current position, target position, mission mode, etc.

## Mission Modes

- **Hover Mode**: Maintain stability at specified position
- **Patrol Mode**: Move along predetermined path (default square patrol)

## Technical Features

- Underwater added mass effects
- Depth-related ocean current influence
- Random turbulence and emergency situations
- Real-time PID control
- Multi-threaded simulation and visualization

## Usage Instructions

1. The program automatically starts a 60-second patrol mission when run
2. Observe various charts to understand drone status
3. Modify parameters in the code to adjust behavior
4. Press Ctrl+C to end simulation early
