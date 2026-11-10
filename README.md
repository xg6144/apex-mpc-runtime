# Apex MPC

Apex MPC is a sampling-based nonlinear model predictive controller for F1TENTH vehicles running ROS 2. It follows a raceline and avoids obstacles tracked by LiDAR. The controller rejects trajectories predicted to collide with walls or obstacles and publishes a stop command on `/drive` if no candidate is safe.

## Build

Install ROS 2 Humble, `colcon`, and `rosdep` on Linux. Then run the following commands from the repository root.

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths race_track --ignore-src -r -y
colcon build --symlink-install --packages-select race_track_mpc
source install/setup.bash
```

Numba enables the accelerated CPU backend. The controller falls back to NumPy when acceleration is unavailable.

## Run

### Connect to an existing ROS 2 setup

If odometry and LiDAR are already running, launch the MPC node directly. By default it subscribes to `/ego_racecar/odom` and `/scan` and publishes commands on `/drive`. The package installs a default map and raceline.

```bash
ros2 launch race_track_mpc mpc_launch.py
```

### Run on the vehicle

Start the VESC driver, LiDAR, localization, and the multiplexer that switches between manual and autonomous commands. The vehicle script uses `/pf/pose/odom` for odometry and `/teleop` for manual override. It checks for other `/drive` publishers before starting the controller.

```bash
F1TENTH_VEHICLE_WORKSPACE="$HOME/f1tenth_ws" \
  bash race_track/scripts/run_car.sh --speed 0.7 --no-bag
```

The vehicle configuration uses the default regenerative braking thresholds from the [VESC patch](race_track/vehicle_patches/vesc_motor_brake.patch). Apply the patch to the vehicle's `ackermann_to_vesc` source and rebuild the vehicle workspace to enable regenerative braking.

### Run in simulation

Build `f1tenth_gym_ros` in a separate workspace. The simulation launch uses `gym_bridge_launch.py` with a `config` argument. To match the real vehicle's steering conversion and vehicle parameters, use a bridge that supports those options.

```bash
F1TENTH_SIM_WORKSPACE="$HOME/src_sim" \
  bash race_track/scripts/run_sim.sh --headless --no-bag
```

Both scripts accept `--map` and `--raceline` to select a different track.
