#!/usr/bin/env bash
# One-shot F1TENTH real-vehicle runner: sources the workspace, verifies the
# vehicle bringup (lidar + odometry) is alive, then starts the MPC controller
# with real-car topic remaps and the validated CSV speed profile.
#
# The vehicle bringup (VESC, lidar driver, localization) must already be
# running - this script starts only the controller.
#
# Usage:
#   ./run_car.sh                          # full CSV profile (speed_scale 1.0)
#   ./run_car.sh --speed 0.7              # reduced-pace validation run
#   ./run_car.sh --full                   # race pace (speed_scale 1.0)
#   ./run_car.sh --race                   # hot-lap pace (speed 1.5; the
#                                         # profile envelope saturates here -
#                                         # more scale changes nothing.
#                                         # Speed-first setting: expect
#                                         # occasional wall brushes at low
#                                         # grip; --speed 1.3 is the
#                                         # consistency option)
#   ./run_car.sh --odom-topic /odom --drive-topic /drive
#   ./run_car.sh --manual-topic /teleop --manual-timeout 0.25
#   ./run_car.sh --map ~/maps/track.yaml --raceline ~/maps/track_raceline.csv
#   ./run_car.sh --no-bag                # skip the rosbag recording
#   ./run_car.sh --bag-dir ~/other/bags  # change where recordings are stored
#   ./run_car.sh --no-avoid              # solo time attack: no obstacle
#                                        # tracking (wall costs + e-stop stay)
#   ./run_car.sh --raceline-editor       # live RViz raceline editor: drag a
#                                        # handle sideways to move the LINE,
#                                        # up/down to set its speed
#   ./run_car.sh --raceline-editor --editor-speed-only
#                                        # the old speed-only editor
#   ./run_car.sh --no-imu                # disable VESC IMU fusion + safety layer
#   ./run_car.sh --grip-limit            # re-enable the realtime grip
#                                        # estimate (off by default)
#   ./run_car.sh --backend gpu            # explicit GPU timing A/B; CPU is
#                                        # always the production default
#   ./run_car.sh --allow-shared-drive    # skip the sole-publisher check
#   ./run_car.sh --build                 # rebuild this apex_ws before launch
# (no -u: the ROS setup scripts reference unbound variables)
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APEX_WORKSPACE="$(cd -- "${PACKAGE_DIR}/.." && pwd)"
VEHICLE_WORKSPACE="${F1TENTH_VEHICLE_WORKSPACE:-$HOME/f1tenth_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
VEHICLE_SETUP="${VEHICLE_WORKSPACE}/install/setup.bash"
APEX_SETUP="${APEX_WORKSPACE}/install/setup.bash"
CONFIG="${PACKAGE_DIR}/config/mpc.yaml"
RACELINE="${PACKAGE_DIR}/track_file/ifac_roboracer_raceline_userdrawn_relief25_a60.csv"
MAP_YAML="${PACKAGE_DIR}/track_file/ifac_roboracer.yaml"

# Common real-car defaults: particle-filter odometry and the ackermann mux
# inputs. /drive is navigation; /teleop is the higher-priority LB/manual input.
# Adjust with the flags below to match the vehicle bringup.
ODOM_TOPIC=/pf/pose/odom
SCAN_TOPIC=/scan
DRIVE_TOPIC=/drive
MANUAL_TOPIC=/teleop
MANUAL_TIMEOUT=0.25
SPEED_SCALE=1.0
LATENCY=0.08 #0.08          # command->yaw transport delay fitted on bags
                      # mpc_20260824_002320/_002707 (~0.10 s incl. ~0.01 s of
                      # PF latency); 0.05 under-compensated the real chain
WAIT_SECONDS=15
RECORD_BAG=1
BAG_DIR="${VEHICLE_WORKSPACE}/bags"
AVOIDANCE=true
RACELINE_EDITOR=false
EDITOR_GEOMETRY=true       # --raceline-editor moves the line, not just its speed
EDITOR_MIN_CLEARANCE=0.10  # body-to-wall floor a live geometry edit may not cross
IMU=true
GRIP_LIMIT=false  # see the resolve step below; --grip-limit turns it on
BUILD=false
ALLOW_SHARED_DRIVE=false
MPC_BACKEND=cpu

while [[ $# -gt 0 ]]; do
    case "$1" in
        --speed)       SPEED_SCALE="$2"; shift ;;
        --full)        SPEED_SCALE=1.0 ;;
        --race)        SPEED_SCALE=1.5 ;;
        --map)         MAP_YAML="$2"; shift ;;
        --raceline)    RACELINE="$2"; shift ;;
        --odom-topic)  ODOM_TOPIC="$2"; shift ;;
        --scan-topic)  SCAN_TOPIC="$2"; shift ;;
        --drive-topic) DRIVE_TOPIC="$2"; shift ;;
        --manual-topic) MANUAL_TOPIC="$2"; shift ;;
        --manual-timeout) MANUAL_TIMEOUT="$2"; shift ;;
        --latency)     LATENCY="$2"; shift ;;
        --no-bag)      RECORD_BAG=0 ;;
        --bag-dir)     BAG_DIR="$2"; shift ;;
        --no-avoid)    AVOIDANCE=false ;;
        --raceline-editor) RACELINE_EDITOR=true ;;
        --editor-speed-only) EDITOR_GEOMETRY=false ;;
        --editor-min-clearance) EDITOR_MIN_CLEARANCE="$2"; shift ;;
        --no-imu)      IMU=false ;;
        --grip-limit)    GRIP_LIMIT=true ;;
        --no-grip-limit) GRIP_LIMIT=false ;;
        --backend)        MPC_BACKEND="$2"; shift ;;
        --allow-shared-drive) ALLOW_SHARED_DRIVE=true ;;
        --build)       BUILD=true ;;
        -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

case "$MPC_BACKEND" in
    cpu|gpu|numpy) ;;
    *) echo "--backend expects cpu, gpu, or numpy (got $MPC_BACKEND)" >&2; exit 1 ;;
esac

[[ -f "$ROS_SETUP" ]] || {
    echo "[run_car] ROS setup not found: $ROS_SETUP" >&2
    exit 1
}

# shellcheck disable=SC1090
source "$ROS_SETUP"
# Vehicle bringup workspace: needed so the bag recorder can serialize
# vesc_msgs types (/sensors/core, /sensors/imu).
if [[ -f "$VEHICLE_SETUP" ]]; then
    # shellcheck disable=SC1090
    source "$VEHICLE_SETUP"
fi

if "$BUILD" || [[ ! -f "$APEX_SETUP" ]]; then
    echo "[run_car] building race_track_mpc from $APEX_WORKSPACE ..."
    (
        cd "$APEX_WORKSPACE"
        colcon build --symlink-install --packages-up-to race_track_mpc
    )
fi
[[ -f "$APEX_SETUP" ]] || {
    echo "[run_car] apex setup not found after build: $APEX_SETUP" >&2
    exit 1
}
# Load this workspace last so its controller takes precedence.
# shellcheck disable=SC1090
source "$APEX_SETUP"

# The particle filter uses RangeLib's CUDA ray marcher at scan rate.  The latest
# loaded bag showed that sharing that GPU stretched MPC solve p95 to 35 ms,
# whereas the fused CPU backend is deterministic without GPU contention.  On
# this Orin the installed TBB interface is too old for Numba's TBB layer.  The
# production kernel is fused and serial; one worker avoids fork/join jitter
# and leaves PF/lidar cores available.
# taskset still lets the node's sensor/control/viz threads use the CPU set.
export MPC_ACCEL_THREADS="${MPC_ACCEL_THREADS:-1}"
# Never inherit a stale GPU selection from the caller's shell.  The weaving
# bag shared that GPU with the particle filter and missed its control deadline;
# a non-CPU backend now requires an explicit --backend choice on this run.
export MPC_ACCEL_BACKEND="$MPC_BACKEND"
MPC_CPU_SET="${MPC_CPU_SET:-}"
# export MPC_DISABLE_ACCEL=1   # uncomment to force the pure-NumPy path
# ./run_car.sh --backend gpu ...  # explicit no-PF/GPU A/B override

# Numba warns whenever a CUDA grid has under 128 blocks, a threshold sized for
# datacenter GPUs; on the 8-SM Orin NX our 8-32 block launches are fine.
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0

[[ -f "$RACELINE" ]] || { echo "[run_car] raceline not found: $RACELINE" >&2; exit 1; }
[[ -f "$MAP_YAML" ]] || { echo "[run_car] map yaml not found: $MAP_YAML" >&2; exit 1; }

# Sector speed scaling removed 2026-08-24: speed shaping lives in the
# raceline CSV (tools/reprofile_raceline.py) plus the runtime feasibility
# passes; no side-channel params file can slow a line silently any more.

wait_for_topic() {
    local topic="$1" deadline=$((SECONDS + WAIT_SECONDS))
    echo -n "[run_car] waiting for $topic "
    while (( SECONDS < deadline )); do
        if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
            echo "- ok"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo
    echo "[run_car] ERROR: $topic not available after ${WAIT_SECONDS}s." >&2
    echo "[run_car] Start the vehicle bringup (lidar/odometry) first." >&2
    return 1
}

wait_for_topic "$SCAN_TOPIC"
wait_for_topic "$ODOM_TOPIC"
wait_for_topic "$MANUAL_TOPIC"

echo "[run_car] starting MPC: speed_scale=$SPEED_SCALE odom=$ODOM_TOPIC drive=$DRIVE_TOPIC"
echo "[run_car] controller workspace=$APEX_WORKSPACE"
echo "[run_car] mode gate: manual=$MANUAL_TOPIC timeout=${MANUAL_TIMEOUT}s (LB=manual, RB=autonomous)"
echo "[run_car] map=$MAP_YAML"
echo "[run_car] raceline=$RACELINE"
# The realtime grip estimate is OFF by default.  It reads the measured/expected
# yaw ratio and, on this car, that expectation is biased: in bag
# mpc_20260822_215337 the estimate sat pinned at its 3.5 m/s^2 floor for 65.7%
# of the AUTO lap while the car was actually sustaining roughly twice that
# (IMU |a_y| p95 4.14, max 9.68).  A limiter that is wrong in the conservative
# direction still throws away corner speed, and until the bias is traced to
# understeer_gradient or steering_time_constant a fixed max_lateral_accel is
# the more honest limit.  It is also what every simulator run has been
# validated with, so leaving it on meant racing an untested configuration.
# Re-enable with --grip-limit once the yaw model is trusted again.
#
# It cannot outlive --no-imu (it reads the IMU), but the reverse is not true
# and conflating them is expensive: IMU fusion also supplies
# imu_use_scan_alignment, which refines the scan/pose alignment ahead of the
# newest odometry sample.  --no-imu therefore still disables both.
if [[ "$IMU" == false ]]; then
    GRIP_LIMIT=false
fi
echo "[run_car] imu fusion=$IMU (latency comp, scan alignment, speed fusion, spin/collision safety)"
echo "[run_car] realtime grip limit=$GRIP_LIMIT"

# --- sole publisher on the command topic ------------------------------------
# The ackermann mux forwards whichever message arrived last, so a second
# controller on this topic does not lose the argument - it interleaves with
# ours at its own rate.  Bags mpc_20260822_{201801,215037,215337} were all
# ruined this way: advanced_pp (a different workspace, never launched by this
# script) published at 80 Hz and took 62% of the commands that reached the
# VESC during the one AUTO lap, defeating max_steer_rate in the process.  It
# costs a second to check, and the check is the difference between a run we
# can analyse and one we cannot.
DRIVE_PUBLISHERS="$(ros2 topic info "$DRIVE_TOPIC" 2>/dev/null \
    | sed -n 's/^Publisher count: //p')"
if [[ -n "$DRIVE_PUBLISHERS" && "$DRIVE_PUBLISHERS" -gt 0 ]]; then
    echo "[run_car] $DRIVE_TOPIC already has $DRIVE_PUBLISHERS publisher(s) before this node starts:" >&2
    ros2 topic info "$DRIVE_TOPIC" --verbose 2>/dev/null \
        | sed -n 's/^  *Node name: /  /p' >&2
    if ! $ALLOW_SHARED_DRIVE; then
        echo "[run_car] refusing to share $DRIVE_TOPIC - stop the other controller" >&2
        echo "[run_car] (e.g. pkill -f advanced_pp), or pass --allow-shared-drive" >&2
        exit 1
    fi
    echo "[run_car] continuing anyway (--allow-shared-drive): commands will interleave" >&2
fi
if $RACELINE_EDITOR; then
    echo "[run_car] raceline editor ENABLED: unrestricted live edits; Save overwrites $RACELINE"
    if $EDITOR_GEOMETRY; then
        echo "[run_car]   drag a handle in the map plane to move the line (speed is"
        echo "[run_car]   re-derived from the new curvature); drag it vertically to set"
        echo "[run_car]   speed. Edits under ${EDITOR_MIN_CLEARANCE} m of body clearance are refused."
    else
        echo "[run_car]   speed only: map X/Y is pinned (--editor-speed-only)"
    fi
fi
if [[ "$SPEED_SCALE" != "1.0" ]]; then
    echo "[run_car] NOTE: running below race pace; use --full once a clean lap is confirmed."
fi

# Humble quirk: --params-file wins over -p overrides, so the overrides go into
# a second params file (the later file takes precedence).
OVERRIDES="$(mktemp /tmp/mpc_car_overrides_XXXX.yaml)"

BAG_PID=""
cleanup() {
    if [[ -n "$BAG_PID" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
        # SIGINT lets rosbag2 finalize metadata.yaml; SIGKILL would corrupt it.
        kill -INT "$BAG_PID" 2>/dev/null || true
        wait "$BAG_PID" 2>/dev/null || true
        echo "[run_car] bag saved: $RUN_DIR/rosbag2"
    fi
    rm -f "$OVERRIDES"
}
trap cleanup EXIT INT TERM

cat > "$OVERRIDES" <<EOF
mpc_controller:
  ros__parameters:
    raceline_path: $RACELINE
    map_yaml_path: $MAP_YAML
    odom_topic: $ODOM_TOPIC
    scan_topic: $SCAN_TOPIC
    drive_topic: $DRIVE_TOPIC
    enable_manual_override: true
    manual_override_topic: $MANUAL_TOPIC
    manual_override_timeout: $MANUAL_TIMEOUT
    speed_scale: $SPEED_SCALE
    latency_compensation: $LATENCY
    emergency_stop_distance: 0.18
    odom_timeout: 0.3
    control_period: 0.02
    control_on_odometry: true
    horizon: 20 #15
    samples: 160 #160 #320
    iterations: 2
    min_accel: -3.8
    max_accel: 6.0 #6.0 #5.0

    avoid_replan_period: 0.12
    avoid_coarse_offset_step: 0.20
    avoid_refine_top_k: 1

    avoid_full_speed_gap: 0.35
    avoid_full_speed_confirm_cycles: 1

    recovery_hold_time: 3.0
    recovery_error_deadband: 0.30
    recovery_surge_accel: 2.5

    lane_memory_enabled: true
    lane_memory_s_tolerance: 0.60
    lane_memory_lateral_tolerance: 0.30
    lane_memory_radius_tolerance: 0.15
    lane_memory_size: 12

    ego_disc_offset: 0.145
    ego_disc_radius: 0.175
    wall_weight: 0.0
    wall_margin: 0.0
    wall_hard_margin: 0.10

    max_lateral_accel: 7.5
    reference_speed_caps: true

    reference_brake_accel: 3.8
    reference_accel_limit: 5.0

    enable_speed_command_cap: true
    speed_command_cap_margin: 0.5
    enable_vesc_brake_command: true

    enable_wheel_slip_correction: true
    wheel_slip_window: 0.20
    wheel_slip_time_constant: 0.05
    wheel_slip_max_correction: 1.0
    friction_weight: 60.0

    enable_realtime_grip_limit: $GRIP_LIMIT
    grip_limit_initial: 5.0
    grip_limit_min: 3.5
    grip_limit_max: 6.0
    grip_understeer_ratio: 0.82
    grip_oversteer_ratio: 1.25
    grip_slip_hold_time: 0.08
    grip_attack_time_constant: 0.08
    grip_release_time_constant: 8.0
    grip_safety_factor: 0.90
    grip_min_speed: 1.5
    grip_min_expected_yaw_rate: 0.5
    grip_hard_weight: 50000.0

    speed_weight: 12.0

    contour_weight: 150.0 #150.0

    speed_command_preview: 0.20

    max_steer_rate: 2.218

    steer_rate_weight: 64.0
    initial_steer_std: 0.12

    min_steer: -0.48
    max_steer: 0.51

    wheelbase: 0.325

    steering_gain: 1.0500

    steering_time_constant: 0.08

    understeer_gradient: 0.0130

    minimum_reference_speed: 1.6

    wall_filter_margin: 0.50
    wall_center_gate: 0.50

    obstacle_path_keep_wall_floor: 0.50

    tracker_velocity_beta: 0.05
    tracker_static_speed: 1.0
    obstacle_merge_gap: 0.10
    enable_obstacle_avoidance: $AVOIDANCE

    enable_static_obstacle_bake: false
    bake_age: 40
    bake_position_std: 0.12
    bake_absent_scans: 20
    bake_radius_margin: 0.08

    ego_center_offset: 0.1625

    enable_raceline_editor: $RACELINE_EDITOR

    raceline_editor_save_path: $RACELINE

    raceline_editor_allow_geometry: $EDITOR_GEOMETRY
    raceline_editor_min_clearance: $EDITOR_MIN_CLEARANCE

    enable_imu: $IMU

    max_speed: 10.0
EOF

if [[ "$RECORD_BAG" == "1" ]]; then
    RUN_DIR="$BAG_DIR/mpc_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    # Archive the exact parameters this run drove with, next to the data.
    cp "$CONFIG" "$RUN_DIR/mpc.yaml"
    cp "$OVERRIDES" "$RUN_DIR/overrides.yaml"
    cp "$RACELINE" "$RUN_DIR/raceline.csv"
    cp "$MAP_YAML" "$RUN_DIR/map.yaml"
    {
        echo "MPC_ACCEL_BACKEND=$MPC_ACCEL_BACKEND"
        echo "MPC_ACCEL_THREADS=$MPC_ACCEL_THREADS"
        echo "MPC_CPU_SET=$MPC_CPU_SET"
    } > "$RUN_DIR/runtime.env"
    # the occupancy image too, so the clearance tools can measure the
    # body against the exact map this run drove
    MAP_IMAGE="$(dirname "$MAP_YAML")/$(sed -n 's/^image:[[:space:]]*//p' "$MAP_YAML" | tr -d '\r')"
    [[ -f "$MAP_IMAGE" ]] && cp "$MAP_IMAGE" "$RUN_DIR/"
    # Everything needed to replay/tune offline: localization + raw sensors,
    # the MPC's command and debug paths, actuator feedback, and the manual
    # override stream (to spot takeovers).  Deduped in case a flag pointed a
    # configurable topic at one already in the fixed list.
    TOPICS=(
        "$ODOM_TOPIC" "$SCAN_TOPIC" "$DRIVE_TOPIC" "$MANUAL_TOPIC"
        /mpc/raceline /mpc/prediction /mpc/obstacles
        /mpc/compute_breakdown /mpc/state
        /pf/viz/inferred_pose /odom
        /sensors/core /sensors/imu /sensors/imu/raw
        /ackermann_cmd /commands/motor/speed /commands/motor/brake
        /commands/servo/position
        /joy /tf /tf_static /map /parameter_events /rosout
    )
    mapfile -t TOPICS < <(printf '%s\n' "${TOPICS[@]}" | sort -u)

    ros2 bag record -o "$RUN_DIR/rosbag2" "${TOPICS[@]}" \
        >"$RUN_DIR/rosbag_record.log" 2>&1 &
    BAG_PID=$!
    ln -sfn "$RUN_DIR" "$BAG_DIR/latest"
    echo "[run_car] recording rosbag: $RUN_DIR/rosbag2 (--no-bag to disable)"
fi

# No exec: the shell must outlive the controller to stop the bag recorder.
# Param file order (later wins): base config < the real-car overrides above.
PARAM_ARGS=(--params-file "$CONFIG")
PARAM_ARGS+=(--params-file "$OVERRIDES")
if [[ -n "$MPC_CPU_SET" ]]; then
    echo "[run_car] MPC backend=$MPC_ACCEL_BACKEND threads=$MPC_ACCEL_THREADS cpus=$MPC_CPU_SET"
    taskset --cpu-list "$MPC_CPU_SET" \
        ros2 run race_track_mpc mpc_controller --ros-args "${PARAM_ARGS[@]}"
else
    echo "[run_car] MPC backend=$MPC_ACCEL_BACKEND threads=$MPC_ACCEL_THREADS (no CPU affinity)"
    ros2 run race_track_mpc mpc_controller --ros-args "${PARAM_ARGS[@]}"
fi
