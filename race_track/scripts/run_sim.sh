#!/usr/bin/env bash
# Run the apex MPC against the f1tenth_gym_ros physics simulator in ~/src_sim.
#
# The simulator is a single-track dynamic model (RK4, 100 Hz) with tyre slip,
# so unlike the kinematic unit tests it can actually spin, understeer and
# crash.  launch/sim_mpc_launch.py matches the simulated vehicle to ours (grip
# 6.0 m/s^2, real command gain 1.0956, wheel travel -0.5259/+0.5588 rad,
# servo slew 4.06285 rad/s) and
# tools/sim_overrides.py copies the controller tuning straight out of
# run_car.sh, so a sim result is about the car and not about a second config.
#
# Usage:
#   ./run_sim.sh                          # RViz, IFAC map, 20-lap-validated straightened line
#   ./run_sim.sh --headless --laps 3      # automated run, stops after 3 laps
#   ./run_sim.sh --raceline <csv>         # A/B a different line
#   ./run_sim.sh --speed 0.8              # scale the reference speeds
#   ./run_sim.sh --spawn-lateral 0.20     # start 20 cm left of the raceline
#   ./run_sim.sh --spawn-heading -0.10    # start with -0.10 rad heading error
#   ./run_sim.sh --samples 512            # CEM quality A/B
#   ./run_sim.sh --domain 78              # isolate from another sim on this PC
#   ./run_sim.sh --stock-vehicle          # gym's f1tenth preset, not our car
#   ./run_sim.sh --mu 0.47                # pessimistic grip (4.6 m/s^2)
#   ./run_sim.sh --set contour_weight=96  # sweep any controller parameter
#   ./run_sim.sh --veh sv_max=2.0         # sweep the simulated plant instead
#   ./run_sim.sh --no-avoid --no-bag
#   ./run_sim.sh --no-controller          # attach another policy to /drive
#   ./run_sim.sh --timeout 180            # hard stop (default 300 s)
#   ./run_sim.sh --stall 60               # give up after 60 s without a lap
#   ./run_sim.sh --build                  # rebuild apex_ws first
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APEX_WORKSPACE="$(cd -- "${PACKAGE_DIR}/.." && pwd)"
SIM_WORKSPACE="${F1TENTH_SIM_WORKSPACE:-$HOME/src_sim}"

MAP_YAML="${PACKAGE_DIR}/track_file/ifac_roboracer.yaml"
RACELINE="${PACKAGE_DIR}/track_file/ifac_roboracer_raceline_userdrawn_corner15_a60.csv"
BAG_DIR="${SIM_WORKSPACE}/bags"
DOMAIN="${SIM_DOMAIN_ID:-78}"
SPEED_SCALE=1.0
SPAWN_LATERAL=0.0       # metres, positive to the raceline's left
SPAWN_HEADING=0.0       # radians, added to the raceline heading
SAMPLES=""
LAPS=0                 # 0 = run until --timeout
TIMEOUT=300
STALL=45               # no new lap for this long => wedged or crashed
RVIZ=true
BUILD=false
AVOIDANCE=true
START_CONTROLLER=true
RECORD_BAG=1
MATCH_CAR=true
EDITOR=false
MU=""                 # empty = 0.612 (6.0 m/s^2), the ggv envelope
SETS=()               # --set key=value, repeatable parameter sweep
VEH=""                # --veh sv_max=2.0,mu=0.5 : gym plant overrides

while [[ $# -gt 0 ]]; do
    case "$1" in
        --map)         MAP_YAML="$2"; shift ;;
        --raceline)    RACELINE="$2"; shift ;;
        --speed)       SPEED_SCALE="$2"; shift ;;
        --spawn-lateral) SPAWN_LATERAL="$2"; shift ;;
        --spawn-heading) SPAWN_HEADING="$2"; shift ;;
        --samples)     SAMPLES="$2"; shift ;;
        --laps)        LAPS="$2"; shift ;;
        --timeout)     TIMEOUT="$2"; shift ;;
        --stall)       STALL="$2"; shift ;;
        --domain)      DOMAIN="$2"; shift ;;
        --bag-dir)     BAG_DIR="$2"; shift ;;
        --headless)    RVIZ=false ;;
        --no-avoid)    AVOIDANCE=false ;;
        --no-controller) START_CONTROLLER=false ;;
        --no-bag)      RECORD_BAG=0 ;;
        --mu)          MU="$2"; shift ;;
        --set)         SETS+=(--set "$2"); shift ;;
        --veh)         VEH="$2"; shift ;;
        --stock-vehicle) MATCH_CAR=false ;;
        --raceline-editor) EDITOR=true ;;
        --build)       BUILD=true ;;
        -h|--help)     sed -n 's/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

[[ -f "$RACELINE" ]] || { echo "[run_sim] raceline not found: $RACELINE" >&2; exit 1; }
[[ -f "$MAP_YAML" ]] || { echo "[run_sim] map yaml not found: $MAP_YAML" >&2; exit 1; }
[[ -d "$SIM_WORKSPACE" ]] || { echo "[run_sim] simulator workspace not found: $SIM_WORKSPACE" >&2; exit 1; }

# --- environment ------------------------------------------------------------
# sim_env.sh refuses to load into a shell that already has a vehicle workspace
# on AMENT_PREFIX_PATH.  That guard is for interactive shells; here the mixing
# is deliberate and one-way: the simulator's overlay goes on first, apex last.
set +u
export SIM_DOMAIN_ID="$DOMAIN"
export SIM_ALLOW_MIXED=1
# shellcheck disable=SC1090
if [[ -f "${SIM_WORKSPACE}/sim_env.sh" ]]; then
    source "${SIM_WORKSPACE}/sim_env.sh"
else
    source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    source "${SIM_WORKSPACE}/install/setup.bash"
    export ROS_DOMAIN_ID="$DOMAIN"
fi

if "$BUILD"; then
    echo "[run_sim] building race_track_mpc ..."
    ( cd "$APEX_WORKSPACE" && colcon build --symlink-install --packages-up-to race_track_mpc )
fi
APEX_SETUP="${APEX_WORKSPACE}/install/setup.bash"
[[ -f "$APEX_SETUP" ]] || { echo "[run_sim] apex setup not found: $APEX_SETUP" >&2; exit 1; }
# shellcheck disable=SC1090
source "$APEX_SETUP"
set -u

export F1TENTH_SIM_WORKSPACE="$SIM_WORKSPACE"
export MPC_ACCEL_THREADS="${MPC_ACCEL_THREADS:-1}"
export MPC_ACCEL_BACKEND="${MPC_ACCEL_BACKEND:-cpu}"
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0

# --- run directory ----------------------------------------------------------
RUN_DIR="${BAG_DIR}/sim_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
OVERRIDES="${RUN_DIR}/overrides.yaml"
python3 "${PACKAGE_DIR}/tools/sim_overrides.py" \
    --raceline "$RACELINE" --map "$MAP_YAML" \
    --speed "$SPEED_SCALE" --avoid "$AVOIDANCE" --editor "$EDITOR" \
    "${SETS[@]}" -o "$OVERRIDES"
cp "$RACELINE" "$RUN_DIR/raceline.csv"
cp "$MAP_YAML" "$RUN_DIR/map.yaml"
# the occupancy image too, so tools/analyze_sim_run.py can measure body
# clearance against the exact map this run drove
MAP_IMAGE="$(dirname "$MAP_YAML")/$(sed -n 's/^image:[[:space:]]*//p' "$MAP_YAML" | tr -d '\r')"
[[ -f "$MAP_IMAGE" ]] && cp "$MAP_IMAGE" "$RUN_DIR/"
cp "${PACKAGE_DIR}/config/mpc.yaml" "$RUN_DIR/mpc.yaml"

echo "[run_sim] simulator : $SIM_WORKSPACE (ROS_DOMAIN_ID=$ROS_DOMAIN_ID, dds=lo)"
echo "[run_sim] map       : $MAP_YAML"
echo "[run_sim] raceline  : $RACELINE"
echo "[run_sim] spawn     : lateral=$SPAWN_LATERAL m, heading=$SPAWN_HEADING rad"
echo "[run_sim] vehicle   : $([[ "$MATCH_CAR" == true ]] && echo 'matched to our car' || echo 'stock gym f1tenth preset')"
echo "[run_sim] controller: $([[ "$START_CONTROLLER" == true ]] && echo 'Apex MPC' || echo 'external /drive policy')"
echo "[run_sim] run dir   : $RUN_DIR"

LAUNCH_ARGS=(
    "map_yaml:=$MAP_YAML"
    "raceline:=$RACELINE"
    "params_file:=$OVERRIDES"
    "match_car:=$MATCH_CAR"
    "start_controller:=$START_CONTROLLER"
    "spawn_lateral_offset:=$SPAWN_LATERAL"
    "spawn_heading_offset:=$SPAWN_HEADING"
    "rviz:=$RVIZ"
    "raceline_editor:=$EDITOR"
)
[[ -n "$SAMPLES" ]] && LAUNCH_ARGS+=("samples:=$SAMPLES")
[[ -n "$MU" ]] && LAUNCH_ARGS+=("mu:=$MU")
[[ -n "$VEH" ]] && LAUNCH_ARGS+=("veh:=$VEH")

# --- teardown ---------------------------------------------------------------
# Everything this run starts goes into its own process group (setsid), and
# teardown signals only that group.  Never pkill by executable name: another
# session on this machine may be driving its own gym_bridge out of the same
# src_sim install, and a name match would take their run down with ours.
LAUNCH_PID=""
BAG_PID=""
cleanup() {
    trap - EXIT INT TERM
    [[ -n "$BAG_PID" ]] && kill -INT "$BAG_PID" 2>/dev/null || true
    if [[ -n "$LAUNCH_PID" ]]; then
        kill -INT -- "-$LAUNCH_PID" 2>/dev/null || kill -INT "$LAUNCH_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$LAUNCH_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$LAUNCH_PID" 2>/dev/null || true
    fi
    [[ -n "$BAG_PID" ]] && wait "$BAG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$RECORD_BAG" == "1" ]]; then
    TOPICS=(
        /ego_racecar/odom /ego_racecar/steering_state
        /scan /drive /ego_collision
        /mpc/raceline /mpc/prediction /mpc/obstacles
        /mpc/compute_breakdown /mpc/state
        /tf /tf_static /map /rosout
    )
    ros2 bag record -o "$RUN_DIR/rosbag2" "${TOPICS[@]}" \
        >"$RUN_DIR/rosbag_record.log" 2>&1 &
    BAG_PID=$!
    ln -sfn "$RUN_DIR" "${BAG_DIR}/latest"
    echo "[run_sim] recording: $RUN_DIR/rosbag2"
fi

LAUNCH_LOG="$RUN_DIR/launch.log"
# setsid: the launch becomes a session/group leader, so its PID is also its
# PGID and cleanup can signal exactly this run's processes and nothing else.
setsid ros2 launch race_track_mpc sim_mpc_launch.py "${LAUNCH_ARGS[@]}" \
    >"$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
tail -n +1 -F "$LAUNCH_LOG" --pid="$LAUNCH_PID" 2>/dev/null &
TAIL_PID=$!

# --- watchdog ---------------------------------------------------------------
# Stops on the lap target, on a collision, or on the hard timeout, so an
# automated run cannot sit spinning after the car has parked itself.
# A wedged car is the failure mode a lap counter misses: it neither completes
# laps nor trips the gym's collision flag (that is a 5 ms time-to-collision
# test), so it would otherwise sit against a wall until --timeout.  Stopping
# when no new lap has appeared for --stall seconds catches it in ~45 s and
# makes "no laps" a result rather than a hang.
START=$SECONDS
LAST_LAP_AT=$SECONDS
SEEN_LAPS=0
STOP_REASON="launch exited"
while true; do
    sleep 2
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then break; fi
    ELAPSED=$(( SECONDS - START ))
    DONE=$(grep -c -E 'Lap [0-9]+: ' "$LAUNCH_LOG" 2>/dev/null) || DONE=0
    if (( DONE > SEEN_LAPS )); then
        SEEN_LAPS=$DONE
        LAST_LAP_AT=$SECONDS
    fi
    if (( LAPS > 0 && DONE >= LAPS )); then
        STOP_REASON="$DONE lap(s) recorded"
        break
    fi
    if (( SECONDS - LAST_LAP_AT >= STALL )); then
        STOP_REASON="stalled: no lap in ${STALL}s after $DONE lap(s) - wedged, crashed or off the line"
        break
    fi
    if (( ELAPSED >= TIMEOUT )); then
        STOP_REASON="timeout after ${ELAPSED}s"
        break
    fi
done
kill "$TAIL_PID" 2>/dev/null || true
echo "[run_sim] stopping: $STOP_REASON"
echo "[run_sim] finished: $RUN_DIR"
