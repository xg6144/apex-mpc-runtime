# Apex MPC

F1TENTH 차량용 ROS 2 샘플링 기반 비선형 MPC입니다. 레이싱 라인을 추종하고 LiDAR로 추적한 장애물을 회피합니다. 벽이나 장애물과 겹치는 예측 궤적을 주행 후보에서 제외하며, 안전한 후보가 없으면 `/drive`에 정지 명령을 보냅니다.

## 빌드

Linux에서 ROS 2 Humble과 `colcon`, `rosdep`을 준비한 뒤 저장소 루트에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths race_track --ignore-src -r -y
colcon build --symlink-install --packages-select race_track_mpc
source install/setup.bash
```

CPU 가속에는 Numba가 필요합니다. 가속을 사용할 수 없는 환경에서는 NumPy로 실행합니다.

## 실행

오도메트리와 LiDAR를 발행하는 ROS 2 환경에 MPC만 연결하려면 다음 명령을 실행합니다. 기본 입력은 `/ego_racecar/odom`, `/scan`이고 출력은 `/drive`입니다. 기본 지도와 레이싱 라인이 함께 설치됩니다.

```bash
ros2 launch race_track_mpc mpc_launch.py
```

실차에서는 VESC, LiDAR, 위치 추정, 수동·자율 명령을 전환하는 mux를 먼저 실행합니다. 실차 스크립트는 `/pf/pose/odom`과 `/teleop`을 사용하고, 다른 제어기가 `/drive`를 발행 중인지 검사합니다.

```bash
F1TENTH_VEHICLE_WORKSPACE="$HOME/f1tenth_ws" \
  bash race_track/scripts/run_car.sh --speed 0.7 --no-bag
```

실차 설정의 회생제동 진입 명령은 [VESC 패치](race_track/vehicle_patches/vesc_motor_brake.patch)의 기본 문턱값과 맞춰져 있습니다. 패치는 차량 쪽 `ackermann_to_vesc`에 별도로 적용하고 빌드하는 통합 코드입니다.

물리 시뮬레이션을 실행하려면 `f1tenth_gym_ros`를 별도 워크스페이스에서 빌드합니다. 이 프로젝트의 시뮬레이션 시작 파일은 `config` 인자를 받는 `gym_bridge_launch.py`를 사용합니다. 실차와 같은 조향 변환 및 차량 파라미터 적용에는 해당 옵션을 지원하는 브리지가 필요합니다.

```bash
F1TENTH_SIM_WORKSPACE="$HOME/src_sim" \
  bash race_track/scripts/run_sim.sh --headless --no-bag
```

두 스크립트 모두 `--map`과 `--raceline`으로 다른 코스를 지정할 수 있습니다.
