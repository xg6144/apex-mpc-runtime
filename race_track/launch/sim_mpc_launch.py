"""Run the apex MPC against the f1tenth_gym_ros physics simulator in src_sim.

The simulator is *not* reimplemented here: this launch file includes
f1tenth_gym_ros' own ``gym_bridge_launch.py`` so the controller always faces
the exact plant (single-track dynamics, RK4 at 100 Hz, Pacejka-style tyres)
that workspace builds.

That bridge launch takes only ``config``/``num_agent``/foxglove arguments -
map, start pose and vehicle parameters all come from the sim config file - so
this launch writes a temporary sim config derived from the installed one, with
the competition map, the raceline's first waypoint as the start pose, and the
vehicle-parameter overrides that make the simulated car match ours.
"""

import math
import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from race_track_mpc.mpc_core import Raceline


# Overrides applied on top of f1tenth_gym's 'f1tenth' preset so the simulated
# plant matches the car we actually race.  Everything else (mass, inertia,
# tyre stiffness, geometry) is already this chassis.
#
#   mu       the preset's 1.0489 is 10.29 m/s^2 of grip at 1 g.  Our car's
#            validated envelope is the 6.0 m/s^2 in gb_optimizer's ggv.csv,
#            and the 2026-08-22 bags never sustained more, so 0.612 * 9.81 =
#            6.0 keeps the simulator from flattering a raceline the car cannot
#            hold.  This is the single biggest sim-to-car knob.
#   s_min    the servo's real travel, in physical WHEEL angle: the /drive
#   s_max    command range -0.48 / +0.51 times the measured steering_gain
#            1.0956 (2026-08-24 fit, bags mpc_20260824_002320/_002707) is
#            -0.5259 / +0.5588 rad.  The preset's +/-0.52 is close but
#            asymmetric the wrong way.
#   sv_*     max_servo_speed 4.45 command rad/s times 1.0956 is 4.8754 rad/s
#            at the wheel.
VEHICLE_PARAM_OVERRIDES = {
    'mu': 0.612,          # overridable at launch with mu:=<value>
    's_min': -0.5259,
    's_max': 0.5588,
    'sv_min': -4.8754,
    'sv_max': 4.8754,
}

REAL_STEERING_COMMAND = {
    'steering_command_gain': 1.0956,
    'steering_command_min': -0.48,
    'steering_command_max': 0.51,
    # First-order servo response measured at 0.03-0.05 s on the car.  Without
    # it the bridge's pure slew limiter flattered tunes that limit-cycled on
    # the real servo (bags mpc_20260823_224139, mpc_20260824_0027*).
    'steering_time_constant': 0.04,
}


def _required_file(path, label):
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError('{} not found: {}'.format(label, resolved))
    return resolved


def _write_sim_config(base_config, map_yaml, ego_pose, static_obstacles,
                      apply_overrides, overrides=None):
    """Derive a sim config from the installed one and return its path."""
    with open(base_config, 'r') as handle:
        config = yaml.safe_load(handle)
    params = config['bridge']['ros__parameters']
    params['map_path'] = map_yaml
    params['num_agent'] = 1
    params['sx'], params['sy'], params['stheta'] = ego_pose
    params['static_obstacles_enabled'] = bool(static_obstacles)
    if apply_overrides:
        overrides = overrides or VEHICLE_PARAM_OVERRIDES
        params['vehicle_param_names'] = list(overrides)
        params['vehicle_param_values'] = [float(v) for v in overrides.values()]
        params.update(REAL_STEERING_COMMAND)
    else:
        # --stock-vehicle must preserve the original gym /drive contract.
        params['vehicle_param_names'] = ['']
        params['vehicle_param_values'] = [0.0]
        params['steering_command_gain'] = 1.0
        params['steering_command_min'] = -0.52
        params['steering_command_max'] = 0.52
        params['steering_time_constant'] = 0.0
    handle = tempfile.NamedTemporaryFile(
        mode='w', prefix='apex_sim_', suffix='.yaml', delete=False)
    with handle:
        yaml.safe_dump(config, handle, default_flow_style=False)
    return handle.name


def _launch_setup(context):
    mpc_share = get_package_share_directory('race_track_mpc')
    simulator_share = get_package_share_directory('f1tenth_gym_ros')

    map_yaml = _required_file(
        LaunchConfiguration('map_yaml').perform(context), 'competition map')
    raceline_path = _required_file(
        LaunchConfiguration('raceline').perform(context), 'competition raceline')
    sim_launch_path = _required_file(
        os.path.join(simulator_share, 'launch', 'gym_bridge_launch.py'),
        'installed simulator launch')

    base_config = LaunchConfiguration('sim_config').perform(context)
    if not os.path.isabs(base_config):
        base_config = os.path.join(simulator_share, 'config', base_config)
    base_config = _required_file(base_config, 'simulator config')

    raceline = Raceline.from_csv(raceline_path)
    base_heading = float(raceline.heading[0])
    lateral_offset = float(
        LaunchConfiguration('spawn_lateral_offset').perform(context))
    heading_offset = float(
        LaunchConfiguration('spawn_heading_offset').perform(context))
    ego_pose = (
        float(raceline.x[0]) - math.sin(base_heading) * lateral_offset,
        float(raceline.y[0]) + math.cos(base_heading) * lateral_offset,
        base_heading + heading_offset,
    )
    match_car = LaunchConfiguration('match_car').perform(context).lower() \
        not in ('false', '0', 'no')
    overrides = dict(VEHICLE_PARAM_OVERRIDES)
    mu = LaunchConfiguration('mu').perform(context).strip()
    if mu:
        overrides['mu'] = float(mu)
    # Anything else on the plant, for robustness sweeps: veh:=sv_max=2.0,m=4.1
    for item in LaunchConfiguration('veh').perform(context).split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('veh expects key=value pairs, got %r' % item)
        key, value = item.split('=', 1)
        overrides[key.strip()] = float(value)
    sim_config = _write_sim_config(
        base_config, map_yaml, ego_pose,
        LaunchConfiguration('static_obstacles').perform(context).lower()
        in ('true', '1', 'yes'),
        match_car, overrides)
    steering_description = (
        'wheel=1.0956*command, command=[-0.48, 0.51]'
        if match_car else
        'wheel=command, command=[-0.52, 0.52] (stock gym)'
    )

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            'config': sim_config,
            'num_agent': '1',
            'enable_foxglove_bridge': 'false',
            'open_foxglove': 'false',
        }.items(),
    )

    controller_parameters = [
        os.path.join(mpc_share, 'config', 'mpc.yaml'),
        os.path.join(mpc_share, 'config', 'mpc_sim.yaml'),
    ]
    extra_params = LaunchConfiguration('params_file').perform(context).strip()
    if extra_params:
        controller_parameters.append(
            _required_file(extra_params, 'extra parameter file'))
    controller_parameters.append({
        'raceline_path': raceline_path,
        'map_yaml_path': map_yaml,
        'enable_manual_override': False,
        'enable_imu': False,
        # Gym publishes odometry at the physics rate (~160 Hz).  Keep the MPC
        # on its configured 50 Hz timer even when this launch is invoked
        # directly without run_sim.sh's generated parameter file.
        'control_on_odometry': False,
        'enable_raceline_editor': ParameterValue(
            LaunchConfiguration('raceline_editor'), value_type=bool),
        # Never overwrite the line under test; editor saves land beside it.
        'raceline_editor_save_path': os.path.splitext(raceline_path)[0]
                                     + '_sim_edited.csv',
    })
    speed_scale = LaunchConfiguration('speed_scale').perform(context).strip()
    if speed_scale:
        controller_parameters.append({'speed_scale': float(speed_scale)})
    samples = LaunchConfiguration('samples').perform(context).strip()
    if samples:
        controller_parameters.append({'samples': int(samples)})

    controller = Node(
        package='race_track_mpc',
        executable='mpc_controller',
        name='mpc_controller',
        output='screen',
        parameters=controller_parameters,
        condition=IfCondition(LaunchConfiguration('start_controller')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    return [
        LogInfo(msg='Physics simulator: ' + simulator_share),
        LogInfo(msg='Generated sim config: ' + sim_config),
        LogInfo(msg='Map: ' + map_yaml),
        LogInfo(msg='Raceline: ' + raceline_path),
        LogInfo(msg='Start pose: {:.3f}, {:.3f}, {:.3f}'.format(*ego_pose)),
        LogInfo(msg='Vehicle matched to the car: {} (mu {:.3f} = {:.2f} m/s^2)'
                .format(match_car, overrides['mu'], overrides['mu'] * 9.81)),
        LogInfo(msg='Vehicle parameters: '
                + ', '.join('%s=%g' % kv for kv in sorted(overrides.items()))),
        LogInfo(msg='Steering interface: ' + steering_description),
        simulator,
        controller,
        rviz,
    ]


def generate_launch_description():
    mpc_share = get_package_share_directory('race_track_mpc')
    default_map = os.path.join(
        mpc_share, 'race_track', 'track_file', 'ifac_roboracer.yaml')
    default_raceline = os.path.join(
        mpc_share, 'race_track', 'track_file',
        'ifac_roboracer_raceline_apex.csv')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_yaml', default_value=default_map,
            description='Competition map YAML (nav2 format).'),
        DeclareLaunchArgument(
            'raceline', default_value=default_raceline,
            description='Raceline CSV the MPC follows.'),
        DeclareLaunchArgument(
            'sim_config', default_value='sim.yaml',
            description='Base f1tenth_gym_ros config to derive the run config from.'),
        DeclareLaunchArgument(
            'params_file', default_value='',
            description='Extra MPC parameter file applied after mpc_sim.yaml.'),
        DeclareLaunchArgument(
            'match_car', default_value='true',
            description='Override the gym vehicle preset with our measured '
                        'grip and servo limits.'),
        DeclareLaunchArgument(
            'mu', default_value='',
            description='Override the simulated tyre friction coefficient. '
                        'Empty keeps 0.612 (6.0 m/s^2); grip in m/s^2 is '
                        'mu * 9.81.'),
        DeclareLaunchArgument(
            'veh', default_value='',
            description='Extra gym vehicle-parameter overrides as a '
                        'comma-separated key=value list, e.g. sv_max=2.0.'),
        DeclareLaunchArgument(
            'rviz', default_value='true', description='Start RViz2.'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(mpc_share, 'rviz', 'raceline_editor.rviz'),
            description='RViz2 configuration.'),
        DeclareLaunchArgument(
            'raceline_editor', default_value='false',
            description='Enable interactive raceline editing.'),
        DeclareLaunchArgument(
            'start_controller', default_value='true',
            description='Start the MPC controller.'),
        DeclareLaunchArgument(
            'static_obstacles', default_value='false',
            description='Allow RViz clicks to place simulator obstacles.'),
        DeclareLaunchArgument(
            'speed_scale', default_value='',
            description='Optional MPC speed scale override.'),
        DeclareLaunchArgument(
            'spawn_lateral_offset', default_value='0.0',
            description='Initial lateral offset in metres; positive is left '
                        'of the raceline heading.'),
        DeclareLaunchArgument(
            'spawn_heading_offset', default_value='0.0',
            description='Initial heading offset in radians.'),
        DeclareLaunchArgument(
            'samples', default_value='',
            description='Optional CEM sample-count override for A/B tests.'),
        OpaqueFunction(function=_launch_setup),
    ])
