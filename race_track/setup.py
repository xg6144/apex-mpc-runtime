from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'race_track_mpc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'race_track', 'track_file'),
         glob('track_file/*.csv') + glob('track_file/*.yaml') + glob('track_file/*.pgm')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dongbeen',
    maintainer_email='dongbeen@todo.todo',
    description='Sampling nonlinear MPC raceline tracker for F1TENTH ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller = race_track_mpc.mpc_node:main',
        ],
    },
)
