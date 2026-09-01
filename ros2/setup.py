from setuptools import setup
from glob import glob

package_name = 'ffem_lidar_mapping'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    entry_points={'console_scripts': ['ffem_node = ffem_lidar_mapping.node:main']},
)
