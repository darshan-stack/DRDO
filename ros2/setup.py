from glob import glob

from setuptools import find_packages, setup

package_name = "ffem_lidar_mapping"
core_packages = find_packages("../src", include=["ffem", "ffem.*"])

setup(
    name=package_name,
    version="0.3.0",
    packages=[package_name, *core_packages],
    package_dir={"ffem": "../src/ffem"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "ffem_node = ffem_lidar_mapping.node:main",
            "ffem_rerun_logger = ffem_lidar_mapping.rerun_logger:main",
        ]
    },
)
