from setuptools import find_packages, setup

package_name = 'online_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=[
        'setuptools',
        'numpy>=1.24,<1.25',
        'scipy>=1.11,<1.12',
        'scikit-learn>=1.2,<1.4',
        'open3d==0.17.0'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='t.rehbronn@irt.rwth-aachen.de',
    description='Online tool for LiDAR sensor extrinsic calibration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "main = src.main_ros:main"
        ],
    },
)
