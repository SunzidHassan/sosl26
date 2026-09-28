from setuptools import find_packages, setup

package_name = 'sosl26'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='sunzidhassan@gmail.com',
    description='Semantic odor source localization (olfaction + vision fusion) on TurtleBot4',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sosl_tb4 = sosl26.sOSL_tb4_main:main',
        ],
    },
)
