from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'new_bimanual_pkg'

data_files = [
    # ROS2 메타
    (f'share/ament_index/resource_index/packages', [f'resource/{package_name}']),
    (f'share/{package_name}', ['package.xml']),
    # 최상위 MJCF (bimanual.xml 등)
    (f'share/{package_name}/mujoco_models',
        glob('mujoco_models/*.xml') + glob('mujoco_models/*.[ux]rdf')),
    # launch
    (f'share/{package_name}/launch', glob('launch/*.py')),
]

# === mujoco_models/** 전체 구조 보존 설치 (assets, furniture_sim 등 모두) ===
for root, _, files in os.walk('mujoco_models'):
    if not files:
        continue
    rel = os.path.relpath(root, 'mujoco_models')  # e.g. assets/ffw_bg2, furniture_sim/counters
    install_root = os.path.join('share', package_name, 'mujoco_models', rel)
    src_files = [os.path.join(root, f) for f in files]
    data_files.append((install_root, src_files))

# (선택) 별도 meshes/ 디렉토리를 MJCF가 직접 참조한다면 같이 넣기
if os.path.isdir('meshes'):
    for root, _, files in os.walk('meshes'):
        if not files:
            continue
        rel = os.path.relpath(root, 'meshes')  # e.g. ffw_bg2, common/rh_p12_rn_a
        install_root = os.path.join('share', package_name, 'meshes', rel)
        src_files = [os.path.join(root, f) for f in files]
        data_files.append((install_root, src_files))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'mujoco'],
    zip_safe=True,
    maintainer='gaga',
    maintainer_email='fhmpsy@gmail.com',
    description='MuJoCo + ROS2 interface',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_node = new_bimanual_pkg.start_node:main',
            'path_node = new_bimanual_pkg.path_master_node:main',
            'basic_node = new_bimanual_pkg.basic_spawn_node:main',
        ],
    },
)
