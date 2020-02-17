NUSCENES_SDK_PATH = '/python-sdk'
NUSCENES_DATABASE_PATH = '/python-sdk/data/nuscenes'

import sys
sys.path.append(NUSCENES_SDK_PATH)
import numpy as np

from nuscenes.nuscenes import NuScenes
from nusc_data_extractor import NuScenesDataExtractor

""" Start """

nusc = NuScenes(dataroot=NUSCENES_DATABASE_PATH)
de_nusc = NuScenesDataExtractor(nusc=nusc)

# select scene
scene_name = 'scene-0123'
de_nusc.set_scene(scene_name)

# extract data
lidar_points = de_nusc.get_lidar_pointcloud()
radar_points, radar_velocities = de_nusc.get_radar_pointcloud()
radar_points, radar_velocities = de_nusc.get_all_radar_pointclouds()
camera_image = de_nusc.get_camera_image()

# advance timestep
de_nusc.advance()


print(" [*] Done.")









''' manual sample extraction

SCENE_NAME = 'scene-0123'
scene_token = nusc.field2token('scene', 'name', SCENE_NAME)[0]
scene = nusc.get('scene', scene_token)
sample = nusc.get('sample', scene['first_sample_token'])

'''