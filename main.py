NUSCENES_SDK_PATH = '/python-sdk'
NUSCENES_DATABASE_PATH = '/python-sdk/data/nuscenes'

import sys
sys.path.append(NUSCENES_SDK_PATH)
import numpy as np
from nuscenes.nuscenes import NuScenes
from nusc_data_extractor import NuScenesDataExtractor, draw2DBboxes, draw_rect, draw3DBboxes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nusc = NuScenes(dataroot=NUSCENES_DATABASE_PATH)
    de_nusc = NuScenesDataExtractor(nusc=nusc)

    scene_list = de_nusc.get_scene_list()
    # select scene
    scene_name = scene_list[0]
    de_nusc.set_scene(scene_name)

    # extract data
    lidar_points = de_nusc.get_lidar_pointcloud() #each component (x,y,z) is a different row

    bboxes, class_ID, velocity=de_nusc.get_annotations(1) #1 if we want 2D bboxes or 2 if we want 3D bboxes

    #Save the image with the LiDAR points, the maximum axis value and the name used to save the file. The labels will also be saved
    de_nusc.save_figure_and_labels("LIDAR_TOP", 40, "image0000")

    #########################
    #### OTHER OPTIONS ######
    #########################
    
    #-----PLOT LIDAR DATA-------
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(lidar_points[0,:], lidar_points[1, :], lidar_points[2, :], s=1)

    fig = plt.figure()
    plt.scatter(lidar_points[0,:], lidar_points[1, :],c=np.sqrt(lidar_points[0,:] * lidar_points[0,:] + lidar_points[1,:] * lidar_points[1,:]), s=1)

    #-----PLOT LIDAR DATA WITH BBOXES--------
    fig = plt.figure()
    plt.scatter(lidar_points[0, :], lidar_points[1, :],c=np.sqrt(lidar_points[0, :] * lidar_points[0, :] + lidar_points[1, :] * lidar_points[1, :]), s=1)
    draw2DBboxes(bboxes)
    
    #------GET RADAR POINTS-----
    radar_points, radar_velocities = de_nusc.get_radar_pointcloud()
    radar_points, radar_velocities = de_nusc.get_all_radar_pointclouds()
    camera_image = de_nusc.get_camera_image()
     
    #-------advance timestep---------
    de_nusc.advance()

