NUSCENES_SDK_PATH = '/python-sdk'
NUSCENES_DATABASE_PATH = '/python-sdk/data/nuscenes'

import sys
import os.path as osp
sys.path.append(NUSCENES_SDK_PATH)
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils  import view_points, box_in_image, BoxVisibility, transform_matrix
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
from coco_creator import label_creation, create_COCO_individual_labels
import json

class NuScenesDataExtractor:
    """ Helper class to extract raw data from the database """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.radar_sensor_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.camera_sensor_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
        print(" [*] NuScenesDataExtractor initiated.")

    def set_scene(self, scene):
        """
        Selecting a scene
        Author: Gabor
        """
        self.scene_name = scene
        scene_token = self.nusc.field2token('scene', 'name', self.scene_name)[0]
        self.scene = self.nusc.get('scene', scene_token)        
        self.sample_token = self.scene['first_sample_token']
        self.sample = self.nusc.get('sample', self.sample_token)
        self.timestep = 0
        self.points_lidar = []
        print(" [*] Scene set to " + self.scene_name)
        print(" [*] Timestep 0")

    def advance(self, n=1):
        """ 
        Advances one time step in the scene.
        Author: Gabor
        :param n: number of timesteps to advance
        """
        if not hasattr(self, 'sample'):
            print(" [!] Can't advance, if no scene was selected before.")
            return
        try:
            self.sample = self.nusc.get('sample', self.sample['next'])
            self.sample_token = self.sample['token']
            self.timestep = self.timestep + 1
            print(" [*] Timestep " + str(self.timestep))
            return True

        except:
            return False
            pass

    def get_camera_image(self, channel='CAM_FRONT'):
        """ 
        Extracting camera image for current timestep in current scene for specified camera channel.
        Author: Gabor
        :param channel: Camera channel selection. Front camera as default
        :return camera image as 
        """

        # Check for correct camera channel selection
        assert channel in self.camera_sensor_channels, " [!] Camera channel \"{}\" not found.".format(channel)
        
        # Select sensor data record
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        filename = osp.join(self.nusc.dataroot, sd_record['filename'])
        image = Image.open(filename)
        
        print(" [*] Camera image extracted.")
        return image

    def get_lidar_pointcloud(self):
        """ 
        Extracting lidar pointcloud for current timestep in current scene
        Author: Javier
        :return Point cloud [Position(x,y,z) X n_points] 
        """

        # Select sensor data record
        channel = 'LIDAR_TOP'
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)

        # Get aggregated point cloud in lidar frame.
        chan = sd_record['channel']
        pc, times = LidarPointCloud.from_file_multisweep(self.nusc, self.sample, chan, channel, nsweeps=1)

        #calibration of LiDAR points
        ref_sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                      rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        self.points_lidar = points
        return points

    def get_all_radar_pointclouds(self):
        """ 
        Extracting radar detection pointclouds with velocities for current timestep in current scene of all radar sensors.
        Author: Gabor
        :return (Point cloud [n_radars, Position(x,y,z), n_points], Point cloud [n_radars, Velocity(x,y,z), n_points])
        """

        points = []
        points_vel = []

        for channel in self.radar_sensor_channels:
            p,v = self.get_radar_pointcloud(channel=channel)
            points = [points, p]
            points_vel = [points_vel, v]

        print(" [*] All radar point clouds extracted.")
        return points, points_vel

    def get_radar_pointcloud(self, channel='RADAR_FRONT'):
        """ 
        Extracting radar detection pointcloud with velocities for current timestep in current scene for specified radar channel.
        Author: Gabor
        :param channel: Radar channel selection. Front radar as default
        :return (Point cloud [Position(x,y,z) X n_points], Point cloud [Velocity(x,y,z) X n_points])
        """

        # Check for correct radar channel selection
        assert channel in self.radar_sensor_channels, " [!] Radar channel \"{}\" not found.".format(channel)
        
        # Select sensor data record
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)
        lidar_token = self.sample['data']['LIDAR_TOP']

        # The point cloud is transformed to the lidar frame for visualization purposes.
        ref_chan = 'LIDAR_TOP'
        pc, times = RadarPointCloud.from_file_multisweep(self.nusc, self.sample, channel, ref_chan, nsweeps=1)

        # Transform radar velocities (x is front, y is left), as these are not transformed when loading the point
        # cloud.
        radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        lidar_sd_record = self.nusc.get('sample_data', lidar_token)
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        velocities = pc.points[8:10, :] # Compensated velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
        velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(lidar_cs_record['rotation']).rotation_matrix.T, velocities)
        velocities[2, :] = np.zeros(pc.points.shape[1])

        # Show point cloud.
        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        points_vel = view_points(pc.points[:3, :] + velocities, np.eye(4), normalize=False)

        print(" [*] Radar point cloud extracted from channel \"" + channel + "\". Shape: " + str(points.shape))
        return points, points_vel

    def get_annotations(self, option):
        '''
        Return the annotations of the scene
        Author: Javier
        :parameter: option = 1 if we want 2D bboxes or 2 if we want 3D bboxes
        :return: bboxes, classification, velocity
        '''

        # Select sensor data record
        channel = 'LIDAR_TOP'
        sample_data_token = self.sample['data'][channel]
        sd_record = self.nusc.get('sample_data', sample_data_token)

        # Get aggregated point cloud in lidar frame.
        chan = sd_record['channel']

        #ref_sd_token it is sample_data_token
        ref_sd_record = self.nusc.get('sample_data', sample_data_token)

        _, boxes, _ = self.nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY, use_flat_vehicle_coordinates=True)

        bboxes = []
        class_ID = []
        velocity_list =[]
        for box in boxes:
            corners = box.corners(1)

            if option == 1:
                corners_2=np.zeros((2,4))
                corners_2[0, 0] = corners[0, 2]
                corners_2[1, 0] = corners[1, 2]
                corners_2[0, 1] = corners[0, 3]
                corners_2[1, 1] = corners[1, 3]

                corners_2[0, 2] = corners[0, 7]
                corners_2[1, 2] = corners[1, 7]
                corners_2[0, 3] = corners[0, 6]
                corners_2[1, 3] = corners[1, 6]

                corners=corners_2.copy()

            category = box.name
            velocity = box.velocity
            class_ID.append(category)
            velocity_list.append(velocity)
            bboxes.append(corners)

        return bboxes, class_ID, velocity_list

    def save_figure_and_labels(self, channel, axes_limit: float = 40, save_name: str = "defaut_name"):
         '''
        Save the figure with the LiDAR points and the labels in separated files (image for LiDAR and .txt for labels)
        Author: Javier
        '''
        point_scale = 0.2 if channel == 'LIDAR_TOP' else 3.0

        #check if we colleted the lidar points, if not collect them now
        if self.points_lidar ==[]:
            points = self.get_lidar_pointcloud()
        else:
            points = self.points_lidar

        #save the image without any frame or axis
        fig = plt.figure(figsize=(9, 9), frameon=False)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.scatter(points[0, :], points[1, :], c=np.sqrt(points[0, :] * points[0, :] + points[1, :] * points[1, :]),
                   s=point_scale)
        fig.savefig("Dataset/" + save_name + ".jpg")
        plt.close('all')

        #Save the labels to match the new dimensions of the image
        image = cv2.imread("Dataset/" + save_name + ".jpg")

        #now the origin of the coordinate system is the top left instead of middle left
        height_image = image.shape[0]
        width_image = image.shape[1]

        #transform coordinates to pixels
        bboxes, class_ID, velocity_list=self.get_annotations(1)

        # get the labels and keep only the ones inside the axis
        new_bboxes = []
        new_class = []
        new_speed = []
        for i in range(0, len(bboxes)):
            box = bboxes[i]
            class_box = class_ID[i]
            speed_box = velocity_list[i]
            if self.check_corners_in_range(box, axes_limit, axes_limit):
                new_bboxes.append(box)
                new_class.append(class_box)
                new_speed.append(speed_box)

        origi_height = 2 * axes_limit
        origi_width = 2 * axes_limit

        calib_bboxes=[]
        for i in range(0, len(new_bboxes)):
            new_coord = np.zeros((2, 4))

            new_coord[0, 0] = (new_bboxes[i][0,0] + axes_limit) * (width_image / origi_width)
            new_coord[1, 0] = height_image-((new_bboxes[i][1,0] + axes_limit) * (height_image/origi_height))

            new_coord[0, 1] = (new_bboxes[i][0, 1] + axes_limit) * (width_image / origi_width)
            new_coord[1, 1] = height_image-((new_bboxes[i][1, 1] + axes_limit) * (height_image/origi_height))

            new_coord[0, 2] = (new_bboxes[i][0, 2] + axes_limit) * (width_image / origi_width)
            new_coord[1, 2] = height_image-((new_bboxes[i][1, 2] + axes_limit) * (height_image/origi_height))

            new_coord[0, 3] = (new_bboxes[i][0, 3] + axes_limit) * (width_image / origi_width)
            new_coord[1, 3] = height_image-((new_bboxes[i][1, 3] + axes_limit) * (height_image/origi_height))

            calib_bboxes.append(new_coord)

        # Saving the labels with coco format v2
        label_coco = create_COCO_individual_labels(calib_bboxes, new_class, save_name + ".jpg", height_image, width_image)

        with open("Dataset/" + save_name  + '.json','w') as f:
            for item in label_coco:
                f.write("%s" % item)
            f.close()

    def check_corners_in_range(self, corners, x_lim, y_lim):
        '''
        Check if the corners of the labels are inside the limits for the axis
        Author: Javier
        '''
        if corners.max()< x_lim and corners.min()> (-1*x_lim):
            valid = True
        else:
            valid = False

        return valid

    def get_scene_list(self):
        '''
        Return the scene list in the database
        Author: Javier
        '''
        scene_names = []
        scene_list = self.nusc.scene
        for scene in scene_list:
            scene_names.append(scene["name"])
        return scene_names


def draw_rect(selected_corners):
    '''
    Draw retangle based on corners. From nuscenes
    :param selected_corners:
    :return:
    '''
    prev = selected_corners[-1]
    for corner in selected_corners:
        plt.plot([prev[0], corner[0]], [prev[1], corner[1]], color=(0, 0, 0), linewidth=0.5)
        prev = corner


def draw2DBboxes(corners):
    '''
    Feed corners to the drawing function in the correct format to draw 2D bboxes
    :param corners:
    :return:
    '''
    for i in range(0, len(corners)):
        draw_rect(corners[i].T[:4])

def draw3DBboxes(corners):
    # Draw the sides of the bboxes in 3D
    for i in range(4):
        plt.plot([corners.T[i][0], corners.T[i + 4][0]],
                  [corners.T[i][1], corners.T[i + 4][1]],
                  color=(0,0,0), linewidth=0.5)

    # Draw front and back face of the 3D bbox
    draw_rect(corners.T[:4], (0,0,0))
    draw_rect(corners.T[4:], (0,0,0))
