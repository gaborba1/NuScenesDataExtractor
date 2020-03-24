# NuScenesDataExtractor
Data extractor class for extracting data from nuScenes dataset

### Prerequisites
Download and set up ``nuScenes``. 

### Run example
Specify nuScenes ``python-sdk`` path in ``main.py`` and ``nusc_data_extractor.py``, then run:
```
>>> python main.py
```

### Commands
#### Control
Select the desired scene with
```
scene_name = 'scene-0001'
de_nusc.set_scene(scene_name)
```

For iterating through the time steps of the scene use
```
de_nusc.advance()
```

It is also possible to get a list with all the scenes using
```
scene_list = de_nusc.get_scene_list()
```

#### Data extraction
Extract data of the lidar sensor for current time step:
```
lidar_points = de_nusc.get_lidar_pointcloud()
```

Extract data of one radar sensor for current time step:
```
radar_channel = 'RADAR_FRONT'
radar_points, radar_velocities = de_nusc.get_radar_pointcloud(radar_channel)
```

Extract data of all radar sensors for current time step:
```
radar_points, radar_velocities = de_nusc.get_all_radar_pointclouds()
```

Extract raw image of one camera sensor for current time step:
```
camera_channel = 'CAM_FRONT'
camera_image = de_nusc.get_camera_image(camera_channel)
```

Get annotations from the current scene using
```
bboxes, class_ID, velocity=de_nusc.get_annotations(1) #1 if we want 2D bboxes or 2 if we want 3D bboxes
```

Get annotations for all possible annotations. If a sensor channel (e.g. ``'RADAR_FRONT'``) is specified, just the annotations seen by the sensor are given.
```
de_nusc.get_annotations()
de_nusc.get_annotations('CAM_FRONT')
de_nusc.get_annotations('RADAR_FRONT')
```

### Saving the data
It is possible to save the current LiDAR point cloud into an image of a fixed maximum axis dimmension as well as the labels of that image using:
```
#Save the image with the LiDAR points, the maximum axis value and the name used to save the file. The labels will also be saved
de_nusc.save_figure_and_labels("LIDAR_TOP", 40, "image0000")
```
