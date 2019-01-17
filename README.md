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

Get annotations for all possible annotations. If a sensor channel (e.g. ``'RADAR_FRONT'``) is specified, just the annotations seen by the sensor are given.
```
de_nusc.get_annotations()
de_nusc.get_annotations('CAM_FRONT')
de_nusc.get_annotations('RADAR_FRONT')
```
