# NuScenesDataExtractor
Data extractor class for extracting data from nuScenes dataset

### Run example
Specify nuScenes ``python-sdk`` path in ``main.py`` and ``nusc_data_extractor.py``, then run:
```
>>> python main.py
```

### Commands
#### Control
```
de_nusc.set_scene(scene_name)
```

```
de_nusc.advance()
```

#### Data extraction
```
lidar_points = de_nusc.get_lidar_pointcloud()
```

```
radar_points, radar_velocities = de_nusc.get_radar_pointcloud()
```

```
radar_points, radar_velocities = de_nusc.get_all_radar_pointclouds()
```

```
camera_image = de_nusc.get_camera_image()
```
