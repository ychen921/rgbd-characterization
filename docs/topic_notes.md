# Orbbec Camera Topics

The experiment recorder in `scripts/record_experiment.sh` is configured to
record the following ROS 2 topics.

## Depth topics

| Topic | Description |
| --- | --- |
| `/camera/depth/image_raw` | Raw depth image from the camera. |
| `/camera/depth/camera_info` | Depth camera calibration and image geometry. |
| `/camera/depth/metadata` | Metadata associated with the depth stream. |
| `/camera/depth/points` | Point cloud generated from the depth stream. |

## Color topics

| Topic | Description |
| --- | --- |
| `/camera/color/image_raw` | Raw color image from the camera. |
| `/camera/color/camera_info` | Color camera calibration and image geometry. |

## Status and diagnostic topics

| Topic | Description |
| --- | --- |
| `/camera/device_status` | Camera device status information. |
| `/camera/depth_filter_status` | Status of the depth-filter processing. |
| `/camera/depth_filters/status` | Status information published by the depth-filters component. |
| `/diagnostics` | General ROS 2 diagnostic information. |

## Check topic availability

Start the Orbbec camera node, then list the topics that are currently
available:

```bash
ros2 topic list | sort
```

To inspect a topic's message type and publisher information, run:

```bash
ros2 topic info --verbose /camera/depth/image_raw
```

Topic availability depends on the camera launch configuration and enabled
streams or filters. Confirm that the required topics appear before starting an
experiment recording.
