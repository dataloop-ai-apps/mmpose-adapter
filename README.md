# MMPose Adapter

A Dataloop model adapter for human pose estimation using MMPose and MMDetection. This adapter combines object detection (RTMDet) with pose estimation (HRNet) to detect and track human keypoints in images.

## Overview

This project provides a seamless integration between Dataloop's annotation platform and OpenMMLab's MMPose framework for human pose estimation. The adapter uses a two-stage approach:

1. **Object Detection**: RTMDet-Tiny model detects human bounding boxes
2. **Pose Estimation**: HRNet-W48 model estimates 17 COCO keypoints for each detected person

## Features

- **Automatic Model Download**: Downloads required model files at runtime using `mim`
- **GPU/CPU Support**: Automatically detects and uses available CUDA devices
- **Batch Processing**: Efficiently processes multiple images in batches
- **Dataloop Integration**: Seamlessly integrates with Dataloop's annotation platform
- **Configurable**: Supports custom model configurations and device settings

## Architecture

The adapter implements a two-stage pipeline:

```
Image → RTMDet (Person Detection) → HRNet (Pose Estimation) → COCO Keypoints
```

### Models Used

- **Detection Model**: `rtmdet_tiny_8xb32-300e_coco`
  - Lightweight person detector
  - Config: `rtmdet_tiny_8xb32-300e_coco.py`
  - Checkpoint: `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`

- **Pose Model**: `td-hm_hrnet-w48_8xb32-210e_coco-256x192`
  - Top-down pose estimator
  - Config: `td-hm_hrnet-w48_8xb32-210e_coco-256x192.py`
  - Checkpoint: `td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth`


## Configuration

The adapter supports the following configuration parameters:

### Model Configuration
- `model_name`: Model identifier (default: pose model name)
- `config_file`: Path to model configuration file
- `checkpoint_file`: Path to model checkpoint file
- `device`: Device to run inference on (`cuda:0`, `cpu`, etc.)

### Pose Estimation
- **Keypoints**: 17 COCO keypoints
- **Input Resolution**: 256x192
- **Top-down Approach**: Uses detected bounding boxes

## Usage

**Process:**
1. Detects persons using RTMDet
2. Filters detections by confidence and class
3. Applies NMS to remove overlapping boxes
4. Estimates pose for each person using HRNet
5. Creates Dataloop pose annotations with keypoints

## Output Format

The adapter creates pose annotations with the following structure:

- **Parent Annotation**: Pose annotation with label "mmpose-model"
- **Child Annotations**: Point annotations for each keypoint
  - **Coordinates**: (x, y) pixel coordinates
  - **Labels**: COCO keypoint names (nose, left_eye, right_eye, etc.)
  - **Confidence**: Model confidence scores
  - **Metadata**: Model information (name, ID, confidence)

### COCO Keypoints (17 points)
1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle
