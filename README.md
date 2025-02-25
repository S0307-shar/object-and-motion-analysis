# Soccer Player Analysis

A computer vision system for analyzing soccer matches, detecting players, tracking movements, and identifying key events like fouls and offsides.

## Features

- Player detection and tracking
- Team classification based on jersey colors
- Ball tracking
- Offside detection
- Foul detection using:
  - Player collisions
  - Rapid decelerations
  - Pose analysis for tackles
- Field line detection
- Real-time visualization


## Installation

1. Clone the repository:
  
2. Install required packages:

3. The YOLOv8x model will be downloaded automatically on first run.

## Usage

Run the analysis on a soccer video


### Arguments
- `--video`: Path to input soccer video file
- `--output`: Path for saving the analyzed video (optional)
- `--save-frames`: Save frames containing violations (optional)
- `--output-dir`: Directory to save violation frames (default: "violation_frames")

## Output

The system generates:
- Analyzed video with visualizations
- Detection of offsides and fouls
- Player tracking trails
- Team classifications
- Ball movement tracking

## Visualization Features

- Player bounding boxes colored by team
- Player movement trails
- Ball position and trajectory
- Field line detection
- Offside line visualization
- Foul incident highlighting

## Notes

- For best results, use videos with clear visibility of players and field
- Performance may vary based on video quality and resolution
- Processing speed depends on hardware capabilities


## Author

Sharanya Vadakapur
