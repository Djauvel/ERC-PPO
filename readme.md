# ExoMy - Software Repository
Repository for the ROS software that runs on the AAU modified ExoMy Rover.

This version of the system runs on a NVIDIA Xavier, using Ubuntu 20.04, jetpack 5.0 and ROS Foxy.

<img src="https://github.com/Djauvel/ERC-PPO/assets/93127878/8527d82a-85d8-4e4e-b548-e727591d2e95" width="300" height="400">


To launch the camera nodes, open a terminal and run:
```
ros2 launch realsense2_camera rs_launch.py
```
Documentation about the RealSense ROS nodes can be found at [their github](https://github.com/IntelRealSense/realsense-ros/tree/ros2-legacy)

Then, run the system in a second terminal:
```
ros2 launch exomy exomy.launch.py
```

Finally, run the RL node using:
```
ros2 run exomy RL_node
```

# Dependencies
This ROS system depends on certain ROS2 packages, we have not been able to install compiled versions on the ARM64 processer, therefore we have compiled the packages ourselves. 
```
ros2_numpy
realsense-ros
sensor_msgs
rviz2 (for debugging)
```
These packages where cloned from their respected Github repositories and built using colcon. 

Python dependencies:
```
scipy
numpy
math
time
torch
csv
adafruit_pca9685
gym 
skrl
```
# Making changes
The ROS system has 4 main nodes:
```
Camera transformation node
RL node
Kinematics node
Motor node
```
The nodes are located in `~/ExoMy_Software/exomy`

The nodes are written in python, and have supportive scripts located in `~/ExoMy_Software/scripts/utils`

Further is nodes called joy and gamepad_parser, these are for controlling the rover with a controller, but is disabled at default.
If you wish to include these, edit the launch file located at `~/ExoMy_Software/launch/exomy.launch.py`

When making changes to either the nodes or subscripts, the ROS system has to be recompiled, do this by opening a terminal in `~/ExoMy_Software` and running:
```
colcon build
```

If you use our Xavier, we have used virtual python environments, so if you have problems compiling or other python problems you can try switching between then with:
```
pyenv local system
pyenv local 3.8.0
```
These commands switch between the two environments on the system.

# Tips
For controlling the robot wirelessly, we have used SSH to control and program the rover. The Eduroam AAU network does not allow communication directly between devices however. Either use a free VPN like tailscale or use a router to host a private network that allows SSH.

# Other questions
For other questions, contact 

- Peter: ppje21@student.aau.dk/petaarz@hotmail.com
