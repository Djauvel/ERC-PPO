#!/usr/bin/env python
from exomy_msgs.msg import Actions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import sys
import torch
from gym.spaces import Box
import gym
import math
import time
import numpy as np
sys.path.append('/home/made/Documents/isaac_rover_physical-main/src/exomy/scripts/utils')
from transforms3d.euler import quat2euler

from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.base import Model as BaseModel
import torch.nn as nn

marker_positions = [
            (5.45,3.75),
            (2.14,2.04),
            (-5.02,-4.27),
            (-6.74,-2.70)
        ]

danger_position = [
            (-1.59,1.18)
        ]

def global_to_local(x_g,y_g,robot_x,robot_y,orientation_z):
    delta_x = x_g - robot_x
    delta_y = y_g - robot_y

    cos_theta = np.cos(orientation_z)
    sin_theta = np.sin(orientation_z)

    x_l = (delta_x * cos_theta + delta_y * sin_theta)
    y_l = (-delta_y * sin_theta - delta_y * cos_theta)
    print(f"x_l, y_l: {x_l}, {y_l}")
    return (x_l, y_l)

def get_activation(activation_name):
    """Get the activation function by name."""
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]

class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=8,
        mlp_layers=[512, 256, 128, 64],
        mlp_activation="leaky_relu",
        encoder_input_size=1,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.encoder_input_size is None:
            x = states
        else:
            x = states["states"][:, 0:self.mlp_input_size]
            #print(f"x: {states[:, 0:self.mlp_input_size]}")
            #print(f"encoder input: {states[:, self.mlp_input_size - 1]}")
            encoder_output = self.dense_encoder(states["states"][:, self.mlp_input_size - 1:])
            x = torch.cat([x, encoder_output], dim=1)
            
        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}

class RLModelNode(Node):
    def __init__(self):
        self.drive = True
        self.node_name = 'RLModel_node'
        super().__init__(self.node_name)
     
        self.Tcamera_sub = self.create_subscription(
            Odometry,
            '/camera/odom/sample',
            self.odom_callback,
            1)
        
        self.robot_pub = self.create_publisher(
            Actions,
            '/exomy/Actions',
            1)
            
        #self.observation_space = Box(-math.inf,math.inf,(8,)) #Set observation space size
        self.observation_space = Box(-math.inf,math.inf,(8,))
        self.action_space = Box(-1.0,1.0,(2,)) #Set action space size
        self.goal = np.array([1.0,0.0])
        
        self.observations = torch.ones((1,8))
        self.observations[0,0] = 0#self.prev_action
        self.observations[0,1] = 0

        self.policy = {"policy": GaussianNeuralNetwork(self.observation_space, self.action_space,device=None),
                        "value": None}

        self.weights = torch.load(r"/home/made/Documents/isaac_rover_physical-main/src/exomy/config/HIM.pt",map_location=torch.device('cpu'))["policy"]
        self.policy["policy"].load_state_dict(self.weights)
       
        self.get_logger().info('\t{} STARTED.'.format(self.node_name.upper()))

    def odom_callback(self, msg):
        #self.get_logger().info("odom_callback called")
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.roll, self.pitch, self.yaw = quat2euler([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])

        self.get_logger().info(f'Global orientation: {angle_to_target_observation(self.goal, [self.x,self.y],self.roll)*(1/torch.pi)}')
        self.get_logger().info('Global position: ({:.2f}, {:.2f})'.format(self.x,self.y))

        #try
        # Observation space:
        # actions :[2] 
        ## dist to target
        target_dist = distance_to_target_euclidean(self.goal,[self.x,self.y])
        self.observations[0,2] = target_dist*0.11
        ## heading diff
        self.observations[0,3] = angle_to_target_observation(self.goal, [self.x,self.y],self.roll)*(1/torch.pi)
        ## dist to each of markers
        self.observations[0,4] = dist_to_marker(marker_positions[0],[self.x,self.y])*0.11
        self.observations[0,5] = dist_to_marker(marker_positions[1],[self.x,self.y])*0.11
        self.observations[0,6] = dist_to_marker(marker_positions[2],[self.x,self.y])*0.11
        self.observations[0,7] = dist_to_marker(marker_positions[3],[self.x,self.y])*0.11
        #self.observations[0,8] = dist_to_marker(danger_position[0],[self.x,self.y])*0.11

        #self.get_logger().info(f"Observation Space: {self.observations}")

        #self.observations = torch.ones((1,8))
        self.input_observations = {"states":self.observations}
        start = time.perf_counter()

        try:
            if target_dist > 0.30 and self.drive:
                motorsCom = self.policy["policy"](self.input_observations)
                
                #print(f"MotorsCom {motorsCom}")
                #self.get_logger().info(f"Values: {motorsCom[2]['mean_actions'][0][0]}, {motorsCom[2]['mean_actions'][0][1]}")
                self.get_logger().info(f"MOTORCOM {motorsCom[0]} ")

                #velocity = torch.clip(motorsCom[2]['mean_actions'][0][0], min = -1, max = 1)
                #steering = torch.clip(motorsCom[2]['mean_actions'][0][1], min = -1, max = 1)

                velocity = torch.clip(motorsCom[0][0,0], min = -1, max = 1)
                steering = torch.clip(motorsCom[0][0,1], min = -1, max = 1)

                self.observations[0,0] = velocity
                self.observations[0,1] = steering

                velocity = velocity.item()
                steering = steering.item()

                message = Actions()
                message.lin_vel = float(velocity) * 3
                message.ang_vel = float(steering) * 3

                self.prev_action = message
                self.robot_pub.publish(message)
                finish = time.perf_counter() - start
                #self.get_logger().info(f"FINISH TIME: {finish}")
                self.get_logger().info("----------------------------------------------")
            else:
                if target_dist <= 0.10:
                    self.get_logger().info("REACHED GOAL")
                    self.drive = False
                    message = Actions()
                    message.lin_vel = float(0)
                    message.ang_vel = float(0)
                    self.robot_pub.publish(message)
                    
        except Exception as e:
            self.get_logger().info('\t Error in the Model Node: {}'.format(e))

def angle_to_target_observation(goalPos, robPos, yaw):
    """Calculate the angle to the target."""
    #goalPos will be in global frame when it is passed:
    angle = np.arctan2((goalPos[1]-robPos[1]),(goalPos[0]-robPos[0]))
    
    angle = angle - yaw - torch.pi/2

    if angle < -torch.pi:
        angle += 2*torch.pi

    angle = torch.as_tensor(angle)
    #print(f"ANGLE TO TARGET: {angle}")

    return angle.unsqueeze(-1)

def distance_to_target_euclidean(goalPos, robPos):
    """Calculate the euclidean distance to the target."""
    
    #TODO: fetch actual goal position and use that
    target_vector = torch.Tensor((goalPos[0]-robPos[0],goalPos[1]-robPos[1]))
    distance: torch.Tensor = torch.norm(target_vector, p=2, dim=-1)

    #print(f"DISTANCE TO TARGET: {distance}")
    return distance.unsqueeze(-1)

def dist_to_marker(marker_position, robPos):
    """Calculate the distance to the ArUco Markers in the environment 

    This function uses the array of marker positions to determine and return the distance to any given marker
    """
    robPos = torch.Tensor(robPos)

    # Convert marker_position to a tensor
    marker_position = torch.Tensor(marker_position)

    # Calculate distance
    distance = torch.norm(marker_position - robPos, p=2, dim=-1)

    return distance.unsqueeze(-1)


def main(args=None):
    rclpy.init(args=args)
    node = RLModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()