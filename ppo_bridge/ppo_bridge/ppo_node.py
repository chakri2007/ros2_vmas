#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import torch
import os
from functools import partial
from typing import List, Dict

class PPOBridge(Node):
    def __init__(self):
        super().__init__('ppo_bridge')

        # --- Parameters (changeable via ros2 params or launch) ---
        self.declare_parameter('robot_names', ['tb1','tb2'])
        self.declare_parameter('model_path', '/path/to/marl_ppo_inference.pth')
        self.declare_parameter('control_rate', 10.0)          # Hz
        self.declare_parameter('lidar_proc_size', 24)         # downsampled lidar points per agent
        self.declare_parameter('max_linear', 0.22)            # m/s
        self.declare_parameter('max_angular', 1.0)            # rad/s
        self.declare_parameter('use_torchscript', False)     # if True, load with torch.jit.load

        self.robot_names: List[str] = self.get_parameter('robot_names').get_parameter_value().string_array_value
        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.rate = float(self.get_parameter('control_rate').get_parameter_value().double_value)
        self.lidar_size = int(self.get_parameter('lidar_proc_size').get_parameter_value().integer_value)
        self.max_lin = float(self.get_parameter('max_linear').get_parameter_value().double_value)
        self.max_ang = float(self.get_parameter('max_angular').get_parameter_value().double_value)
        self.use_torchscript = bool(self.get_parameter('use_torchscript').get_parameter_value().bool_value)

        # --- State ---
        self.last_scan: Dict[str, tuple] = {ns: None for ns in self.robot_names}  # (ranges np.array, range_max)
        self.last_odom: Dict[str, tuple] = {ns: None for ns in self.robot_names}  # (vx, wz)
        self.cmd_pubs: Dict[str, rclpy.publisher.Publisher] = {}

        # --- Load model once ---
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(self.model_path)

        device = torch.device('cpu')
        try:
            if self.use_torchscript:
                self.model = torch.jit.load(self.model_path, map_location=device)
                self.get_logger().info("Loaded TorchScript model.")
            else:
                # attempt to load regular torch state
                self.model = torch.load(self.model_path, map_location=device)
                # if it's a state_dict, user must supply code to rebuild model; otherwise assume it's a module
                if isinstance(self.model, dict) and 'state_dict' in self.model:
                    self.get_logger().warn("Loaded dictionary (state_dict?). You may need to load into model class instead.")
                else:
                    self.model.eval()
                    self.get_logger().info("Loaded PyTorch model.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        # --- ROS interfaces ---
        qos = QoSProfile(depth=10)
        for ns in self.robot_names:
            self.create_subscription(LaserScan, f'/{ns}/scan', partial(self.scan_cb, ns), qos_profile=qos)
            self.create_subscription(Odometry, f'/{ns}/odom', partial(self.odom_cb, ns), qos_profile=qos)
            self.cmd_pubs[ns] = self.create_publisher(Twist, f'/{ns}/cmd_vel', qos_profile=qos)

        # Timer for synchronous control loop
        self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
        self.get_logger().info(f"PPOBridge initialized for robots: {self.robot_names}")

    # --- Callbacks ---
    def scan_cb(self, ns: str, msg: LaserScan):
        rngs = np.array(msg.ranges, dtype=np.float32)
        self.last_scan[ns] = (rngs, float(msg.range_max))

    def odom_cb(self, ns: str, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        wz = float(msg.twist.twist.angular.z)
        self.last_odom[ns] = (vx, wz)

    # --- Preprocessers ---
    def process_scan(self, ranges: np.ndarray, rmax: float) -> np.ndarray:
        # Replace inf / nan with range max, clip, downsample uniformly, and normalize to [0,1]
        rng = np.array(ranges, copy=True)
        rng[np.isinf(rng)] = rmax
        rng[np.isnan(rng)] = rmax
        rng = np.clip(rng, 0.0, rmax)
        if len(rng) == 0:
            return np.ones(self.lidar_size, dtype=np.float32)
        idx = np.linspace(0, len(rng)-1, self.lidar_size).astype(int)
        small = rng[idx].astype(np.float32)
        small = small / float(rmax)    # normalized
        return small

    def assemble_obs_for_agent(self, ns: str) -> np.ndarray:
        # Returns 1D observation vector for single agent in the same order each time
        if self.last_scan[ns] is None or self.last_odom[ns] is None:
            return None
        ranges, rmax = self.last_scan[ns]
        scan_vec = self.process_scan(ranges, rmax)          # length = lidar_size
        vx, wz = self.last_odom[ns]
        # normalize velocities relative to known maxima
        vx_n = np.clip(vx / max(self.max_lin, 1e-6), -1.0, 1.0)
        wz_n = np.clip(wz / max(self.max_ang, 1e-6), -1.0, 1.0)
        obs = np.concatenate([scan_vec, np.array([vx_n, wz_n], dtype=np.float32)])
        return obs

    # --- Main control loop (synchronous batched inference) ---
    def control_loop(self):
        # Make sure we have fresh obs for all agents
        obs_list = []
        active_names = []
        for ns in self.robot_names:
            obs = self.assemble_obs_for_agent(ns)
            if obs is None:
                # if any agent missing data, skip this tick
                return
            obs_list.append(obs)
            active_names.append(ns)

        # Build batch: shape (N_agents, obs_dim)
        batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)

        # Forward pass with flexible handling of return types
        with torch.no_grad():
            try:
                out = self.model(batch)   # common case
            except Exception as e:
                # try calling with dict or as (obs,) argument
                try:
                    out = self.model({'obs': batch})
                except Exception as e2:
                    self.get_logger().error(f"Model forward failed: {e} / {e2}")
                    return

        # Interpret out to obtain action tensor of shape (N, action_dim)
        actions = None
        if isinstance(out, torch.Tensor):
            actions = out
        elif isinstance(out, (tuple, list)):
            # commonly (actions, values, log_probs) etc.
            actions = out[0]
        elif isinstance(out, dict):
            # torchrl sometimes returns dict-like TensorDict
            # try common keys
            for k in ('actions','action','act','output'):
                if k in out:
                    actions = out[k]
                    break
            # attempt to get first tensor if not found
            if actions is None:
                # pick first tensor-like value
                for v in out.values():
                    if isinstance(v, torch.Tensor):
                        actions = v
                        break
        if actions is None:
            self.get_logger().error("Could not parse model output into actions. Output keys/types: " + str(type(out)))
            return

        # Ensure numpy
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            try:
                actions_np = np.array(actions)
            except Exception as e:
                self.get_logger().error(f"Failed to convert actions to numpy: {e}")
                return

        # Publish per-agent
        for i, ns in enumerate(active_names):
            # Assume action [:,0] = linear in [-1,1], [:,1] = angular in [-1,1]
            a = actions_np[i]
            if a.ndim == 0:
                self.get_logger().error("Action for agent is scalar; expected vector of length >=2.")
                continue
            # handle if actions are probabilities or other shapes: user may need to adapt here
            lin = float(a[0]) if a.size >= 1 else 0.0
            ang = float(a[1]) if a.size >= 2 else 0.0
            # scale to robot max
            lin = np.clip(lin, -1.0, 1.0) * self.max_lin
            ang = np.clip(ang, -1.0, 1.0) * self.max_ang
            msg = Twist()
            msg.linear.x = float(lin)
            msg.angular.z = float(ang)
            self.cmd_pubs[ns].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PPOBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
