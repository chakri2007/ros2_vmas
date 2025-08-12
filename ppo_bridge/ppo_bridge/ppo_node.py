import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped  # <-- ADD TwistStamped
from std_msgs.msg import Header  # <-- ADD Header
import torch
import numpy as np
from tensordict import TensorDict

# TorchRL imports
from torchrl.modules import MultiAgentMLP, TanhNormal
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor
from torchrl.envs.utils import set_exploration_type

# --- Define the PolicyNet architecture from your training script ---
class PolicyNet(ProbabilisticActor):
    def __init__(self, obs_size, action_size):
        actor_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_size,
                n_agent_outputs=2 * action_size,
                n_agents=2,
                centralised=False,
                share_params=True,
                device='cpu',
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            module=actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        super().__init__(
            module=policy_module,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[("agents", "action")],
            distribution_class=TanhNormal,
            return_log_prob=False,
            spec=None 
        )

# --- The rest of your ROS 2 node code ---
class MultiTB3PolicyNode(Node):
    def __init__(self):
        super().__init__('multi_tb3_policy_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.obs_size = 18
        self.action_size = 2

        self.policy = PolicyNet(obs_size=self.obs_size, action_size=self.action_size)
        
        self.policy.load_state_dict(torch.load('path/multi_agent_policy.pth', map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()

        self.robots = ['tb1', 'tb2']
        self.scan_data = {name: None for name in self.robots}
        self.cmd_vel_pubs = {}
        for name in self.robots:
            # Change the publisher to TwistStamped
            self.create_subscription(LaserScan, f'/{name}/scan', self.scan_callback_factory(name), 10)
            self.cmd_vel_pubs[name] = self.create_publisher(TwistStamped, f'/{name}/cmd_vel', 10)  # <-- UPDATED
            
        self.create_timer(0.1, self.control_loop)

    def scan_callback_factory(self, robot_name):
        def callback(msg):
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.nan_to_num(ranges, nan=3.5)
            self.scan_data[robot_name] = ranges
        return callback

    def control_loop(self):
        obs_batch = []
        ready_robots = []

        for name in self.robots:
            if self.scan_data[name] is not None:
                obs = self.preprocess_obs(self.scan_data[name])
                obs_batch.append(obs)
                ready_robots.append(name)
            else:
                obs_batch.append(np.zeros(self.obs_size, dtype=np.float32))

        obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        tensordict_obs = TensorDict({("agents", "observation"): obs_tensor.unsqueeze(0)}, batch_size=[1, len(self.robots)])

        with set_exploration_type("deterministic"):
            with torch.no_grad():
                actions_td = self.policy(tensordict_obs)

        # Define a velocity scaling factor here
        # A value of 1.0 means no scaling. A value of 0.5 will halve the speed.
        velocity_scaling_factor = 0.1

        for idx, name in enumerate(ready_robots):
            twist_stamped_msg = TwistStamped()
            twist_stamped_msg.header.stamp = self.get_clock().now().to_msg()
            twist_stamped_msg.header.frame_id = f'{name}/base_link'

            action_data = actions_td.get(("agents", "action"))[0, self.robots.index(name)].cpu()
            
            # Apply the scaling factor to the linear and angular velocities
            twist_stamped_msg.twist.linear.x = float(action_data[0] * velocity_scaling_factor)
            twist_stamped_msg.twist.angular.z = float(action_data[1] * velocity_scaling_factor)
            
            self.cmd_vel_pubs[name].publish(twist_stamped_msg)
    def preprocess_obs(self, scan):
        step = max(1, len(scan) // self.obs_size)
        return scan[::step][:self.obs_size]

def main(args=None):
    rclpy.init(args=args)
    node = MultiTB3PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
