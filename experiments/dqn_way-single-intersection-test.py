import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

if __name__ == '__main__':
    model = DQN.load("dqn_2_way_weibull")

    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        route_file="nets/2way-single-intersection/train/uniform.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn/train",
        single_agent=True,
        use_gui=True,
        num_seconds=5400,
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=1,
        deterministic=True
    )

    print(mean_reward, std_reward)