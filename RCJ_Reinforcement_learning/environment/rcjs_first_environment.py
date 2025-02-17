import random
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from RCJ_Reinforcement_learning.tools.rcjs_unit import Unit
from RCJ_Reinforcement_learning.tools.rcjs_calculation_tool import CalculationTool
from RCJ_Reinforcement_learning.tools.rcjs_reward_calculation import FirstRewardCalculation


class Environment(gym.Env):
    def __init__(self, create_position, max_epoch, GUI=False):
        super().__init__()
        # PyBulletの初期化
        if GUI:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadSDF("stadium.sdf")
        p.setGravity(0, 0, -9.81)
        self.action_space = spaces.Discrete(360)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(3,),
                                            dtype=np.float32)
        self.unit = Unit()
        self.cal = CalculationTool()
        self.reward_cal = FirstRewardCalculation()
        self.cp = create_position
        self.agent_random_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.unit.create_unit(self.cp, self.agent_random_pos)

        self.hit_ids = []
        self.max_steps = max_epoch
        self.step_count = 0

        self.reset()

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        self.unit.action(robot_id=self.unit.agent_id,
                         angle_deg=action,
                         magnitude=7.0)
        for _ in range(10):
            p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(self.unit.agent_id)
        fixed_ori = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.unit.agent_id,
                                          pos,
                                          fixed_ori)

        self.step_count += 1

        agent_pos, _ = p.getBasePositionAndOrientation(self.unit.agent_id)

        ball_angle = self.cal.angle_calculation_id(self.unit.agent_id,
                                                     self.unit.ball_id)

        enemy_goal_angle = self.cal.angle_calculation_pos(agent_pos,
                                                           self.unit.court.enemy_goal_position)

        my_goal_angle = self.cal.angle_calculation_pos(agent_pos,
                                                        self.unit.court.my_goal_position)

        self.hit_ids = self.unit.detection_line()
        reward = self.reward_cal.reward_calculation(self.hit_ids,
                                              self.unit.agent_id,
                                              self.unit.ball_id,
                                              self.step_count)

        observation = np.array([ball_angle,
                                enemy_goal_angle,
                                my_goal_angle],
                               dtype=np.float32)

        if self.reward_cal.is_goal:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        #print(my_goal_angle, enemy_goal_angle, reward)
        #print(observation)
        #print(reward)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF("stadium.sdf")

        if seed is not None:
            np.random.seed(seed)

        self.agent_random_pos[0] = random.uniform(0.4, 1.5)
        self.agent_random_pos[1] = random.uniform(0.4, 1.5)

        self.unit = Unit()
        self.unit.create_unit(self.cp, self.agent_random_pos)

        self.step_count = 0
        initial_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        info = {}
        return initial_obs, info

    def render(self):
        pass

    def close(self):
        p.disconnect()
