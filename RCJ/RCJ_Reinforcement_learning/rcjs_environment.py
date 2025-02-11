import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rcjs_unit import Unit


class Environment(gym.Env):
    def __init__(self):
        super().__init__()
        # PyBulletの初期化
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadSDF("stadium.sdf")
        p.setGravity(0, 0, -9.81)
        self.action_space = spaces.Discrete(360)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(3,),
                                            dtype=np.float32)

        self.cp = [0, 0, 0]
        self.detection_interval = 0

        self.unit = Unit()
        self.unit.create_unit(self.cp)

        self.hit_ids = []
        self.max_steps = 25000
        self.step_count = 0

        self.reset()

    def step(self, action):
        self.unit.action(robot_id=self.unit.attacker_id,
                         angle_deg=action,
                         magnitude=7.5)
        for _ in range(10):
            p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(self.unit.attacker_id)
        fixed_ori = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.unit.attacker_id, pos, fixed_ori)

        self.step_count += 1

        attacker_pos, _ = p.getBasePositionAndOrientation(self.unit.attacker_id)

        ball_angle = self.unit.angle_calculation_id(self.unit.attacker_id,
                                                     self.unit.ball_id)

        enemy_goal_angle = self.unit.angle_calculation_pos(attacker_pos,
                                                           self.unit.court.enemy_goal_position)

        my_goal_angle = self.unit.angle_calculation_pos(attacker_pos,
                                                        self.unit.court.my_goal_position)

        self.hit_ids = self.unit.detection_line()
        reward = self.unit.reward_calculation(self.hit_ids,
                                              self.unit.attacker_id,
                                              self.unit.ball_id,
                                              self.step_count)

        # self.step_count = self.unit.is_ball_touch(self.unit.attacker_id,
        #                                           self.unit.ball_id,
        #                                           self.step_count)

        observation = np.array([ball_angle, enemy_goal_angle, my_goal_angle], dtype=np.float32)

        terminated = False
        truncated = False
        info = {}
        if self.unit.is_goal:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        #print(my_goal_angle, enemy_goal_angle, reward)
        #print(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadSDF("stadium.sdf")

        self.unit = Unit()
        self.unit.create_unit(self.cp)

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        initial_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        info = {}
        return initial_obs, info

    def render(self):
        pass

    def close(self):
        p.disconnect()
