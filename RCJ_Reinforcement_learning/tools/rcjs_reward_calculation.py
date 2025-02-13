import abc
import math

import pybullet as p

class RewardCalculation(mataclass=abc.ABCMeta):
    def __init__(self):
        pass

class FirstRewardCalculation(RewardCalculation):
    def __init__(self):
        super().__init__()

    def reward_calculation(self, hit_ids, attacker_id, ball_id, step_count):
        pass
