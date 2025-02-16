import abc

import pybullet as p

from RCJ_Reinforcement_learning.tools.rcjs_court import Court
from RCJ_Reinforcement_learning.tools.rcjs_calculation_tool import CalculationTool

class RewardCalculation(metaclass=abc.ABCMeta):

    def __init__(self):
        self.my_goal_line_idx = 6
        self.enemy_goal_line_idx = 7
        self.is_goal = False

    @staticmethod
    def reward_calculation(self, *args):
        self.is_goal = False

class FirstRewardCalculation():
    def __init__(self):
        super().__init__()
        self.my_goal_line_idx = 6
        self.enemy_goal_line_idx = 7
        self.previous_attacker_pos = [0, 0, 0]
        self.past_distance = 0
        self.ball_past_distance = 0
        self.is_goal = False
        self.cal = CalculationTool()

    def reward_calculation(self, hit_ids, agent_id, ball_id, step_count):
        reward = 0
        self.is_goal = False
        agent_pos, _ = p.getBasePositionAndOrientation(agent_id)

        reward += self.cal.movement_reward_calculation(reward,
                                                   agent_pos,
                                                   self.previous_attacker_pos,
                                                   self.past_distance)
        self.previous_attacker_pos = agent_pos

        ball_pos, _ = p.getBasePositionAndOrientation(ball_id)

        reward += self.cal.distance_reward_calculation(reward,
                                                   agent_pos,
                                                   ball_pos,
                                                   self.ball_past_distance)

        self.ball_past_distance = self.cal.euclidean_distance_pos(agent_pos,
                                                                  ball_pos)

        is_touch = p.getContactPoints(ball_id, agent_id)
        if is_touch:
            reward += 5
        if hit_ids[self.my_goal_line_idx] == ball_id:
            reward -= 10
        if hit_ids[self.enemy_goal_line_idx] == ball_id:
            reward += 10
            self.is_goal = True
        for i in range(len(hit_ids)):
            if hit_ids[i] == agent_id:
                reward -= 0.2
        angle = self.cal.angle_calculation_id(agent_id, ball_id)
        if angle<=90 or angle>=270:
            reward += 0.1
        else:
            reward -= 0.1
        if angle<=45 or angle>=315:
            reward += 0.2
        return reward
