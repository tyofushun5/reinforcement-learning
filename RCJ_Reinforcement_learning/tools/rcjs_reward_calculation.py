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
        reward = 0
        attacker_pos, _ = p.getBasePositionAndOrientation(attacker_id)

        reward += self.movement_reward_calculation(reward,
                                                   attacker_pos,
                                                   self.previous_attacker_pos,
                                                   self.past_distance)
        self.previous_attacker_pos = attacker_pos

        ball_pos, _ = p.getBasePositionAndOrientation(ball_id)

        reward += self.distance_reward_calculation(reward,
                                                   attacker_pos,
                                                   ball_pos,
                                                   self.ball_past_distance)

        self.ball_past_distance = self.euclidean_distance_pos(attacker_pos, ball_pos)

        is_touch = p.getContactPoints(ball_id, attacker_id)
        if is_touch:
            reward += 5
        if hit_ids[self.my_goal_line_idx] == ball_id:
            reward -= 10
        if hit_ids[self.enemy_goal_line_idx] == ball_id:
            reward += 10
            self.is_goal = True
        for i in range(len(hit_ids)):
            if hit_ids[i] == attacker_id:
                reward -= 0.2
        angle = self.angle_calculation_id(attacker_id, ball_id)
        if angle<=90 or angle>=270:
            reward += 0.1
        else:
            reward -= 0.1
        if angle<=45 or angle>=315:
            reward += 0.2
        return reward
