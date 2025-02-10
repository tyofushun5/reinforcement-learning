import math

import pybullet as p

from rcjs_court import Court
from rcjs_robot import Agent


class Unit(object):

    my_goal_line_idx = 6
    enemy_goal_line_idx = 7

    def __init__(self):
        self.cp = None
        self.ball_id = None
        self.attacker_id = None
        self.court = None
        self.agent = None
        self.hit_ids = []
        self.previous_attacker_pos = [0, 0, 0]
        self.past_distance = 0
        self.ball_past_distance = 0
        self.is_goal = False

    def create_unit(self, create_position):
        self.cp = create_position
        self.court = Court(self.cp)
        self.agent = Agent(self.cp)
        self.court.create_court()
        self.court.create_court_line()
        self.ball_id = self.court.create_ball()
        self.attacker_id = self.agent.create()
        self.previous_attacker_pos = self.agent.start_pos

    def detection_line(self):
        self.hit_ids = self.court.detection_line()
        return self.hit_ids

    def action(self, robot_id, angle_deg, magnitude):
        self.agent.action(robot_id, angle_deg, magnitude)

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

    @staticmethod
    def is_ball_touch(attacker_id, ball_id, count):
        is_touch = p.getContactPoints(ball_id, attacker_id)
        if is_touch:
            count = 0
        return count

    @staticmethod
    def angle_calculation_id(a_id, b_id):
        a_pos, _ = p.getBasePositionAndOrientation(a_id)
        b_pos, _ = p.getBasePositionAndOrientation(b_id)
        dx = b_pos[0] - a_pos[0]
        dy = b_pos[1] - a_pos[1]
        angle_radians = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_radians)
        angle_deg = angle_deg % 360
        angle_deg = round(angle_deg, 1)
        return angle_deg

    @staticmethod
    def angle_calculation_pos(a_pos, b_pos):
        dx = b_pos[0] - a_pos[0]
        dy = b_pos[1] - a_pos[1]
        angle_radians = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_radians)
        angle_deg = angle_deg % 360
        angle_deg = round(angle_deg, 1)
        return angle_deg

    @staticmethod
    def euclidean_distance_id(a_id, b_id):
        a_pos, _ = p.getBasePositionAndOrientation(a_id)
        b_pos, _ = p.getBasePositionAndOrientation(b_id)
        if len(a_pos) != len(b_pos):
            raise ValueError
        x1, y1 = a_pos[:2]
        x2, y2 = b_pos[:2]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def euclidean_distance_pos(a_pos, b_pos):
        if len(a_pos) != len(b_pos):
            raise ValueError
        x1, y1 = a_pos[:2]
        x2, y2 = b_pos[:2]
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def movement_reward_calculation(reward, pos, previous_pos, past_distance):
        distance = Unit.euclidean_distance_pos(pos, previous_pos)
        if distance < past_distance:
            reward += 0.5
        else:
            reward -= 0.5
        return reward

    @staticmethod
    def distance_reward_calculation(reward, agent_pos, ball_pos, ball_past_distance):
        distance = Unit.euclidean_distance_pos(agent_pos, ball_pos)
        if distance < ball_past_distance:
            reward += 0.5
        else:
            reward -= 0.4
        return reward
