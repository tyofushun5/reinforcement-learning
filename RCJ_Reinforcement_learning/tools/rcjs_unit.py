import math

import pybullet as p

from RCJ_Reinforcement_learning.tools.rcjs_court import Court
from RCJ_Reinforcement_learning.tools.rcjs_robot import Agent

from RCJ_Reinforcement_learning.tools.rcjs_reward_calculation import FirstRewardCalculation

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

