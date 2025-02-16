import abc
import logging
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

class Court(object):
    def __init__(self, create_position):
        self.cp = create_position
        self.num_lines = 8
        self.line_ids = []
        self.line_default_color = [0, 1, 0]
        self.line_hit_color = [1, 0, 0]
        self.line_from = [[0.06, 0.06, 0.05],
                         [1.76, 0.06, 0.05],
                         [0.06, 0.06, 0.05],
                         [1.22, 0.06, 0.05],
                         [0.06, 2.37, 0.05],
                         [1.22, 2.37, 0.05],
                         [0.61, 0.11, 0.05],
                         [0.61, 2.32, 0.05]]
        self.line_to = [[0.06, 2.37, 0.05],
                       [1.76, 2.37, 0.05],
                       [0.60, 0.06, 0.05],
                       [1.76, 0.06, 0.05],
                       [0.60, 2.37, 0.05],
                       [1.76, 2.37, 0.05],
                       [1.21, 0.11, 0.05],
                       [1.21, 2.32, 0.05]]
        self.from_positions = []
        self.to_positions = []
        self.__my_goal_line_idx = 6
        self.__enemy_goal_line_idx = 7
        self.hit_ids = []
        self.my_goal_position = [0.91+self.cp[0], 0.11+self.cp[1], 0.05]
        self.enemy_goal_position = [0.91+self.cp[0], 2.32+self.cp[1], 0.05]

    @property
    def my_goal_line_idx(self):
        return self.__my_goal_line_idx

    @property
    def enemy_goal_line_idx(self):
        return self.__enemy_goal_line_idx

    def create_court(self):
        """STLファイルから衝突形状を作成"""
        collision_court = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="stl_data/court.stl",
            meshScale=[0.001, 0.001, 0.001],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        """STLファイルから視覚形状を作成"""
        visual_court = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="stl_data/court.stl",
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[1, 1, 1, 1]
        )
        """衝突形状と視覚形状を持つマルチボディを作成"""
        court_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_court,
            baseVisualShapeIndex=visual_court,
            basePosition=self.cp,
            baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        )

    def create_court_line(self):
        """コートのデバックラインを作成"""
        for i in range(self.num_lines):
            self.line_ids.append(p.addUserDebugLine(
                [self.line_from[i][0] + self.cp[0],
                 self.line_from[i][1] + self.cp[1],
                 self.line_from[i][2] + self.cp[2]],
                [self.line_to[i][0] + self.cp[0],
                 self.line_to[i][1] + self.cp[1],
                 self.line_to[i][2] + self.cp[2]],
                lineColorRGB=self.line_default_color,
                lineWidth=1.0,
                lifeTime=0
            ))

    def detection_line(self):
        """ラインの当たり判定を検出し、ヒット時はラインを上書き更新"""
        self.from_positions = []
        self.to_positions = []
        self.hit_ids = []

        for i in range(self.num_lines):
            self.from_positions.append([
                self.line_from[i][0] + self.cp[0],
                self.line_from[i][1] + self.cp[1],
                self.line_from[i][2] + self.cp[2]
            ])
            self.to_positions.append([
                self.line_to[i][0] + self.cp[0],
                self.line_to[i][1] + self.cp[1],
                self.line_to[i][2] + self.cp[2]
            ])

        results = p.rayTestBatch(self.from_positions, self.to_positions)

        for i in range(self.num_lines):
            hit_id = results[i][0]
            self.hit_ids.append(hit_id)
            # ここで、replaceItemUniqueIdを使用して同じラインIDを書き換える
            if hit_id == -1:
                # ヒットがない場合は元の色で上書き
                p.addUserDebugLine(
                    self.from_positions[i],
                    self.to_positions[i],
                    lineColorRGB=self.line_default_color,
                    lineWidth=1.0,
                    lifeTime=0,
                    replaceItemUniqueId=self.line_ids[i]
                )
            else:
                # ヒットがある場合はヒット色で上書き
                p.addUserDebugLine(
                    self.from_positions[i],
                    self.to_positions[i],
                    lineColorRGB=self.line_hit_color,
                    lineWidth=1.0,
                    lifeTime=0,
                    replaceItemUniqueId=self.line_ids[i]
                )

        return self.hit_ids

    def create_ball(self):
        collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE,
                                                    radius=0.037)

        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE,
                                              radius=0.037,
                                              rgbaColor=[0.4, 0.4, 0.4, 1])

        ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.915+self.cp[0], 1.21+self.cp[1], 0.1+self.cp[2]]
        )
        return ball_id

if __name__ == '__main__':
    import doctest
    doctest.testmod()