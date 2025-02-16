import abc
import math

import pybullet as p

from RCJ_Reinforcement_learning.tools.rcjs_calculation_tool import CalculationTool

class Robot(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def action(self, robot_id, angle_deg, magnitude):
        pass


class Agent(Robot):
    def __init__(self, create_position):
        super().__init__()
        self.cal = CalculationTool()
        self.cp = create_position
        self.start_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.base_position = self.start_pos

    # def create(self):
    #      """アタッカーを生成
    #      オブジェクト設定"""
    #      attacker = p.createCollisionShape(
    #          shapeType=p.GEOM_MESH,
    #          fileName="stl_data/robot.stl",
    #          meshScale=[0.001, 0.001, 0.001]
    #      )
    #      #外観設定
    #      attacker_visual = p.createVisualShape(
    #          shapeType=p.GEOM_MESH,
    #          fileName="stl_data/robot.stl",
    #          meshScale=[0.001, 0.001, 0.001],
    #          rgbaColor=[0.2, 0.2, 0.2, 1] #黒色
    #      )
    #      # 動的ボディとしてオブジェクトを作成
    #      attacker_id = p.createMultiBody(
    #          baseMass=1.4,
    #          baseCollisionShapeIndex=attacker,
    #          baseVisualShapeIndex=attacker_visual,
    #          basePosition=[1+self.cp[0], 1+self.cp[1], 0.1+self.cp[2]],
    #          baseOrientation = p.getQuaternionFromEuler([0, 0, 0])
    #      )
    #      return attacker_id

    def create(self, base_position=None):
        if base_position is None:
            self.base_position = self.start_pos

        radius = 0.11  # 半径（メートル）
        height = 0.11  # 高さ（メートル）
        mass = 1.4  # 質量（キログラム）

        # 円柱の衝突形状を作成
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )

        # 円柱の視覚形状を作成
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.2, 0.2, 0.2, 1]
        )

        # 円柱のマルチボディを作成
        attacker_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=base_position
        )
        return attacker_id

    def action(self, robot_id, angle_deg=0, magnitude=7.0):
        """ロボットを動かす関数"""

        # Dynamics情報を取得
        dynamics_info = p.getDynamicsInfo(robot_id, -1)
        center_of_mass = dynamics_info[3]  # 重心の位置

        # ダイナミクス設定（摩擦係数を調整）
        p.changeDynamics(
            bodyUniqueId=robot_id,
            linkIndex=-1,
            lateralFriction=0.5,  # 適度に低い摩擦係数
            spinningFriction=0.1,  # 回転摩擦を抑える
            rollingFriction=0.1,  # 転がり摩擦を抑える
            angularDamping=1.0  # 回転の減衰を大きくする
        )

        x, y = self.cal.vector_calculations(angle_deg=angle_deg, magnitude=magnitude)

        # 円柱の中心に力を加える
        p.applyExternalForce(
            objectUniqueId=robot_id,
            linkIndex=-1,
            forceObj=[x, y, 0],
            posObj=center_of_mass,  # 重心の位置に力を適用
            flags=p.WORLD_FRAME
        )

        # 回転速度をリセット
        p.resetBaseVelocity(
            objectUniqueId=robot_id,
            angularVelocity=[0, 0, 0]  # 回転速度をゼロに設定
        )

class Defender(object):
    def __init__(self, create_position):
        super().__init__()
        self.cp = create_position
        self.start_pos = [1+self.cp[0], 0.5+self.cp[1], 0.1+self.cp[2]]
        self.base_position = self.start_pos



if __name__ == '__main__':
    import doctest
    doctest.testmod()

