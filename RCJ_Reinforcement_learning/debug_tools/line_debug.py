import time

import pybullet as p
import pybullet_data

from RCJ_Reinforcement_learning.tools.rcjs_unit import Unit


# PyBulletの物理エンジンに接続
physicsClient = p.connect(p.GUI)

# データパスを追加
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadSDF("stadium.sdf")

# 重力を設定
p.setGravity(0, 0, -9.8)

detection_interval = 0


unit1 = Unit()
unit1.create_unit([0, 0, 0])

unit2 = Unit()
unit2.create_unit([0, 4, 0])

# シミュレーションをステップ実行
t = 0
while True:
    p.stepSimulation()
    time.sleep(0.01)
    t += 0.01

    detection_interval += 0.01
    if detection_interval >= 0.10:
        unit2.detection_line()
        #print(len(x))
        x, y = p.getBasePositionAndOrientation(unit1.attacker_id)
        print(y)
        detection_interval = 0




