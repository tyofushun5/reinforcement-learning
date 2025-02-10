import os
import sys
from picamera2 import Picamera2
import cv2 as cv
import numpy as np


#CAM0 (V3カメラ広角)
cam0 = Picamera2(0)
#CAM1 (HQカメラM12)
cam1 = Picamera2(1)

#CAM0の設定
config0 = cam0.create_preview_configuration(
    #V3カメラの解像度
    main={"size": (1280, 720)},
    #露光時間とゲイン調整
    controls={"ExposureTime": 5000, "AnalogueGain": 2.0}
)
cam0.configure(config0)

#CAM1の設定
config1 = cam1.create_preview_configuration(
    # HQカメラの解像度
    main={"size": (640, 480)},
    #露光時間とゲイン調整
    controls={"ExposureTime": 10000, "AnalogueGain": 2.0}
)
cam1.configure(config1)

def initialization(frame):
    """RGBからBGR"""
    frame_BGR = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    # ノイズ除去
    frame_blur = cv.blur(frame_BGR, (3, 3))
    return frame_blur

def blue_detection(frame):
    """青色検知"""
    # HSV色空間に変換
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #マスクを作成
    lower = np.array([100, 150, 0])
    upper = np.array([125, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    #カーネル
    kernel = np.ones((3, 3), dtype=np.uint8)
    #クロージング
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    #収縮処理
    mask = cv.erode(mask, kernel)
    return mask

def blob_detection(frame, mask):
    """blob検出"""
    nLabels, labelImage, stats, centroids = cv.connectedComponentsWithStats(mask)
    frame_blob = frame.copy()
    h, w = frame.shape[:2]

    # 最大面積のblobを探す（ゴール候補）
    max_area = 0
    max_label = -1
    for i in range(1, nLabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i

    if max_label == -1:
        # ゴールが検出されなかった場合
        return frame_blob, None

    # 最大のラベルに対応する重心を取得
    cx, cy = centroids[max_label]

    # デバッグ用にblobの部分を強調
    frame_blob = np.zeros_like(frame)
    frame_blob[labelImage == max_label] = [255, 0, 0]

    return frame_blob, (cx, cy)

def calculate_angle(cx, frame_width, fov_h=60.0):
    """
    重心cxから角度を計算する簡易例
    fov_h: カメラの水平画角(度)
    frame_width: 画像の幅
    ここではシンプルに、画像中心からcxまでのピクセル偏差をfovで割り、
    画角換算して角度にしている。
    """
    center_x = frame_width / 2.0
    # 画像中心との差（画素数）
    dx = cx - center_x
    # ピクセルあたりの角度(度/px)
    deg_per_px = fov_h / frame_width
    angle = dx * deg_per_px
    return angle

#カメラの起動
cam0.start()
cam1.start()

# メインループ
while True:
    ret0, frame0 = cam0.capture_array()
    ret1, frame1 = cam1.capture_array()
    if not ret0 or not ret1:
        break
    #前処理
    frame0 = initialization(frame0)
    frame1 = initialization(frame1)
    #色検出
    mask0 = blue_detection(frame0)
    mask1 = blue_detection(frame1)
    #ブロフ検出
    frame_blob0, centroid = blob_detection(frame0, mask0)
    frame_blob1, centroid= blob_detection(frame1, mask1)

    cv.imshow("frame0", frame0)
    cv.imshow("frame1", frame1)

    if cv.waitKey(0) == 27:
        break

# カメラの停止とリソースの解放
cam0.stop()
cam1.stop()
cam0.close()
cam1.close()
cv.destroyAllWindows()