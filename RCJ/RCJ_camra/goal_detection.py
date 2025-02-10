import cv2 as cv
import numpy as np


#CAM　(logitech開発用カメラ)
cam = cv.VideoCapture(0)

def initialization(frame):
    """RGBからBGR"""
    #frame_BGR = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    # ノイズ除去
    frame_blur = cv.blur(frame, (3, 3))
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

"""def blob_detection(frame, mask):
    nLabels, labelImage, stats, centroids = cv.connectedComponentsWithStats(mask)
    frame_blob = frame.copy()
    h, w = frame.shape[:2]
    for y in range(h):
        for x in range(w):
            label = labelImage[y, x]
            if label > 0:
                frame_blob[y, x] = [255, 0, 0]
            else:
                frame_blob[y, x] = [0, 0, 0]
    return frame_blob"""

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

# メインループ
while True:
    ret, frame = cam.read()
    if not ret:
        break

    #前処理
    frame = initialization(frame)
    #色検出
    mask = blue_detection(frame)
    #ブロフ検出
    frame_blob, centroid = blob_detection(frame, mask)

    if centroid is not None:
        cx, cy = centroid
        # 簡易的な角度計算（水平方向）
        angle = calculate_angle(cx, frame.shape[1], fov_h=60.0)
        cv.putText(frame_blob,
                   f"Angle: {angle:.2f} deg",
                   (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 0), 2)

    cv.imshow("frame", frame_blob)

    if cv.waitKey(1) == 27:
        break

# カメラの停止とリソースの解放
cam.release()
cv.destroyAllWindows()