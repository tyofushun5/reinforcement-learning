from picamera2 import Picamera2
import cv2 as cv
import threading

# HQカメラ (CAM1) の特性を考慮
# カメラインスタンス作成
cam0 = Picamera2(0)  # CAM0 (V3カメラ広角)
cam1 = Picamera2(1)  # CAM1 (HQカメラM12)

# CAM0 (V3カメラ広角) の設定
config0 = cam0.create_preview_configuration(
    main={"size": (1536, 864)},  # 解像度: フルHD
    controls={"ExposureTime": 10000, "AnalogueGain": 1.0}  # 手動露出調整
)
cam0.configure(config0)

# CAM1 (HQカメラM12) の設定
config1 = cam1.create_preview_configuration(
    main={"size": (1332, 990)},  # HQカメラの高解像度
    controls={"ExposureTime": 20000, "AnalogueGain": 2.0}  # 高感度設定
)
cam1.configure(config1)

# カメラの起動
cam0.start()
cam1.start()

# グローバル変数でフレームを保持
frame0 = None
frame1 = None
running = True

# CAM0 (V3カメラ広角) のスレッド
def capture_cam0():
    global frame0, running
    while running:
        frame0 = cam0.capture_array()

# CAM1 (HQカメラM12) のスレッド
def capture_cam1():
    global frame1, running
    while running:
        frame1 = cam1.capture_array()

# スレッド開始
thread0 = threading.Thread(target=capture_cam0)
thread1 = threading.Thread(target=capture_cam1)
thread0.start()
thread1.start()

# メインループ
while running:
    if frame0 is not None and frame1 is not None:

        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        cv.imshow("frame0", frame0)
        cv.imshow("frame1", frame1)

    # 'q'キーで終了
    if cv.waitKey(1) == 27:
        break

thread0.join()
thread1.join()
cam0.stop()
cam1.stop()
cv.destroyAllWindows()


