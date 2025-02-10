from picamera2 import Picamera2
import cv2 as cv

# HQカメラ (CAM1) の特性を考慮
# カメラインスタンス作成
cam0 = Picamera2(0)  # CAM0 (V3カメラ広角)
cam1 = Picamera2(1)  # CAM1 (HQカメラM12)

config0 = cam0.create_preview_configuration(
    main={"size": (1280, 720)},  # 解像度: フルHD
    controls={"ExposureTime": 5000, "AnalogueGain": 2.0}  # 手動露出調整
)
cam0.configure(config0)


config1 = cam1.create_preview_configuration(
    main={"size": (640, 480)},  # HQカメラの高解像度
    controls={"ExposureTime": 10000, "AnalogueGain": 2.0}  # 高感度設定
)
cam1.configure(config1)

# カメラの起動
cam0.start()
cam1.start()

# メインループ
while True:

    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    if frame0 is not None and frame1 is not None:
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        cv.imshow("frame0", frame0)
        cv.imshow("frame1", frame1)

    if cv.waitKey(1) == 27:
        break

# カメラの停止とリソースの解放
cam0.stop()
cam1.stop()
cv.destroyAllWindows()
