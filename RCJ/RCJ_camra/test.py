from picamera2 import Picamera2
import cv2
import threading

# カメラ1とカメラ2のインスタンス作成
cam0 = Picamera2(0)  # CAM0ポート
cam1 = Picamera2(1)  # CAM1ポート

# カメラの設定と起動
cam0.start()
cam1.start()

# グローバル変数でフレームを保持
frame0 = None
frame1 = None
running = True

# カメラ0のスレッド
def capture_cam0():
    global frame0, running
    while running:
        frame0 = cam0.capture_array()

# カメラ1のスレッド
def capture_cam1():
    global frame1, running
    while running:
        frame1 = cam1.capture_array()

# スレッド開始
thread0 = threading.Thread(target=capture_cam0)
thread1 = threading.Thread(target=capture_cam1)
thread0.start()
thread1.start()

print("Press 'q' to quit.")

try:
    while True:
        if frame0 is not None and frame1 is not None:
            # 両カメラのフレームを表示
            cv2.imshow("Camera 0", frame0)
            cv2.imshow("Camera 1", frame1)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
finally:
    # カメラ停止とリソース解放
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
    thread0.join()
    thread1.join()
    print("Program stopped.")
