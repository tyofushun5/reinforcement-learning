from picamera2 import Picamera2, Preview

# Picamera2インスタンスの作成
picam2 = Picamera2()

# カメラ設定
camera_config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},  # 出力解像度（広角レンズに適した解像度を設定）
    transform={"rotation": 0, "hflip": False, "vflip": False}  # 必要に応じて反転
)

picam2.configure(camera_config)

# プレビュー開始
picam2.start_preview(Preview.QTGL)  # QTGLでプレビューを表示
picam2.start()
