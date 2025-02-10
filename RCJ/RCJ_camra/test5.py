import cv2

# 抜き出す特徴点の数
COUNT = 200
# 特徴点を探す時の収束条件
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03)
# Lucas-Kanadeに用いるパラメーター
lk_params = dict(winSize=(10, 10), maxLevel=4, criteria=criteria)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 追うべき特徴点を探す
    feature_pre = cv2.goodFeaturesToTrack(frame_pre, COUNT, 0.001, 5)
    if feature_pre is None or len(feature_pre) == 0:
        continue

    # オプティカルフロー
    feature_now, status, err = cv2.calcOpticalFlowPyrLK(frame_pre, frame_now, feature_pre, None, **lk_params)
    if feature_now is None or len(feature_now) == 0:
        continue

    for i in range(len(feature_now)):
        if status[i][0] == 1:  # 追跡に成功した点だけを描画
            pre_x = int(feature_pre[i][0][0])
            pre_y = int(feature_pre[i][0][1])
            now_x = int(feature_now[i][0][0])
            now_y = int(feature_now[i][0][1])
            cv2.line(frame, (pre_x, pre_y), (now_x, now_y), (255, 0, 0), 3)

    cv2.imshow("img", frame)
    frame_pre = frame_now.copy()

    key = cv2.waitKey(10)
    if key == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
