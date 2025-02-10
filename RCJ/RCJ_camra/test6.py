import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, ch = frame.shape
# 探索窓の初期位置、大きさ
rct = (600, 500, 200, 200)
# MeanShiftの収束条件
cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
while(True):
    th = 160
    ret, frame = cap.read()
    if ret == False:
        break
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_g, th, 255, cv2.THRESH_BINARY)
    # MeanShift
    #ret, rct = cv2.meanShift(img_bin, rct, cri)
    # CamShift
    ret, rct = cv2.CamShift(img_bin, rct, cri)
    x, y, w, h = rct
    # 探索窓を四角形で表示
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),3)
    cv2.imshow("win", frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()