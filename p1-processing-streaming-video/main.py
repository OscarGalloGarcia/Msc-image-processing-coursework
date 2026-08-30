import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Controls")
cv2.createTrackbar("Angle",  "Controls", 0,     360,   nothing)
cv2.createTrackbar("TX",     "Controls", w,     2 * w, nothing)
cv2.createTrackbar("TY",     "Controls", h,     2 * h, nothing)
cv2.createTrackbar("Scale",  "Controls", 100,   300,   nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    angle = cv2.getTrackbarPos("Angle", "Controls")
    tx    = cv2.getTrackbarPos("TX", "Controls") - w      # back to -w..+w
    ty    = cv2.getTrackbarPos("TY", "Controls") - h
    scale = max(cv2.getTrackbarPos("Scale", "Controls"), 1) / 100.0

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(frame, M, (w, h))

    cv2.imshow("Controls", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
