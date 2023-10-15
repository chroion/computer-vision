import numpy as np
import cv2

# 체커보드의 모서리 수 설정
corner_x = 4
corner_y = 4

# 3D 좌표를 저장할 배열
objpoints = [] 
imgpoints = [] 

# 체커보드의 모서리 좌표 생성
objp = np.zeros((corner_x*corner_y, 3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

# AR 큐브를 그리는 함수
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the ground in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    if ret == True:
        if len(imgpoints) <= 20:
            objpoints.append(objp)
            imgpoints.append(corners)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # 카메라 pose 추정
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        
        # AR 큐브의 좌표 정의
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        
        frame = draw_cube(frame, corners, imgpts)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()