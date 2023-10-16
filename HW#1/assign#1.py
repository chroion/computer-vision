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

    # 바닥을 초록색으로 그린다
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # 기둥을 파란색으로 그린다
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # 윗부분을 빨간색으로 그린다
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# 점들을 투영하는 함수
def project_points(axis, rvec, tvec, mtx, dist):
    # 회전 벡터를 회전 행렬로 변환한다
    R, _ = cv2.Rodrigues(rvec)

    # 외부 행렬 [R | t] 구성한다
    RT = np.hstack((R, tvec))

    # 투영 행렬 P = K [R | t]를 계산한다
    P = np.dot(mtx, RT)

    # 3D 점들을 2D로 투영한다
    imgpts = []
    for point in axis:
        # 점을 homogeneous 좌표로 확장한다
        point_homogeneous = np.append(point, 1)
        
        # 투영 행렬을 사용하여 점을 투영한다
        imgpt_homogeneous = np.dot(P, point_homogeneous)
        
        # 픽셀 좌표로 다시 변환한다
        imgpt = imgpt_homogeneous[:2] / imgpt_homogeneous[2]
        
        imgpts.append(imgpt)

    return np.array(imgpts)


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
        imgpts = project_points(axis, rvec, tvec, mtx, dist)
        
        frame = draw_cube(frame, corners, imgpts)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()