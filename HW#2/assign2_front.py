from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

path = os.getcwd()
file = "./erp.png"

# ERP 이미지 로드
erp_image = cv2.imread(file)

# ERP를 직사각형 (정면 뷰) 이미지로 변환하는 함수
def erp2rect(src, theta, hfov, vfov):
    # 초점 길이 계산
    f = src.shape[1] / (2 * np.pi)
    
    # 평면 이미지의 크기 계산
    front_view_image_rows = int(2 * f * np.tan(vfov / 2) + 0.5)
    front_view_image_cols = int(2 * f * np.tan(hfov / 2) + 0.5)
    
    # 평면 이미지 초기화
    front_view_image = np.zeros((front_view_image_rows, front_view_image_cols, 3), dtype=np.uint8)
    front_view_image_cx = front_view_image_cols / 2
    front_view_image_cy = front_view_image_rows / 2
    
    for x in range(front_view_image_cols):
        xth = np.arctan((x - front_view_image_cx) / f)
        # 원본 이미지에서의 x 좌표 계산
        src_x = int((xth + theta) * src.shape[0] / np.pi + 0.5) % src.shape[1]
        yf = f / np.cos(xth)
        
        for y in range(front_view_image_rows):
            yth = np.arctan((y - front_view_image_cy) / yf)
            # 원본 이미지에서의 y 좌표 계산
            src_y = int(yth * src.shape[0] / np.pi + src.shape[0] / 2 + 0.5)
            src_y = max(0, min(src_y, src.shape[0] - 1))  # y 좌표를 이미지 내로 제한
            front_view_image[y, x] = src[src_y, src_x]
    
    return front_view_image

# ERP를 정면 뷰로 변환
theta = np.deg2rad(0)  # 각도를 라디안으로 변환
hfov = np.deg2rad(120)  # 각도를 라디안으로 변환
vfov = np.deg2rad(90)  # 각도를 라디안으로 변환
front_view = erp2rect(erp_image, theta, hfov, vfov)

# 정면 뷰 이미지 표시
image_title = f"Front View: θ={np.rad2deg(theta):.2f}°, hfov={np.rad2deg(hfov):.2f}°, vfov={np.rad2deg(vfov):.2f}°"
plt.imshow(front_view)
plt.title(image_title)
plt.axis('off')
plt.show()
