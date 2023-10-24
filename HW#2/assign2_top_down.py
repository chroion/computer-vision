import numpy as np
import cv2
import matplotlib.pyplot as plt

# D가 무한대가 되는 경우를 처리하기 위해 업데이트된 함수
def erp2topdown(src, hfov, vfov):
    # ERP 이미지의 가로, 세로 해상도
    W, H = src.shape[1], src.shape[0]
    
    # 초점 거리(구의 반지름) 계산
    f = W / (2 * np.pi)
    
    # 탑다운 이미지 해상도 계산
    W_prime = int(2 * f * np.tan(hfov / 2) + 0.5)
    H_prime = int(2 * f * np.tan(vfov / 2) + 0.5)
    
    # 탑다운 이미지의 중심 좌표 계산
    cx_prime, cy_prime = W_prime // 2, H_prime // 2
    
    # 결과 이미지를 저장할 배열 초기화
    top_view_image = np.zeros((H_prime, W_prime, 3), dtype=np.uint8)
    
    # ERP 이미지의 중심 좌표
    cx, cy = W // 2, H // 2
    
    # D의 최대 값 설정하여 무한대 값을 방지
    max_D = W_prime
    
    # 모든 픽셀에 대해 반복
    for y in range(H):
        # 수직 각도 계산
        phi = (y - cy) * np.pi / H
        
        # 중심부터의 거리 D 계산
        D = f / np.tan(phi)
        
        # D가 max_D보다 큰 경우 계산 생략
        if abs(D) > max_D:
            continue
        
        for x in range(W):
            # 수평 각도 계산
            theta = (x - cx) * 2 * np.pi / W
            
            # 탑다운 이미지에서의 픽셀 좌표 계산
            x_prime = int(cx_prime + D * np.sin(theta))
            y_prime = int(cy_prime - D * np.cos(theta))
            
            # 좌표를 이미지 크기 내로 제한
            x_prime = max(0, min(x_prime, W_prime - 1))
            y_prime = max(0, min(y_prime, H_prime - 1))
            
            # 탑다운 이미지에 픽셀 값 할당
            top_view_image[y_prime, x_prime] = src[y, x]
    
    return top_view_image

# ERP 이미지 파일 경로
file = "./erp.png"

# ERP 이미지 불러오기
erp_image = cv2.imread(file)

# 시야각을 라디안으로 변환
hfov_topdown, vfov_topdown = np.deg2rad(90), np.deg2rad(90)

# ERP 이미지를 탑다운 뷰로 변환
topdown_view = erp2topdown(erp_image, hfov_topdown, vfov_topdown)

# 변환된 탑다운 이미지 출력
image_title_topdown = f"Top-Down View: hfov={np.rad2deg(hfov_topdown):.2f}°, vfov={np.rad2deg(vfov_topdown):.2f}°"
plt.imshow(topdown_view)
plt.title(image_title_topdown)
plt.axis('off')
plt.show()
