import numpy as np
import cv2
import matplotlib.pyplot as plt

def bilinear_interpolation(img, x, y):
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, img.shape[1] - 1)
    y1 = min(y0 + 1, img.shape[0] - 1)

    P00 = img[y0, x0]
    P01 = img[y1, x0]
    P10 = img[y0, x1]
    P11 = img[y1, x1]

    dx = x - x0
    dy = y - y0
    
    interpolated_value = (1 - dx) * (1 - dy) * P00 + \
                         dx * (1 - dy) * P10 + \
                         (1 - dx) * dy * P01 + \
                         dx * dy * P11

    return interpolated_value.astype(np.uint8)

def erp2topdown(src, hfov, vfov):
    # ERP 이미지의 가로와 세로 크기를 가져옵니다.
    W, H = src.shape[1], src.shape[0]  
    
    # ERP 이미지를 기반으로 초점 거리(구의 반지름)을 계산합니다.
    f = W / (2 * np.pi)  
    
    # 최종 top-down 이미지의 가로와 세로 크기를 계산합니다.
    W_prime = int(2 * f * np.tan(hfov / 2) + 0.5)
    H_prime = int(2 * f * np.tan(vfov / 2) + 0.5)
    
    # top-down 이미지의 중심 좌표를 계산합니다.
    cx_prime, cy_prime = W_prime // 2, H_prime // 2  
    
    # ERP 이미지의 중심 좌표를 계산합니다.
    cx, cy = W // 2, H // 2  
    
    # top-down 이미지를 저장할 빈 배열을 생성합니다. 
    top_view_image = np.zeros((H_prime, W_prime, 3), dtype=np.uint8)  
    
    # 모든 픽셀에 대해 반복을 수행합니다.
    for y_prime in range(H_prime):
        for x_prime in range(W_prime):
            # 중심에서 현재 픽셀까지의 거리를 계산합니다.
            D = np.sqrt((x_prime - cx_prime)**2 + (y_prime - cy_prime)**2)
            
            # 거리가 0인 경우, 즉 중심인 경우 ERP 이미지의 중심 픽셀 값을 사용합니다.
            if D == 0:
                top_view_image[y_prime, x_prime] = src[H-1, 0]
                continue
            
            # theta 각도를 계산합니다.
            theta = np.arctan2(x_prime - cx_prime, cy_prime - y_prime)
            
            # phi 각도를 계산합니다.
            phi = np.arctan(f / D)
            
            # ERP 이미지에서의 좌표를 계산합니다.
            erp_x = theta * W / (2 * np.pi) + cx
            erp_y = phi * H / np.pi + cy
            
            # 좌표의 범위를 이미지 크기 내로 제한합니다.
            erp_x = np.clip(erp_x, 0, W - 1)
            erp_y = np.clip(erp_y, 0, H - 1)
            
            # bilinear_interpolation 함수를 사용하여 ERP 이미지에서 해당 좌표의 픽셀 값을 가져와 top-down 이미지에 저장합니다.
            top_view_image[y_prime, x_prime] = bilinear_interpolation(src, erp_x, erp_y)
    
    # 최종 top-down 이미지를 반환합니다.
    return top_view_image

file = "./erp.png"
erp_image = cv2.imread(file)

# 시야각을 라디안으로 변환합니다.
hfov_topdown, vfov_topdown = np.deg2rad(90), np.deg2rad(90)

# ERP 이미지를 top-down 뷰로 변환하는 함수를 호출하고 결과를 저장합니다.
topdown_view = erp2topdown(erp_image, hfov_topdown, vfov_topdown)

# 변환된 top-down 이미지를 화면에 출력합니다.
image_title_topdown = f"Top-Down View: hfov={np.rad2deg(hfov_topdown):.2f}°, vfov={np.rad2deg(vfov_topdown):.2f}°"
plt.imshow(topdown_view)
plt.title(image_title_topdown)
plt.axis('off')
plt.show()
