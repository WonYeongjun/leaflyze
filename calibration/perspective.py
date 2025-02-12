#저장된 이미지를 가져와서 원근감 보정함
import cv2
import numpy as np
import json

with open("config.json", "r") as f:
    config = json.load(f)

def correct_perspective(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return
    
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, _ = detector.detectMarkers(gray)
    
    desired_ids = config["ArUco_list"]
    
    if ids is not None:
        marker_points = []
        
        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]
                    # 이미지 중심과 가장 가까운 꼭짓점 선택
                    closest_corner = min(marker_corners, key=lambda pt: np.linalg.norm(pt - image_center))
                    marker_points.append(closest_corner)
                    break
        
        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")
            print(f"pts1 (selected marker corners): {pts1}")
            
            width, height = config["ArUco_width"], config["ArUco_height"]
            pts2 = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            
            matrix, _ = cv2.findHomography(pts1, pts2)
            
            dst = cv2.warpPerspective(image, matrix, (width, height))
            
            cv2.imwrite(output_path, dst)
            print(f"Corrected image saved to {output_path}")
        else:
            print("Not enough markers detected to calculate perspective.")
    else:
        print("No ArUco markers detected.")

# 실행
test_image_path = config["pc_file_path"]
output_image_path = config["pc_modified_file_path"]
correct_perspective(test_image_path, output_image_path)