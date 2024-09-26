import cv2
import os
import re

# 人脸识别模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



# 获取所有图片文件
#image_files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

# 指定 images 目录
image_directory = 'images'

# 获取 images 目录下的所有图片文件
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

# 根据文件名序号进行排序
image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
print("找到的图片文件：", image_files)




# 初始化记录上下左右最小值的变量
min_top = float('inf')
min_bottom = float('inf')
min_left = float('inf')
min_right = float('inf')

# 存储眼睛的位置和距离图片边界的距离
eye_positions = []

# 遍历所有图片，检测人脸和眼睛位置
for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    print(f"img:  {image_file}")
    if len(faces) > 0:
        # 取第一个检测到的人脸
        x, y, w, h = faces[0]
        face_region = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_region, 1.1, 4)
        print(f"img:  {image_file}")

        if len(eyes) > 0:
            # 获取第一个眼睛的位置
            eye_x, eye_y, eye_w, eye_h = eyes[0]
            eye_center_x = x + eye_x + eye_w // 2
            eye_center_y = y + eye_y + eye_h // 2
            


            # 计算眼睛到图片上下左右边界的距离
            top_distance = eye_center_y
            bottom_distance = img.shape[0] - eye_center_y
            left_distance = eye_center_x
            right_distance = img.shape[1] - eye_center_x

            # 更新最小距离
            min_top = min(min_top, top_distance)
            min_bottom = min(min_bottom, bottom_distance)
            min_left = min(min_left, left_distance)
            min_right = min(min_right, right_distance)

            # 记录当前图片的眼睛位置和边界距离
            eye_positions.append((image_file, eye_center_x, eye_center_y))


# 创建输出文件夹
output_directory = 'out'
os.makedirs(output_directory, exist_ok=True)

# 根据最小距离进行裁剪
for i, (image_file, eye_center_x, eye_center_y) in enumerate(eye_positions):
    img = cv2.imread(image_file)

    # 计算裁剪区域的上下左右坐标
    crop_x1 = max(0, eye_center_x - min_left)
    crop_x2 = min(img.shape[1], eye_center_x + min_right)
    crop_y1 = max(0, eye_center_y - min_top)
    crop_y2 = min(img.shape[0], eye_center_y + min_bottom)

    # 裁剪图片
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 保存裁剪后的图片
    #cv2.imwrite(f"cropped_{i}.jpg", cropped_img)
    #print(f"Cropped and saved image {i}: {image_file}")


    # 保存裁剪后的图片到指定路径
    output_path = os.path.join(output_directory, f"cropped_{i}.jpg")
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped and saved image {i}: {output_path} :{image_file}")

print("所有图片裁剪完成。")
