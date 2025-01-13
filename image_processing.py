import cv2
import os
import re
import numpy as np

# 人脸识别模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_and_sort_images(image_directory):
    # 获取 images 目录下的所有图片文件
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    # 根据文件名序号进行排序
    return sorted(image_files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

def resize_images(image_files, target_width):
    resized_images = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        height, width = img.shape[:2]
        new_height = int(height * target_width / width)
        resized_img = cv2.resize(img, (target_width, new_height))
        resized_images.append(resized_img)
    return resized_images

def detect_faces_and_positions(resized_images):
    min_top, min_bottom, min_left, min_right = float('inf'), float('inf'), float('inf'), float('inf')
    head_positions = []
    max_face_width, max_face_height = 0, 0

    for i, resized_img in enumerate(resized_images):
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            max_face_width, max_face_height = max(max_face_width, w), max(max_face_height, h)
            head_center_x, head_center_y = x + w // 2, y + h // 2

            top_distance, bottom_distance = head_center_y, resized_img.shape[0] - head_center_y
            left_distance, right_distance = head_center_x, resized_img.shape[1] - head_center_x

            min_top, min_bottom, min_left, min_right = min(min_top, top_distance), min(min_bottom, bottom_distance), min(min_left, left_distance), min(min_right, right_distance)

            head_positions.append((i, head_center_x, head_center_y))

    return head_positions, (min_top, min_bottom, min_left, min_right), (max_face_width, max_face_height)

def crop_images(resized_images, head_positions, min_distances):
    cropped_images = []
    for i, (index, head_center_x, head_center_y) in enumerate(head_positions):
        img = resized_images[index]

        crop_x1, crop_x2 = max(0, head_center_x - min_distances[2]), min(img.shape[1], head_center_x + min_distances[3])
        crop_y1, crop_y2 = max(0, head_center_y - min_distances[0]), min(img.shape[0], head_center_y + min_distances[1])

        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_images.append(cropped_img)

    return cropped_images

def overlay_head_on_images(cropped_images, head_image):
    resized_heads = []
    for cropped_img in cropped_images:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            # 根据当前脸部大小调整头像大小
            scale_factor = max(w, h) / max(*head_image.shape[:2]) * 2.5
            new_head_width, new_head_height = int(head_image.shape[1] * scale_factor), int(head_image.shape[0] * scale_factor)
            resized_head = cv2.resize(head_image, (new_head_width, new_head_height), interpolation=cv2.INTER_AREA)

            # 计算放置头像的左上角坐标
            overlay_x, overlay_y = x + (w - new_head_width) // 2, y + (h - new_head_height) // 2

            # 确保头像不会超出裁剪图片的边界
            if overlay_y < 0: overlay_y, new_head_height = 0, new_head_height + overlay_y
            if overlay_x < 0: overlay_x, new_head_width = 0, new_head_width + overlay_x
            if overlay_x + new_head_width > cropped_img.shape[1]: new_head_width = cropped_img.shape[1] - overlay_x
            if overlay_y + new_head_height > cropped_img.shape[0]: new_head_height = cropped_img.shape[0] - overlay_y

            alpha_head = resized_head[:new_head_height, :new_head_width, 3] / 255.0
            for c in range(3):
                cropped_img[overlay_y:overlay_y+new_head_height, overlay_x:overlay_x+new_head_width, c] = (
                    alpha_head * resized_head[:new_head_height, :new_head_width, c] +
                    (1 - alpha_head) * cropped_img[overlay_y:overlay_y+new_head_height, overlay_x:overlay_x+new_head_width, c]
                )

        resized_heads.append(cropped_img)

    return resized_heads


def create_video_from_images(images, output_path, fps=1.0, duration_per_image=2.0):
    if not images:
        print("No images to create video.")
        return


    # 获取第一张图片的尺寸
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 计算每个图像需要写入多少次以达到指定的持续时间
    frames_per_image = int(fps * duration_per_image)

    # 调试信息
    total_frames_written = 0

    for i, image in enumerate(images):
        for _ in range(frames_per_image):
            out.write(image)
            total_frames_written += 1
        print(f"Image {i + 1} written, total frames: {total_frames_written}")

    # 确保最后一张图片也被正确写入
    for _ in range(frames_per_image):
        out.write(images[-1])
        total_frames_written += 1
    print(f"Last image written, total frames: {total_frames_written}")

    out.release()
    expected_duration = total_frames_written / fps
    print(f"Video saved to {output_path}. Expected duration: {expected_duration:.2f} seconds")

def process_images(image_directory, output_directory, overlay_head=False):
    os.makedirs(output_directory, exist_ok=True)

    image_files = load_and_sort_images(image_directory)
    first_image = cv2.imread(image_files[0])
    target_width = first_image.shape[1]
    resized_images = resize_images(image_files, target_width)

    head_positions, min_distances, max_face_size = detect_faces_and_positions(resized_images)
    cropped_images = crop_images(resized_images, head_positions, min_distances)

    if overlay_head:
        head_image = cv2.imread('head/head-no.png', cv2.IMREAD_UNCHANGED)
        cropped_images = overlay_head_on_images(cropped_images, head_image)

    for i, cropped_img in enumerate(cropped_images):
        output_path = os.path.join(output_directory, f"cropped_with_head_{i}.jpg")
        cv2.imwrite(output_path, cropped_img)
        print(f"Cropped and saved image {i} with head: {output_path}")

    # 创建视频
    video_output_path = os.path.join(output_directory, "output_video.mp4")
    create_video_from_images(cropped_images, video_output_path, fps=1.0 ,duration_per_image=2.0)

    print("All images have been processed and video created.")