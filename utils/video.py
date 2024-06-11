import cv2
from utils.face import get_lip_landmark_mediapipe


def systematic_sampling(data, sample_count=29):
    n = len(data)
    if n < sample_count:
        return data, list(range(n))  # Return all data and their indices if not enough data
    k = n // sample_count
    if k <= 0:
        return data, list(range(n))  # Return all data and their indices if k is zero or negative
    start = np.random.randint(k)  # Random start within the first interval
    sampled_indices = list(range(start, n, k))[:sample_count]
    sampled_data = [data[i] for i in sampled_indices]
    return sampled_data, sampled_indices


def get_lip_points_video(video_path, average_sampling=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    lip_points_video = []
    img_list = []
    missing_frame = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, lip_points_frame, img = get_lip_landmark_mediapipe(img, visualize=True)
        # _, cropped_img = detect_face(img)
        # lip_points_frame, img = get_lip_landmark_mediapipe(cropped_img, visualize=True)
        if lip_points_frame is None:
            missing_frame = True
            break
        else:
            img_list.append(img)
            lip_points_video.append(lip_points_frame)
    cap.release()
    if not missing_frame:
        if average_sampling:
            sampled_imgs, sampled_indices = systematic_sampling(img_list)
            sampled_lip_points = [lip_points_video[i] for i in sampled_indices]
        return sampled_lip_points, sampled_imgs
    else:
        return None, None