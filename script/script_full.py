# 2. extract frames according to the decided sample rate and save them in a image folder
# 3. extract lip points from these frames and save them in three npy files: train, valid, test
# [
# (label, video_name, [[40 lip points], [40 lip points], ...]),
# (label, vidoe_name, [[40 lip points], [40 lip points], ...]),
# ]
# 4. modelling, read from this file
import os
import cv2
import pickle
import time
import warnings
import mediapipe as mp


def get_lip_landmark_mediapipe(img, visualize=True):
    """
    Using MediaPipe to extract lip landmark features.
    Number of points: 40
    :param img:
    :return:
    """
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Face Mesh module
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Process the image and find faces
    results = mp_face_mesh.process(image_rgb)

    if visualize:
        # Prepare to collect landmark data
        visualize_lip_points = []

    lip_points = []

    lip_connections = [
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
        (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
        (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
    ]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            unique_indices = set([idx for pair in lip_connections for idx in pair])  # Flatten and remove duplicates

            # Get both outer and inner lip ranges
            for i in unique_indices:
                landmark = face_landmarks.landmark[i]
                lip_points.append((landmark.x, landmark.y))

                if visualize:
                    # Draw lip landmarks on the lip
                    visualize_x = landmark.x * img.shape[1]
                    visualize_y = landmark.y * img.shape[0]
                    cv2.circle(img, (int(visualize_x), int(visualize_y)), 2, (0, 255, 0), -1)

            # Min-max scale lip points
            min_x = min(lip_points, key=lambda p: p[0])[0]
            max_x = max(lip_points, key=lambda p: p[0])[0]
            min_y = min(lip_points, key=lambda p: p[1])[1]
            max_y = max(lip_points, key=lambda p: p[1])[1]
            scaled_lip_points = [
                (
                    (x - min_x) / (max_x - min_x) if max_x != min_x else 0,
                    (y - min_y) / (max_y - min_y) if max_y != min_y else 0,
                )
                for (x, y) in lip_points
            ]

            # Draw scaled lip points
            scale = 100
            for (scaled_x, scaled_y) in scaled_lip_points:
                cv2.circle(img, (int(scaled_x * scale), int(scaled_y * scale)), 3, (255, 255, 0), -1)

        return scaled_lip_points, img
    else:
        return None, None


# Suppress specific warnings
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.*")


def get_lip_points_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
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

        # Save individual sampled frames to the specified directory
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lip_points_frame, img = get_lip_landmark_mediapipe(img, visualize=True)
        # _, cropped_img = detect_face(img)
        # lip_points_frame, img = get_lip_landmark_mediapipe(cropped_img, visualize=True)
        time.sleep(1)
        if lip_points_frame is None:
            missing_frame = True
            break
        else:
            img_list.append(img)
            lip_points_video.extend(lip_points_frame)
    # Release the video capture object
    cap.release()
    if not missing_frame:
        return lip_points_video, img_list
    else:
        return None, None


if __name__ == "__main__":
    while True:
        try:
            video_dir_path = '/Users/mikky/lipread_mp4'
            label_names = [label for label in os.listdir(video_dir_path) if label != '.DS_Store']
            train_pickle_output_dir = '../pickle/train'
            val_pickle_output_dir = '../pickle/val'
            test_pickle_output_dir = '../pickle/test'
            train_frame_output_dir = '../lip_points_frame/train'
            val_frame_output_dir = '../lip_points_frame/val'
            test_frame_output_dir = '../lip_points_frame/test'

            for label in label_names:
                print(label)
                source_directory = os.path.join('../dataset/', label)

                # train
                if label in ['RIGHT', 'MORNING', 'ORDER']:
                    continue
                train_directory = os.path.join(source_directory, 'train/')
                label_folder = os.path.join(train_pickle_output_dir, label.split('.')[0])
                if os.path.exists(label_folder):
                    extracted_files = [pkl_file.split('.')[0] for pkl_file in os.listdir(label_folder) if label != '.DS_Store']
                else:
                    extracted_files = []
                video_files = [file for file in os.listdir(train_directory) if file.endswith('.mp4')]
                for video_file in video_files:
                    if video_file.split('.')[0] not in extracted_files:
                        print('train:', video_file)
                        video_file_path = os.path.join(train_directory, video_file)
                        lip_points, img_list = get_lip_points_video(video_file_path)
                        if lip_points is None:
                            print('train:', video_file, ':not complete')
                            continue
                        else:
                            # create label folder
                            if not os.path.exists(label_folder):
                                os.mkdir(label_folder)
                            train_data = (label.split('.')[0], video_file, lip_points)
                            pickle_path = os.path.join(label_folder, video_file.split('.')[0]+'.pkl')
                            with open(pickle_path, 'wb') as file:
                                pickle.dump(train_data, file)
                            for i, img in enumerate(img_list):
                                img_name = video_file.split('.')[0] + '_' + str(i) + '.jpg'
                                img_path = os.path.join(train_frame_output_dir, img_name)
                                cv2.imwrite(img_path, img)
                        time.sleep(1)


                # valid
                if label not in ['RIGHT','MORNING', 'ORDER', 'MONTH']:
                    continue
                val_directory = os.path.join(source_directory, 'val/')
                label_folder = os.path.join(val_pickle_output_dir, label.split('.')[0])
                if os.path.exists(label_folder):
                    extracted_files = [pkl_file.split('.')[0] for pkl_file in os.listdir(label_folder) if label != '.DS_Store']
                else:
                    extracted_files = []
                video_files = [file for file in os.listdir(val_directory) if file.endswith('.mp4')]
                for video_file in video_files:
                    if video_file.split('.')[0] not in extracted_files:
                        print('val:', video_file)
                        video_file_path = os.path.join(val_directory, video_file)
                        lip_points, img_list = get_lip_points_video(video_file_path)
                        if lip_points is None:
                            print('val:', video_file, ':not complete')
                            continue
                        else:
                            # create label folder
                            if not os.path.exists(label_folder):
                                os.mkdir(label_folder)
                            train_data = (label.split('.')[0], video_file, lip_points)
                            pickle_path = os.path.join(label_folder, video_file.split('.')[0]+'.pkl')
                            with open(pickle_path, 'wb') as file:
                                pickle.dump(train_data, file)
                            for i, img in enumerate(img_list):
                                img_name = video_file.split('.')[0] + '_' + str(i) + '.jpg'
                                img_path = os.path.join(val_frame_output_dir, img_name)
                                cv2.imwrite(img_path, img)
                        time.sleep(1)

                # test
                # if label not in ['RIGHT', 'MORNING', 'ORDER', 'MONTH']:
                #     continue
                # test_directory = os.path.join(source_directory, 'test/')
                # label_folder = os.path.join(test_pickle_output_dir, label.split('.')[0])
                # if os.path.exists(label_folder):
                #     extracted_files = [pkl_file.split('.')[0] for pkl_file in os.listdir(label_folder) if label != '.DS_Store']
                # else:
                #     extracted_files = []
                # video_files = [file for file in os.listdir(test_directory) if file.endswith('.mp4')]
                # for video_file in video_files:
                #     if video_file.split('.')[0] not in extracted_files:
                #         print('test:', video_file)
                #         video_file_path = os.path.join(test_directory, video_file)
                #         lip_points, img_list = get_lip_points_video(video_file_path)
                #         if lip_points is None:
                #             print('test:', video_file, ':not complete')
                #             continue
                #         else:
                #             # create label folder
                #             if not os.path.exists(label_folder):
                #                 os.mkdir(label_folder)
                #             train_data = (label.split('.')[0], video_file, lip_points)
                #             pickle_path = os.path.join(label_folder, video_file.split('.')[0]+'.pkl')
                #             with open(pickle_path, 'wb') as file:
                #                 pickle.dump(train_data, file)
                #             for i, img in enumerate(img_list):
                #                 img_name = video_file.split('.')[0] + '_' + str(i) + '.jpg'
                #                 img_path = os.path.join(test_frame_output_dir, img_name)
                #                 cv2.imwrite(img_path, img)
                #         time.sleep(1)
        except:
            print("Handled error - continuing...")
            time.sleep(10)  # Pause the execution for a second to prevent a tight loop