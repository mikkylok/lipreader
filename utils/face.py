import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='model/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def get_lip_landmark_mediapipe(img, visualize=True):
    """
    Using MediaPipe to extract lip landmark features.
    Number of points: 40
    :param img:
    :return:
    """
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(image)

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
    unique_indices = set([idx for pair in lip_connections for idx in pair])  # Flatten and remove duplicates

    if len(detection_result.face_landmarks) != 0:
        for i, landmark in enumerate(detection_result.face_landmarks[0]):
            if i in unique_indices:
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
        for i, (scaled_x, scaled_y) in enumerate(scaled_lip_points):
            cv2.circle(img, (int(scaled_x * scale), int(scaled_y * scale)), 3, (255, 255, 0), -1)

        return lip_points, scaled_lip_points, img
    else:
        return None, None, None