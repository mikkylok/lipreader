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
from utils.video import get_lip_points_video
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference.*")


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
                if label in ['RIGHT', 'MORNING', 'ORDER', 'MONTH']:
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
                # if label not in ['RIGHT','MORNING', 'ORDER', 'MONTH']:
                #     continue
                # val_directory = os.path.join(source_directory, 'val/')
                # label_folder = os.path.join(val_pickle_output_dir, label.split('.')[0])
                # if os.path.exists(label_folder):
                #     extracted_files = [pkl_file.split('.')[0] for pkl_file in os.listdir(label_folder) if label != '.DS_Store']
                # else:
                #     extracted_files = []
                # video_files = [file for file in os.listdir(val_directory) if file.endswith('.mp4')]
                # for video_file in video_files:
                #     if video_file.split('.')[0] not in extracted_files:
                #         print('val:', video_file)
                #         video_file_path = os.path.join(val_directory, video_file)
                #         lip_points, img_list = get_lip_points_video(video_file_path)
                #         if lip_points is None:
                #             print('val:', video_file, ':not complete')
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
                #                 img_path = os.path.join(val_frame_output_dir, img_name)
                #                 cv2.imwrite(img_path, img)
                #         time.sleep(1)

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