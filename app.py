import os
import cv2
import streamlit as st
from utils.predict import predict_video


TEMP_FILE_NAME = f'output.mp4'
CAMERA_INDEX = 2


def get_video_duration(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def display_tags(words):
    html_tags = ''.join(f'<span style="display:inline-block; background-color:#f63366; color:white; padding:4px; margin:2px; border-radius:8px;">{word}</span>' for word in words)
    st.markdown(html_tags, unsafe_allow_html=True)


def display_and_predict(save_path):
    # Check duration
    duration = get_video_duration(save_path)
    if duration and duration > 5:
        st.error(f"The uploaded video is too long (={duration:.2f} seconds). Please upload a video of 5 seconds or less.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.video(save_path)
            except:
                pass
        with col2:
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    with open(save_path, 'rb') as f:
                        try:
                            pred, prob, total_time = predict_video(save_path, average_sampling=True)
                            st.success("Prediction completed")
                            st.write(f"**Prediction:** {pred}")
                            st.write(f"**Probability:** {prob:.2f}")
                            st.write(f"**Total Time:** {total_time}s")
                        except TypeError as e:
                            if "'NoneType' object is not iterable" in str(e):
                                st.error(f"Face can not be recognized in some frames. Please follow the instructions and record again!")
                        except ValueError as e:
                            if "cannot reshape array of size" in str(e):
                                st.error(f"The video is too short!")


def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def capture_video(video_writer):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        st.error("Failed to open the camera.")
        return
    stframe = st.empty()
    while st.session_state.is_recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break
        video_writer.write(frame)
        stframe.image(frame, channels="BGR")
    cap.release()
    video_writer.release()


def video_recording(save_path):
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    button = st.button("Start/Stop Recording")
    if button:
        if st.session_state.is_recording:
            st.session_state.is_recording = False
        else:
            st.session_state.is_recording = True
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                st.error("Failed to open the camera.")
                return
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            video_writer = cv2.VideoWriter(save_path, fourcc, 25.0, (frame_width, frame_height))
            capture_video(video_writer)
    return save_path


if __name__ == '__main__':
    st.title("Lip Reader")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ('example video', 'upload a video', 'record a video'))
    display_tags(["ABOUT", "ANSWER", "FAMILY", "FRIDAY", "MIDDLE", "PRICE", "RIGHT", "SEVEN", "SOMETHING", "THEIR"])

    # Clear history videos
    def on_reload():
        if os.path.exists(TEMP_FILE_NAME):
            os.remove(TEMP_FILE_NAME)

    if 'reload_flag' not in st.session_state:
        st.session_state['reload_flag'] = False

    if not st.session_state['reload_flag']:
        on_reload()
        st.session_state['reload_flag'] = True

    # Video processing
    if page == 'example video':
        example_videos = {
            "Example 1": "uploaded_videos/1.mp4",
            "Example 2": "uploaded_videos/2.mp4",
        }
        example_video_choice = st.selectbox("Choose an example video", ["None"] + list(example_videos.keys()))
        if example_video_choice != "None":
            save_path = example_videos[example_video_choice]
            display_and_predict(save_path)
    elif page == 'upload a video':
        uploaded_file = st.file_uploader("Please upload a video less than 5 seconds. (Supported format: mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            save_path = save_uploaded_file(uploaded_file, TEMP_FILE_NAME)
            display_and_predict(save_path)
    elif page == 'record a video':
        st.markdown("""
        To ensure correct recognition, please follow below instructions:
        1. Do not be too distant from the camera.
        2. Show complete face, without any cropping or occlusion.
        3. Face the camera upfront and upright.
        """)
        save_path = video_recording(TEMP_FILE_NAME)
        display_and_predict(save_path)