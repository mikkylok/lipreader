# import streamlit as st
#
# picture = st.camera_input("Take a picture")
#
# if picture:
#     st.image(picture)

#
# import streamlit as st
# import cv2
# import threading
#
# # Function to capture and display video frames
# def capture_video(stop_event, video_writer):
#     cap = cv2.VideoCapture(2)  # Open the default camera
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.write("Failed to capture video frame.")
#             break
#         if stop_event.is_set():
#             break
#         video_writer.write(frame)
#         st.image(frame, channels="BGR")
#     cap.release()
#     video_writer.release()
#
# # Home page function
# def home():
#     st.title("Home Page")
#     st.write("Welcome to the home page!")
#
#     picture = st.camera_input("Take a picture")
#
#     if picture:
#         st.image(picture)
#
# # Video recording page function
# def video_recording():
#     st.title("Video Recording Page")
#
#     # Use session state to keep track of the recording state
#     if 'is_recording' not in st.session_state:
#         st.session_state.is_recording = False
#
#     # Button to start and stop recording
#     if st.button("Start/Stop Recording"):
#         if st.session_state.is_recording:
#             # Stop recording
#             st.session_state.is_recording = False
#         else:
#             # Start recording
#             st.session_state.is_recording = True
#
#     if st.session_state.is_recording:
#         st.write("Recording...")
#         # Get the default resolutions
#         # Initialize video writer with MP4 codec
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
#
#         stop_event = threading.Event()
#         try:
#             capture_video(stop_event, video_writer)
#         finally:
#             stop_event.set()
#             video_writer.release()
#     else:
#         st.write("Recording stopped.")
#
# # Main app
# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home", "Video Recording"])
#
#     if page == "Home":
#         home()
#     elif page == "Video Recording":
#         video_recording()
#
# if __name__ == "__main__":
#     main()
#

import streamlit as st
import cv2

# Function to capture and display video frames
def capture_video(video_writer):
    cap = cv2.VideoCapture(2)  # Open the default camera (adjust index if needed)
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

# Home page function
def home():
    st.title("Home Page")
    st.write("Welcome to the home page!")

    picture = st.camera_input("Take a picture")

    if picture:
        st.image(picture)

# Video recording page function
def video_recording():
    st.title("Video Recording Page")

    # Use session state to keep track of the recording state
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    # Input box for naming the MP4 file
    file_name = st.text_input("Enter the name of the video file (without extension):", "output")

    # Ensure file_name is not empty
    if not file_name.strip():
        st.error("Please enter a valid file name.")
        return

    # Button to start and stop recording
    if st.button("Start/Stop Recording"):
        if st.session_state.is_recording:
            # Stop recording
            st.session_state.is_recording = False
        else:
            # Start recording
            st.session_state.is_recording = True

            # Initialize video writer with MP4 codec
            cap = cv2.VideoCapture(2)  # Ensure cap is initialized here
            if not cap.isOpened():
                st.error("Failed to open the camera.")
                return

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            video_writer = cv2.VideoWriter(f'{file_name}.mp4', fourcc, 20.0, (frame_width, frame_height))

            capture_video(video_writer)

    if not st.session_state.is_recording:
        st.write("Recording stopped.")

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Home", "Video Recording"])

    if page == "Home":
        home()
    elif page == "Video Recording":
        video_recording()

if __name__ == "__main__":
    main()
