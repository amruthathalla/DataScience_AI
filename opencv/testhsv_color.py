import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.title("🎨 Color Detection App using HSV")

# Input type
option = st.sidebar.selectbox(
    "Choose Input Type",
    ("Upload Image", "Upload Video", "Use Camera")
)

# 🎯 Color Mode Selection
mode = st.sidebar.radio(
    "Color Selection Mode",
    ("Preset Colors", "Custom HSV", "All Colors (Except White)")
)

# 🎯 Preset colors
preset_color = st.sidebar.selectbox(
    "Select Preset Color",
    ("Red", "Blue", "Green")
)

# 🎯 HSV Sliders (for custom mode)
h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
h_max = st.sidebar.slider("Hue Max", 0, 179, 179)
s_min = st.sidebar.slider("Sat Min", 0, 255, 0)
s_max = st.sidebar.slider("Sat Max", 0, 255, 255)
v_min = st.sidebar.slider("Val Min", 0, 255, 0)
v_max = st.sidebar.slider("Val Max", 0, 255, 255)


# 🎯 Detection Function
def detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # PRESET COLORS
    if mode == "Preset Colors":
        if preset_color == "Red":
            low = np.array([161, 155, 84])
            high = np.array([179, 255, 255])
        elif preset_color == "Blue":
            low = np.array([94, 80, 2])
            high = np.array([126, 255, 255])
        elif preset_color == "Green":
            low = np.array([40, 100, 100])
            high = np.array([102, 255, 255])

    # CUSTOM HSV
    elif mode == "Custom HSV":
        low = np.array([h_min, s_min, v_min])
        high = np.array([h_max, s_max, v_max])

    # ALL COLORS EXCEPT WHITE
    elif mode == "All Colors (Except White)":
        low = np.array([0, 42, 0])
        high = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result, mask


# 📤 IMAGE
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result, mask = detect(frame)

        st.image(image, caption="Original", width="stretch")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption="Detected Output", width="stretch")
        st.image(mask, caption="Mask", width="stretch")


# 🎥 VIDEO
elif option == "Upload Video":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result, _ = detect(frame)

            combined = np.hstack((frame, result))
            combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            stframe.image(combined, channels="RGB")

        cap.release()


# 📷 CAMERA
elif option == "Use Camera":
    picture = st.camera_input("Capture Image")

    if picture:
        image = Image.open(picture)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result, mask = detect(frame)

        st.image(image, caption="Captured", width="stretch")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption="Detected Output", width="stretch")
        st.image(mask, caption="Mask", width="stretch")