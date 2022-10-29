import streamlit as st
import cv2
import numpy as np
from MoveNet_Processing_Utils import movenet_processing
import tempfile
import time
from PIL import Image

DEMO_IMAGE = "./demos/demo.jpg"
DEMO_VIDEO = "./demos/demo.mp4"

st.title('Pose Estimation and Classification with MoveNet')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div: first-child {
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div: first-child {
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('MoveNet App Sidebar')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Note: inter is for interpolating the image (to shrink it)
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        ratio = width/float(w)
        dim = (int(w * ratio), height)
    else:
        ratio = width/float(w)
        dim = (width, int(h * ratio))

    # Resize image
    return cv2.resize(image, dim, interpolation=inter)


app_mode = st.sidebar.selectbox('Select App Mode', ['Home', 'Run on Image', 'Run on Video'])


if app_mode == 'Home':
    st.markdown('## About App')
    st.markdown('This web-application utilises **MoveNet** to perform **pose estimation** on an input image or video. \
        The coordinates obtained by MoveNet are then passed to a **custom classifying algorithm** to perform **pose classification**. \
        **Streamlit** was used to create the **web-GUI**.')
    st.markdown('The **pose classes** involved are:\n1) Standing\n2) Sitting\n3) Lying')
    st.markdown('## Instructions')
    st.markdown('You may run the models on **either images or video** (see sidebar). In each case, you may either use the \
        provided demo image/video, or **upload** your own image/video **through the sidebar**. For the **video** tab, you \
        may also use **your device webcam**.')
    st.markdown('The settings that can be modified include:\n1) **Draw MoveNet Skeleton**\n&emsp;&emsp;&emsp;Whether skeletons obtained \
        from MoveNet are displayed on the image.')
    st.markdown('2) **Maximum Number of People**\n&emsp;&emsp;&emsp;The maximum number \
        of people MoveNet will detect (Ranges from 1 to 6).')
    st.markdown('3) **Minimum Person Detection Confidence** (Average of all keypoint confidences from MoveNet)\n&emsp;&emsp;&emsp;The \
        minimum confidence required for the persons skeleton and pose to be displayed.')
    st.markdown('4) **Minimum Keypoint Detection Confidence**\n&emsp;&emsp;&emsp;The minimum confidence required for a keypoint to be displayed.')
    st.markdown('5) **Minimum Classification Confidence**\n&emsp;&emsp;&emsp;The minimum confidence required for a classified pose to be displayed.')
    st.markdown('## Additional Note')
    st.markdown('1) The **drawing functions** from **opencv-python** are used in this app, and work **best on higher quality** \
        images. We recommend that any image or video used **is at least 480 × 480 pixels in resolution.**')
    st.markdown('2) For best classifications, ensure that your full body is shown in the frame.')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child {
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child {
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif app_mode == 'Run on Image':

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child {
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child {
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown('---')
    draw_skeleton = st.sidebar.checkbox('Draw MoveNet Skeleton', value=True)
    st.sidebar.markdown('---')
    max_people = st.sidebar.number_input('Maximum Number of People', value=2, min_value=1, max_value=6)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider("Minimum Person Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    keypoint_confidence = st.sidebar.slider("Minimum Keypoint Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    classification_confidence = st.sidebar.slider("Minimum Classification Confidence", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg"])

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        width, height = img.size
        image = np.array(img)

    else:
        demo_image = DEMO_IMAGE
        img = Image.open(demo_image)
        width, height = img.size
        image = np.array(img)

    st.markdown('**Detected People**')
    kpi1_text = st.markdown('')

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    st.sidebar.markdown('---')

    # Dashboard
    out_image = image.copy()
    out_image, people_count = movenet_processing(out_image, max_people = max_people, mn_conf = detection_confidence,\
        kp_conf = keypoint_confidence, pred_conf = classification_confidence, draw_movenet_skeleton = draw_skeleton)

    kpi1_text.write(f"<h1 style='text-align: left;'>{people_count}</h1>", unsafe_allow_html=True)

    st.subheader('Output Image')
    st.image(out_image, use_column_width = True)

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child {
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child {
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown('---')
    use_webcam = st.sidebar.checkbox('Use Webcam')
    draw_skeleton = st.sidebar.checkbox('Draw MoveNet Skeleton', value=True)
    st.sidebar.markdown('---')
    max_people = st.sidebar.number_input('Maximum Number of People', value=2, min_value=1, max_value=6)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider("Minimum Person Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    keypoint_confidence = st.sidebar.slider("Minimum Keypoint Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    classification_confidence = st.sidebar.slider("Minimum Classification Confidence", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    video_file_buffer = st.sidebar.file_uploader('Upload Video', type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    # Obtain input video
    webcam_in_use = False
    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(0)
            webcam_in_use = True
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tffile.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    st.markdown('## Output')

    stframe = st.empty()
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    st.sidebar.markdown('---')

    fps = 0
    i = 0

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown('')

    with kpi2:
        st.markdown("**Detected People**")
        kpi2_text = st.markdown('')

    st.markdown('<hr/>', unsafe_allow_html=True)

    # Dashboard

    prevTime = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()

        if not ret:
            continue

        if webcam_in_use and (not use_webcam):
            break

        out_image = frame.copy()
        out_image, people_count = movenet_processing(out_image, max_people = max_people, mn_conf = detection_confidence,\
            kp_conf = keypoint_confidence, pred_conf = classification_confidence, draw_movenet_skeleton = draw_skeleton)

        # FPS Counter logic
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center;'>{people_count}</h1>", unsafe_allow_html=True)

        out_image = cv2.resize(out_image, (0, 0), fx=0.8, fy=0.8)
        out_image = image_resize(image=out_image, width=640)
        stframe.image(out_image, channels='BGR', use_column_width=True)
    cap.release()
        