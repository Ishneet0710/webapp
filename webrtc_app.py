import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from MoveNet_Processing_Utils import movenet_processing
import av
import threading
from PIL import Image
import numpy as np

DEMO_IMAGE = "./demos/demo.jpg"

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

lock = threading.Lock()
no_ppl = {"ppl_count": None}

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    out_image = img.copy()

    out_image, people_count = movenet_processing(out_image, max_people = max_people, mn_conf = detection_confidence,\
            kp_conf = keypoint_confidence, pred_conf = classification_confidence, draw_movenet_skeleton = draw_skeleton)

    with lock:
        no_ppl["ppl_count"] = people_count

    return av.VideoFrame.from_ndarray(out_image, format="bgr24")

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


app_mode = st.sidebar.selectbox('Select App Mode', ['Home', 'Run on Image', 'Run in Real-Time'])


if app_mode == 'Home':
    st.markdown('## About App')
    st.markdown('This web-application utilises **MoveNet** to perform **pose estimation** on an input image or video. \
        The coordinates obtained by MoveNet are then passed to a **custom classifying algorithm** to perform **pose classification**. \
        **Streamlit** was used to create the **web-GUI**.')
    st.markdown('The **pose classes** involved are:\n1) Standing\n2) Sitting\n3) Lying')
    st.markdown('## Instructions')
    st.markdown('You may run the models on **either images or real-time webcam feed** (see sidebar). For running on image, you may either use the \
        provided demo image, or **upload** your own image **through the sidebar**')
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
        image = image[:,:,::-1] # Convert to BGR

    else:
        img = Image.open(DEMO_IMAGE)
        width, height = img.size
        image = np.array(img)
        image = image[:,:,::-1] # Convert to BGR

    st.markdown('**Detected People**')
    kpi1_text = st.markdown('')

    st.sidebar.text('Original Image')
    st.sidebar.image(image, channels="BGR")
    st.sidebar.markdown('---')

    # Dashboard
    out_image = image.copy()
    out_image, people_count = movenet_processing(out_image, max_people = max_people, mn_conf = detection_confidence,\
        kp_conf = keypoint_confidence, pred_conf = classification_confidence, draw_movenet_skeleton = draw_skeleton)

    kpi1_text.write(f"<h1 style='text-align: left;'>{people_count}</h1>", unsafe_allow_html=True)

    st.subheader('Output Image')
    st.image(out_image, channels="BGR", use_column_width = True)

elif app_mode == 'Run in Real-Time':
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

    st.markdown("## Real-Time Output")

    kpi, _ = st.columns(2)

    ctx = webrtc_streamer(
        key="real-time",
        video_frame_callback=callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        # # For Deploying
        # rtc_configuration={
        #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        # }
    )

    with kpi:
        st.markdown("**Detected People**")
        kpi_text = st.markdown('N/A')

    while ctx.state.playing:
        with lock:
            ppl_count = no_ppl["ppl_count"]
            
        if ppl_count is None:
            continue
            
        kpi_text.write(f"<h1 style='text-align: center;'>{int(ppl_count)}</h1>", unsafe_allow_html=True)