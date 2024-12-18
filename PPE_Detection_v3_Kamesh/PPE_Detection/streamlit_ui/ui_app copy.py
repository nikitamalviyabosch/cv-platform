import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

from Main import *

st.set_page_config(
    page_title="Drowsiness Detection",
    layout="wide",  
)

st.markdown('<p style="font-size: 2em; text-align: center;">Driver Monitoring Dashboard</p>', unsafe_allow_html=True)

st.markdown('''
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 280px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 280px; 
    margin-left: -280px;
}
            
            
</style>
''', unsafe_allow_html=True)

st.subheader('Configuration')
image_paths = []

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('---')  
    detection_conf = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.markdown('---')
    eyeDrow_conf = st.slider('Eye Drowsiness threshold',min_value=0.0,max_value=1.0, value=0.15, step=0.15)
    st.markdown('---')
with col2:
    st.markdown('---') 
    distra_conf = st.slider('Distraction threshold',min_value=50,max_value=100, value=80, step=10)
    st.markdown('---')
    eyeGaze_conf = st.slider('Eye Gaze threshold',min_value=1,max_value=10, value=2, step=1)
    st.markdown('---')

with col3:
    st.markdown('---') 
    drowsfall_conf = st.slider('Drowsiness fall threshold',min_value=-100,max_value=-200, value=-100, step=-1)
    st.markdown('---')
    Yawn_conf = st.slider('Yawn threshold',min_value=1,max_value=10, value=3, step=1)
    st.markdown('---') 


@st.cache_resource()
def image_resize(image,width = None,height=None,inter=cv2.INTER_AREA):
    dim = None
    (h,w,_) = image.shape
    if width is None and height is None:
        return image
    if not width is None:
        r = width/float(w)
        dim = (width,int(h*r))
    resized = cv2.resize(image,dim,interpolation=inter)
    return resized

st.subheader('Output')

st.sidebar.title('Analytics')

st.sidebar.markdown('---') 

col1, col2 = st.sidebar.columns([3, 3])  # Create two equal-width columns

with col1: 
    st.markdown('<p style="font-size: 12px;">Frame Rate</p>', unsafe_allow_html=True)
    kpi1_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True) 

    st.markdown('<p style="font-size: 12px;">Drowsiness Fall</p>', unsafe_allow_html=True)
    kpi2_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<p style="font-size: 12px;">Eye Drowsiness</p>', unsafe_allow_html=True)
    kpi3_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<p style="font-size: 12px;">Mouth Drowsiness</p>', unsafe_allow_html=True)
    kpi4_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<p style="font-size: 12px;">Gaze</p>', unsafe_allow_html=True)
    kpi5_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<p style="font-size: 12px;">Mobile Usage</p>', unsafe_allow_html=True)
    kpi6_Text = st.markdown('<p style="font-size: 8px;">0</p>', unsafe_allow_html=True)

st.sidebar.markdown('<hr>', unsafe_allow_html=True)

fall = None
drow = None
gazeHead = None
yawn = None
mobile = None

stframe = st.empty()
sframe = st.empty()
tffile = tempfile.NamedTemporaryFile(delete=False)
prev = time.time()
fps = 0
i = 0

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=detection_conf, min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ct = CentroidTracker(maxDisappeared=100, maxDistance=50)
# trackableObjects = {}

width = int(cv2.CAP_PROP_FRAME_WIDTH)
height = int(cv2.CAP_PROP_FRAME_HEIGHT)
fps = int(cv2.CAP_PROP_FPS)
alert_param = False

while cap.isOpened():
    try:
        i += 1
        ret,frame = cap.read()
        if not ret:
            continue
        # Convert the BGR image to RGB before processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        handresults = mp_hands.process(image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            top_left = (int(face_landmarks.landmark[127].x * frame.shape[1]),int(face_landmarks.landmark[10].y * frame.shape[0]))
            bottom_right = (int(face_landmarks.landmark[356].x * frame.shape[1]), int(face_landmarks.landmark[152].y * frame.shape[0]))
            id = 'kamesh'
            cv2.rectangle(frame,(top_left[0],top_left[1]),(bottom_right[0],bottom_right[1]),(255,0,0),2,1)
            cv2.putText(frame,f'person_id : {id}',(top_left[0],top_left[1] - 10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)
            # try:
            #     objects = ct.update([(top_left[0],top_left[1],bottom_right[0],bottom_right[1])])
            #     for (objectId,centroid) in object.items():
            #         if objectId in trackableObjects:
            #             pass
            #         else:
            #             pass
            # except Exception as e:
            #     print('error occured while : ',str(e))
            curr_dir = os.getcwd()
            curr_time = str(datetime.datetime.utcnow().strftime('%H-%M-%S'))

            frame,headfall = HeadMovement(frame, face_landmarks,drowsfall_conf)
            if headfall is not None:
                fall = datetime.datetime.now().strftime('%H:%M:%S')
                alertdir = os.path.join(curr_dir, 'alerts', 'fall')
                if not os.path.isdir(alertdir):
                    os.makedirs(alertdir)
                filename = os.path.join(alertdir,f"fallalert_{curr_time}.jpg")
                cv2.imwrite(filename,frame)              

            frame,eyeDrow,yawning = drowsiness(frame, face_landmarks,eyeDrow_conf,Yawn_conf)
            if eyeDrow is not None:
                drow = datetime.datetime.now().strftime('%H:%M:%S')
                alertdir = os.path.join(curr_dir, 'alerts', 'eyedrow')
                if not os.path.isdir(alertdir):
                    os.makedirs(alertdir)
                filename = os.path.join(alertdir,f"drowsiness_{curr_time}.jpg")
                cv2.imwrite(filename,frame) 
            if yawning is not None:
                yawn = datetime.datetime.now().strftime('%H:%M:%S')
                alertdir = os.path.join(curr_dir, 'alerts', 'yawn')
                if not os.path.isdir(alertdir):
                    os.makedirs(alertdir)
                filename = os.path.join(alertdir,f"yawn_{curr_time}.jpg")
                cv2.imwrite(filename,frame) 

            frame,gazHead = GazeHeadMovement(frame, face_landmarks,eyeGaze_conf)
            if gazHead is not None:
                gazeHead = datetime.datetime.now().strftime('%H:%M:%S')
                alertdir = os.path.join(curr_dir, 'alerts', 'gaze')
                if not os.path.isdir(alertdir):
                    os.makedirs(alertdir)
                filename = os.path.join(alertdir,f"gaze_{curr_time}.jpg")
                cv2.imwrite(filename,frame) 
            
            frame,Mobileuse = mobileUsage(frame,face_landmarks,handresults,distra_conf)
            if Mobileuse is not None:
                mobile = datetime.datetime.now().strftime('%H:%M:%S')
                alertdir = os.path.join(curr_dir, 'alerts', 'distraction')
                if not os.path.isdir(alertdir):
                    os.makedirs(alertdir)
                filename = os.path.join(alertdir,f"distraction_{curr_time}.jpg")
                cv2.imwrite(filename,frame) 

        curr = time.time()
        fps = 1/(curr-prev)
        prev = curr

        kpi1_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{int(fps)}</h1>",unsafe_allow_html=True)
        kpi2_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{fall}</h1>",unsafe_allow_html=True)
        kpi3_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{drow}</h1>",unsafe_allow_html=True)
        kpi4_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{yawn}</h1>",unsafe_allow_html=True)
        kpi5_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{gazeHead}</h1>",unsafe_allow_html=True)
        kpi6_Text.markdown(f"<h1 style = 'text-align;centre;color:red;'>{mobile}</h1>",unsafe_allow_html=True)
        
    except Exception as e:
        print('Exception Occured while ',str(e))    
    frame = cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
    frame = image_resize(frame,width=300)
    stframe.image(frame,channels='BGR',width=700)
    if alert_param is False:
        st.subheader('Alerts')
        alert_param = True

    for folder_path in os.listdir(os.path.join(os.getcwd(),'alerts')):
        for images in os.listdir(os.path.join(os.getcwd(),'alerts',folder_path)):
            if not images in image_paths:
                im = cv2.imread(os.path.join(os.getcwd(),'alerts',folder_path, images))
                im = cv2.resize(frame,(0,0),fx=0.7,fy=0.7)
                st.image(image_resize(im,width=300),caption = images.split('_')[0],channels='BGR', width=200)
                image_paths.append(images)

cap.release()