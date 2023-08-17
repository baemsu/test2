import streamlit as st
import numpy as np
import cv2
import insightface

# InsightFace 모델 로드
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)

st.title("Face Age and Gender Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = model.get(image_rgb)
    for face in faces:
        bbox = face.bbox.astype(np.int).flatten()
        cv2.rectangle(image_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image_rgb, f"Age: {face.age}, Gender: {'Male' if face.gender == 0 else 'Female'}", 
                    (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    st.image(image_rgb, channels="RGB")
