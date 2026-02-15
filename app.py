import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX
providers = ['CPUExecutionProvider']
session = ort.InferenceSession('tomato_blight.onnx', providers=providers)

severity_map = {
    0: ('Healthy', 0, '🟢 No infection'),
    1: ('Mild', 12, '🟡 Low spread'),
    2: ('Moderate', 37, '🟠 Medium spread'),
    3: ('Severe', 75, '🔴 High spread')
}

st.title("🍅 Tomato Late Blight Detector")
uploaded = st.file_uploader("Upload leaf...", type=['jpg','png'])

if uploaded:
    img = Image.open(uploaded).resize((224, 224)).convert('RGB')
    st.image(img, caption="Uploaded", use_column_width=True)
    
    # Preprocess
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    # Predict
    ort_inputs = {session.get_inputs()[0].name: img_array}
    pred = session.run(None, ort_inputs)[0][0]
    
    class_id = np.argmax(pred)
    confidence = pred[class_id] * 100
    
    label, percent, desc = severity_map[class_id]
    
    st.subheader(f"**{label}** ({confidence:.1f}%)")
    col1, col2 = st.columns([1,2])
    col1.metric("Spread", f"{percent}%")
    col2.progress(percent/100)
    st.success(desc)
