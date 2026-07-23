import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="🕵️",
    layout="wide"
)

MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fake", "Real"]  # index 0 = fake, index 1 = real (alphabetical, matches classification_report.txt)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def crop_to_face(pil_img: Image.Image, margin: float = 0.35):
    """Detect the largest face and crop to it with a margin so the
    framing matches typical face-forensics training data (tight crop,
    no background). Falls back to the full image if no face is found."""
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return pil_img, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(rgb.shape[1], x + w + mx)
    y1 = min(rgb.shape[0], y + h + my)

    cropped = rgb[y0:y1, x0:x1]
    return Image.fromarray(cropped), True


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def confidence_bar_chart(fake_pct: float, real_pct: float):
    """Dark navy bar chart matching the reference demo style: red FAKE bar, green REAL bar."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    bars = ax.bar(["FAKE", "REAL"], [fake_pct, real_pct],
                   color=["#e74c3c", "#2ecc71"], width=0.55,
                   edgecolor="white", linewidth=1.2)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability %", color="white", fontsize=11)
    ax.set_title("Confidence Breakdown", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#3a4a5c")

    for bar, pct in zip(bars, [fake_pct, real_pct]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{pct:.1f}%", ha="center", color="white", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------- Sidebar ----------
st.sidebar.title("🕵️ Face Detector")
st.sidebar.markdown(
    """
This app uses a **MobileNetV2**-based transfer-learning model
trained to distinguish **real** human faces from **AI-generated
(synthetic) faces**.

**Model stats**
- Accuracy: **81.42%**
- AUC-ROC: **0.891**
"""
)
page = st.sidebar.radio("Go to", ["🔍 Try the Model", "📊 Model Performance"])

# ---------- Page 1: Live prediction ----------
if page == "🔍 Try the Model":
    st.markdown("<h1 style='text-align:center;'>AI Face Detection System — Live Demo</h1>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            raw_image = Image.open(uploaded).convert("RGB")
            face_image, found = crop_to_face(raw_image)

            with st.spinner("Analyzing..."):
                model = load_model()
                x = preprocess(face_image)
                preds = model.predict(x, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                label = CLASS_NAMES[pred_idx]
                confidence = float(preds[pred_idx]) * 100
                fake_pct = float(preds[0]) * 100
                real_pct = float(preds[1]) * 100

            col1, col2 = st.columns([1, 1])

            with col1:
                color = "#2ecc71" if label == "Real" else "#e74c3c"
                st.markdown(
                    f"<h3 style='color:{color};'>Prediction: {label.upper()}</h3>"
                    f"<h4 style='color:{color};'>Confidence: {confidence:.1f}%</h4>",
                    unsafe_allow_html=True
                )
                st.image(np.array(face_image), caption="Detected face" if found else "Uploaded image (no face auto-detected)", width=380)

            with col2:
                fig = confidence_bar_chart(fake_pct, real_pct)
                st.pyplot(fig)

            if label == "Real":
                st.markdown(
                    "<div style='text-align:center; padding:10px; border:2px solid #2ecc71; "
                    "border-radius:8px; color:#2ecc71; font-weight:bold; font-size:18px;'>"
                    "✓ AUTHENTIC FACE</div>", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='text-align:center; padding:10px; border:2px solid #e74c3c; "
                    "border-radius:8px; color:#e74c3c; font-weight:bold; font-size:18px;'>"
                    "⚠ AI-GENERATED FACE DETECTED</div>", unsafe_allow_html=True
                )

            if not found:
                st.caption("⚠️ No face was auto-detected in this image — the whole image was used instead. For best results, upload a clear front-facing photo.")

        except Exception as e:
            st.error("Something went wrong processing this image:")
            st.exception(e)
    else:
        st.info("👆 Upload a JPG or PNG face image to get started.")

# ---------- Page 2: Results dashboard ----------
else:
    st.title("📊 Model Performance")
    st.caption("Evaluation on the held-out test set")

    with open("results/classification_report.txt") as f:
        report = f.read()

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "81.42%")
    m2.metric("AUC-ROC", "0.8914")

    st.text(report)

    st.subheader("Confusion Matrix")
    st.image("results/confusion_matrix.png", use_container_width=False)

    st.subheader("ROC Curve")
    st.image("results/roc_curve.png", use_container_width=False)

    st.subheader("Training History")
    st.image("results/training_history.png", use_container_width=True)

    st.subheader("Grad-CAM — what the model is looking at")
    st.image("results/gradcam_samples.png", use_container_width=True)

    st.subheader("Sample Predictions")
    st.image("results/sample_predictions.png", use_container_width=True)
