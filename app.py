import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from PIL import Image
from predict import detect_animals

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="ForestGuard | Wildlife Monitoring",
    page_icon="🌳",
    layout="wide"
)

HISTORY_FILE = "detection_history.csv"

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background: #101820;
    color: #F4F6F8;
}
[data-testid="stSidebar"] {
    background: #182631;
}
h1, h2, h3 {
    color: #E8F1F2 !important;
}
div[data-testid="stMetric"] {
    background: #1D303C;
    border: 1px solid #34505C;
    padding: 15px;
    border-radius: 12px;
}
.stButton button {
    background: #168C83;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HISTORY STORAGE ----------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_detections(records):
    if not records:
        return

    new_df = pd.DataFrame(records)

    if os.path.exists(HISTORY_FILE):
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            pass

    new_df.to_csv(HISTORY_FILE, index=False)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🌳 ForestGuard")
st.sidebar.caption("AI Wildlife Monitoring System")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Image Detection",
        "Video Analysis",
        "Detection History",
        "Analytics",
        "Model Performance",
        "Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Pretrained YOLO animal detection")

# ---------------- LOAD HISTORY ----------------
history_df = load_history()

# ---------------- OVERVIEW ----------------
if page == "Overview":

    st.title("🌳 ForestGuard Wildlife Dashboard")
    st.write("AI-powered wildlife image and video monitoring")
    st.markdown("---")

    total = len(history_df)
    species = (
        history_df["Animal"].nunique()
        if not history_df.empty and "Animal" in history_df.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Detections", total)
    c2.metric("Animal Species", species)
    c3.metric("Analysis Records", total)
    c4.metric("System Status", "Ready")

    st.subheader("📊 Wildlife Overview")

    if not history_df.empty and "Animal" in history_df.columns:

        counts = (
            history_df["Animal"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["Animal", "Count"]

        left, right = st.columns(2)

        with left:
            fig_pie = px.pie(
                counts,
                names="Animal",
                values="Count",
                hole=0.45,
                title="Animal Distribution",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with right:
            fig_bar = px.bar(
                counts,
                x="Animal",
                y="Count",
                title="Detection Count by Animal",
                text="Count"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Recent Detection Records")
        st.dataframe(history_df.tail(20), use_container_width=True)

    else:
        st.info("No detections yet. Upload an image or video to begin.")

    st.markdown("---")
    st.subheader("System Modules")
    a, b, c = st.columns(3)
    a.markdown("### 📷 Image Detection")
    a.write("Upload a photo and detect animals.")
    b.markdown("### 🎥 Video Analysis")
    b.write("Sample video frames and detect animals.")
    c.markdown("### 📈 Analytics")
    c.write("View charts and detection history.")

# ---------------- IMAGE DETECTION ----------------
elif page == "Image Detection":

    st.title("📷 Image Animal Detection")
    st.write("Upload a photograph and run AI detection.")

    photo = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="photo"
    )

    if photo is not None:

        image = Image.open(photo).convert("RGB")
        image_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyse Image", type="primary"):

            with st.spinner("AI is analysing the image..."):
                annotated_image, detections = detect_animals(
                    image_array,
                    source="Image"
                )

            st.subheader("AI Detection Result")
            st.image(
                annotated_image,
                caption="Detected animals with bounding boxes",
                use_container_width=True
            )

            if detections:
                st.success(f"{len(detections)} animal(s) detected!")

                result_df = pd.DataFrame(detections)
                st.dataframe(result_df, use_container_width=True)

                save_detections(detections)
                st.success("Saved to Detection History.")

            else:
                st.warning("No supported animals were detected in this photo.")

# ---------------- VIDEO ANALYSIS ----------------
elif page == "Video Analysis":

    st.title("🎥 Video Animal Detection")
    st.write("Upload a video. ForestGuard analyses sampled frames.")

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"],
        key="video"
    )

    # Sample one frame every N seconds
    sample_seconds = st.slider(
        "Frame sampling interval (seconds)",
        min_value=1,
        max_value=10,
        value=2
    )

    if video_file is not None:

        st.video(video_file)

        if st.button("▶️ Analyse Video", type="primary"):

            temp_path = None
            all_detections = []
            preview_frames = []

            try:
                # Save uploaded video temporarily for OpenCV
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp:
                    temp.write(video_file.getbuffer())
                    temp_path = temp.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error("Could not open video. Try another file.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if fps <= 0:
                        fps = 25

                    duration = total_frames / fps
                    frame_step = max(1, int(fps * sample_seconds))

                    st.write(f"Video duration: {duration:.1f} seconds")
                    st.write(f"Analysing approximately every {sample_seconds} second(s)")

                    progress = st.progress(0)
                    status = st.empty()

                    frame_number = 0
                    analysed_count = 0

                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            break

                        if frame_number % frame_step == 0:

                            video_time = frame_number / fps

                            rgb_frame = cv2.cvtColor(
                                frame, cv2.COLOR_BGR2RGB
                            )

                            status.write(
                                f"Analysing frame {frame_number} "
                                f"({video_time:.1f} sec)"
                            )

                            annotated, detections = detect_animals(
                                rgb_frame,
                                source="Video",
                                frame_number=frame_number,
                                video_time=video_time
                            )

                            all_detections.extend(detections)
                            analysed_count += 1

                            # Keep only a few frames to show as results
                            if detections and len(preview_frames) < 6:
                                preview_frames.append(annotated)

                        frame_number += 1

                        if total_frames > 0:
                            progress.progress(
                                min(frame_number / total_frames, 1.0)
                            )

                    cap.release()
                    progress.progress(1.0)
                    status.success("Video analysis completed!")

                    if all_detections:

                        st.success(
                            f"Found {len(all_detections)} animal detection(s) "
                            f"in {analysed_count} analysed frames."
                        )

                        save_detections(all_detections)

                        st.subheader("Detection Results")
                        result_df = pd.DataFrame(all_detections)
                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        if preview_frames:
                            st.subheader("Detected Frame Previews")
                            for i, frame_image in enumerate(preview_frames):
                                st.image(
                                    frame_image,
                                    caption=f"Detected frame {i + 1}",
                                    use_container_width=True
                                )

                        st.success("Video detections saved to Detection History.")

                    else:
                        st.warning(
                            "No supported animals detected in the sampled frames."
                        )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

# ---------------- DETECTION HISTORY ----------------
elif page == "Detection History":

    st.title("📋 Detection History")

    history_df = load_history()

    if not history_df.empty:

        st.metric("Total Records", len(history_df))

        st.dataframe(
            history_df.sort_index(ascending=False),
            use_container_width=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History CSV",
            data=csv_data,
            file_name="forestguard_detection_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Detection History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("Detection history cleared.")
            st.rerun()

    else:
        st.info("No detection history yet. Analyse an image or video first.")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Wildlife Analytics")

    history_df = load_history()

    if not history_df.empty:

        counts = (
            history_df["Animal"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["Animal", "Count"]

        col1, col2 = st.columns(2)

        with col1:
            pie = px.pie(
                counts,
                names="Animal",
                values="Count",
                hole=0.4,
                title="Animal Distribution"
            )
            st.plotly_chart(pie, use_container_width=True)

        with col2:
            bar = px.bar(
                counts,
                x="Animal",
                y="Count",
                color="Animal",
                title="Animal Detection Frequency"
            )
            st.plotly_chart(bar, use_container_width=True)

        st.subheader("Source Breakdown")
        if "Source" in history_df.columns:
            source_counts = history_df["Source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            st.bar_chart(source_counts.set_index("Source"))

    else:
        st.info("Analytics will appear when real detections are recorded.")

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.title("🧠 Model Information")

    st.write("Model: YOLOv8 Nano pretrained on COCO")
    st.write("Detection confidence threshold: 25%")

    st.warning(
        "This is a pretrained model, not yet trained on your custom "
        "forest dataset. It only recognizes its supported COCO classes."
    )

# ---------------- REPORTS ----------------
elif page == "Reports":

    st.title("📄 Wildlife Reports")

    history_df = load_history()

    if not history_df.empty:

        st.subheader("Animal Detection Summary")

        counts = history_df["Animal"].value_counts().reset_index()
        counts.columns = ["Animal", "Count"]

        st.dataframe(counts, use_container_width=True)

        st.download_button(
            "⬇️ Download Report CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="forestguard_report.csv",
            mime="text/csv"
        )

    else:
        st.info("Reports will be available after detections are recorded.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("ForestGuard | AI Wildlife Monitoring System")