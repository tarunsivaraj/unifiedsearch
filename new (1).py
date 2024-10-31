import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from PyPDF2 import PdfReader
import numpy as np
import os
import tempfile
import nltk
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone, ServerlessSpec, PineconeException
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import librosa  # For audio processing
from transformers import ClapProcessor, ClapModel
from pydub import AudioSegment
# Download nltk data for tokenization
nltk.download('punkt')

# API Key
API_KEY = "b9003922-accb-4691-95bd-3440da22397e"

# Initialize CLIP model for text, image, and video; CLAP for audio
if 'clip_model' not in st.session_state:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", force_download=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    st.session_state.clip_model = clip_model
    st.session_state.clip_processor = clip_processor
    st.session_state.clip_tokenizer = clip_tokenizer
else:
    clip_model = st.session_state.clip_model
    clip_processor = st.session_state.clip_processor
    clip_tokenizer = st.session_state.clip_tokenizer

if 'clap_model' not in st.session_state:
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    st.session_state.clap_model = clap_model
    st.session_state.clap_processor = clap_processor
else:
    clap_model = st.session_state.clap_model
    clap_processor = st.session_state.clap_processor

# Pinecone Setup
def create_index(api_key, index_name="multimedia-index", dimension=512):
    if 'pinecone_initialized' not in st.session_state:
        try:
            st.write("Initializing Pinecone...")
            pc = Pinecone(api_key=api_key, environment='us-east-1')
            st.session_state.pc = pc
            st.session_state.pinecone_initialized = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return

    if f"{index_name}_created" not in st.session_state:
        try:
            existing_indexes = st.session_state.pc.list_indexes()
            if index_name not in existing_indexes:
                st.session_state.pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1'))
            st.session_state.index = st.session_state.pc.Index(index_name)
            st.session_state[f"{index_name}_created"] = True
            st.write(f"Index '{index_name}' created or connected successfully.")
        except PineconeException as e:
            st.error(f"Error creating index: {str(e)}")

# Convert Embeddings to Lists
def convert_to_list(embedding):
    return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

# Text Embedding Function for Search Queries
def get_text_embedding(query_text):
    inputs = clip_processor(text=[query_text], return_tensors="pt")
    text_embedding = clip_model.get_text_features(**inputs)
    return convert_to_list(text_embedding[0].detach().numpy().astype(np.float32))

# Process Text for Embeddings (PDF)
def process_pdf_for_embeddings(pdf_file):
    reader = PdfReader(pdf_file)
    text_content = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text_content += page_text.replace("\n", " ") if page_text else ""
    
    sentences = sent_tokenize(text_content)
    text_embeddings = []

    for i, sentence in enumerate(sentences):
        if sentence.strip():
            inputs = clip_tokenizer(sentence, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                text_embedding = clip_model.get_text_features(input_ids=input_ids).detach().numpy().flatten()
            embedding_list = text_embedding.astype(np.float32).tolist()
            text_id = f"text_{i+1}"
            text_embeddings.append({
                "id": text_id,
                "values": embedding_list,
                "metadata": {"type": "text", "label": text_id, "content": sentence}
            })

    # Store in Pinecone
    index = st.session_state.index
    try:
        vectors = [(item["id"], item["values"], item["metadata"]) for item in text_embeddings]
        index.upsert(vectors=vectors)
        st.write("Text embeddings created and stored successfully.")
    except PineconeException as e:
        st.error(f"Failed to upsert text embeddings: {str(e)}")

# Process Image for Embeddings
def process_images(uploaded_images):
    image_embeddings = []
    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image)
        inputs = clip_processor(images=image, return_tensors="pt")
        image_embedding = clip_model.get_image_features(**inputs)
        embedding = convert_to_list(image_embedding[0].detach().numpy().astype(np.float32))
        image_embeddings.append({"id": uploaded_image.name, "values": embedding, "metadata": {"type": "image"}})

    # Store in Pinecone
    index = st.session_state.index
    try:
        vectors = [(item["id"], item["values"], item["metadata"]) for item in image_embeddings]
        index.upsert(vectors=vectors)
        st.write("Image embeddings created and stored successfully.")
    except PineconeException as e:
        st.error(f"Failed to upsert image embeddings: {str(e)}")

# Process Video for Embeddings
def process_video_for_embedding(video_path, interval_sec=10):
    frames = get_frames_from_video_by_interval(video_path, interval_sec)
    video_embeddings = []
    for i, frame in enumerate(frames):
        inputs = clip_processor(images=frame, return_tensors="pt")
        frame_embedding = clip_model.get_image_features(**inputs)
        embedding = convert_to_list(frame_embedding[0].detach().numpy().astype(np.float32))
        video_embeddings.append((f"video_frame_{i}", embedding, {"type": "video", "time_sec": i * interval_sec}))

    # Store in Pinecone
    index = st.session_state.index
    try:
        index.upsert(vectors=video_embeddings)
        st.write("Video frame embeddings created and stored successfully.")
    except PineconeException as e:
        st.error(f"Failed to upsert video embeddings: {str(e)}")

# Frame Extraction Helper for Video
def get_frames_from_video_by_interval(video_path, interval_sec=10):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_duration_sec = total_frames / fps

    for time_sec in np.arange(0, total_duration_sec, interval_sec):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        success, frame = video_capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    video_capture.release()
    return frames

# Process Audio for Embeddings using CLAP
def process_audio_for_embedding(audio_file, interval_sec=10):
    # Load the audio with the target sample rate of 48000
    target_sample_rate = 48000
    y, sr = librosa.load(audio_file, sr=target_sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)
    audio_embeddings = []

    # Segment and process the audio in intervals
    for start_time in np.arange(0, duration, interval_sec):
        end_time = min(start_time + interval_sec, duration)
        audio_segment = y[int(start_time * sr):int(end_time * sr)]
        
        # Ensure segment is non-empty and meets ClapProcessor input expectations
        if audio_segment.size > 0:
            # Convert audio segment to CLAP input format
            inputs = clap_processor(audios=audio_segment, sampling_rate=target_sample_rate, return_tensors="pt")
            audio_embedding = clap_model.get_audio_features(**inputs)

            # Convert embedding to list and add metadata
            embedding = audio_embedding[0].detach().numpy().astype(np.float32).tolist()
            audio_embeddings.append({
                "id": f"audio_segment_{int(start_time)}",
                "values": embedding,
                "metadata": {
                    "type": "audio",
                    "time_sec": float(start_time),  # Explicitly convert to Python float
                    "filename": os.path.basename(audio_file)  # Extracts the filename
                }
            })

    # Store audio embeddings in Pinecone
    index = st.session_state.index
    try:
        vectors = [(item["id"], item["values"], item["metadata"]) for item in audio_embeddings]
        index.upsert(vectors=vectors)
        st.write("Audio embeddings created and stored successfully.")
    except PineconeException as e:
        st.error(f"Failed to upsert audio embeddings: {str(e)}")

def play_audio_segment(audio_path, time_sec, segment_duration=30):
    # Calculate start and end times for a 30-second segment centered around `time_sec`
    start_time_sec = max(time_sec - segment_duration // 2, 0)
    end_time_sec = min(start_time_sec + segment_duration, librosa.get_duration(filename=audio_path))

    # Load the full audio file and extract the 30-second segment
    full_audio = AudioSegment.from_file(audio_path)
    segment_audio = full_audio[start_time_sec * 1000:end_time_sec * 1000]  # pydub works in milliseconds

    # Save the segment to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        segment_audio.export(temp_audio_file.name, format="wav")
        temp_audio_path = temp_audio_file.name

    return temp_audio_path if os.path.exists(temp_audio_path) else None

def play_video_segment(video_path, time_sec, segment_duration=5):
    start_time_sec = max(time_sec - segment_duration // 2, 0)
    end_time_sec = start_time_sec + segment_duration

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_path = temp_video_file.name

    ffmpeg_extract_subclip(video_path, start_time_sec, end_time_sec, targetname=temp_video_path)

    if os.path.exists(temp_video_path):
        return temp_video_path
    else:
        st.error("Failed to create video segment.")
        return None

# Search across all data types
def search_query(query_text, top_k=2):
    query_embedding = get_text_embedding(query_text)
    index = st.session_state.index

    # Fetch top results for each type
    text_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"type": "text"})["matches"]
    image_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"type": "image"})["matches"]
    video_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"type": "video"})["matches"]
    audio_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"type": "audio"})["matches"]

    # Store results in session state
    st.session_state.search_results = {
        "text": text_results if text_results else None,
        "image": image_results if image_results else None,
        "video": video_results if video_results else None,
        "audio": audio_results if audio_results else None
    }

    # Display search results
    st.subheader("Search Results")
    
    # Display Text Results
    st.write("### Text Results")
    if st.session_state.search_results["text"]:
        for result in st.session_state.search_results["text"]:
            st.markdown(f"**Match: {result['metadata']['content']}** (Score: {result['score']:.4f})")
    else:
        st.write("No text results found for this query.")

    # Display Image Results
    st.write("### Image Results")
    if st.session_state.search_results["image"]:
        for result in st.session_state.search_results["image"]:
            image_id = result['id']  # Get the image ID, which we assume is the image path
            score = result['score']
        
            try:
            # Open and display the image by its path
                with open(image_id, 'rb') as img_file:
                    img = Image.open(img_file)
                    st.image(img, caption=f"Score: {score:.4f}", use_column_width=True)
            except FileNotFoundError:
                st.write(f"Image {image_id} not found.")
    else:
        st.write("No image results found for this query.")

    # Display Video Results
    st.write("### Video Results")
    if st.session_state.search_results["video"]:
        for result in st.session_state.search_results["video"]:
            time_sec = result["metadata"]["time_sec"]
            st.markdown(f"**Video Frame at {time_sec} seconds** (Score: {result['score']:.4f})")
            segment_video_path = play_video_segment(st.session_state.temp_video_path, time_sec)
            if segment_video_path:
                st.video(segment_video_path)
    else:
        st.write("No video results found for this query.")

    # Display Audio Results
    st.write("### Audio Results")
    if st.session_state.search_results["audio"]:
        for result in st.session_state.search_results["audio"]:
            time_sec = result["metadata"]["time_sec"]
            st.markdown(f"**Audio Segment at {time_sec} seconds** (Score: {result['score']:.4f})")
            segment_audio_path = play_audio_segment(st.session_state.temp_audio_path, time_sec)
            if segment_audio_path:
                st.audio(segment_audio_path)
    else:
        st.write("No audio results found for this query.")
# Main function for app flow
def main():
    st.title("Unified Multimedia Search")
    create_index(API_KEY)
    
    # Text Upload
    uploaded_text = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_text:
        st.success("Text file uploaded successfully.")
        if st.button("Create and Store Text Embeddings"):
            process_pdf_for_embeddings(uploaded_text)

    # Image Upload
    uploaded_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        st.success("Image files uploaded successfully.")
        if st.button("Create and Store Image Embeddings"):
            process_images(uploaded_images)

    # Video Upload
    uploaded_video = st.file_uploader("Upload Video Files", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            st.session_state.temp_video_path = temp_video_file.name

        st.video(st.session_state.temp_video_path)
        st.success("Video file uploaded successfully.")
        if st.button("Create and Store Video Embeddings"):
            process_video_for_embedding(st.session_state.temp_video_path)

    # Audio Upload and Processing
    uploaded_audio = st.file_uploader("Upload Audio Files", type=["mp3", "wav"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file.write(uploaded_audio.read())
            st.session_state.temp_audio_path = temp_audio_file.name
        st.success("Audio file uploaded successfully.")
        if st.button("Create and Store Audio Embeddings"):
            process_audio_for_embedding(st.session_state.temp_audio_path)
    
    # Search Query
    query_text = st.text_input("Enter search query:")
    if query_text and st.button("Search"):
        search_query(query_text, top_k=2)

if __name__ == "__main__":
    main()
