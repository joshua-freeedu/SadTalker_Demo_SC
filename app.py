import streamlit as st
import os

from scipy.io import wavfile
from PIL import Image

import requests
import base64
from io import BytesIO

from gtts import gTTS

server_url = os.environ["ngrok_url"]

# App information and setup
project_title = "SadTalker: Image-to-Animation App"
project_link = "https://github.com/OpenTalker/SadTalker"
project_icon = "icon.png"
st.set_page_config(page_title=project_title, initial_sidebar_state='collapsed', page_icon=project_icon)
#######################################################################################################################
def main():
    head_col = st.columns([1,8])
    with head_col[0]:
        st.image(project_icon)
    with head_col[1]:
        st.title(project_title)
    st.write(f"Source Project: {project_link}")

    try:
        response = requests.get(f'{server_url}/ping')
        if response.text == "pong":
            st.success("• Connected to the server.")
        else:
            st.warning(f"• Server is running but returned unexpected response.")
    except requests.exceptions.ConnectionError:
        st.error("• Could not connect to the server.")

    st.markdown("***")
    st.subheader("")

#########################################

    if "driven_audio" not in st.session_state:
        st.session_state.driven_audio = None

    if "source_image" not in st.session_state:
        st.session_state.source_image = None
    image_files = [os.path.join('examples/images/', f) for f in os.listdir('examples/images/') if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'))]
    audio_files = [os.path.join('examples/audio/', f) for f in os.listdir('examples/audio/') if f.lower().endswith(('mp3', 'wav'))]

    # Examples / Upload section
    if st.checkbox('Use example image?'):
        example_im = st.selectbox('Example images', image_files)
        example_image = Image.open(example_im)
        # Convert to BytesIO
        byte_io = BytesIO()
        example_image.save(byte_io, format='PNG')
        byte_image = byte_io.getvalue()
        
        st.session_state.source_image = BytesIO(byte_image)
    else:
        uploaded_image = st.file_uploader("Image", ['png', 'jpg', 'jpeg'])
        st.session_state.source_image = uploaded_image
    if st.session_state.source_image:
        st.image(st.session_state.source_image, width=250)
    
    audio_choice = st.radio("Choose audio input method:",
                            ("Use example audio", "Upload audio", "Text to speech")
                            )
    if audio_choice == "Use example audio":
        example_au = st.selectbox('Example audios', audio_files)
        sample_rate, audio_data = wavfile.read(example_au)
        
        # Convert to bytes
        byte_io = BytesIO()
        wavfile.write(byte_io, sample_rate, audio_data)
        byte_audio = byte_io.getvalue()
        
        st.session_state.driven_audio = BytesIO(byte_audio)

    elif audio_choice == "Upload audio":
        st.session_state.driven_audio = st.file_uploader("Audio", ['mp3', 'wav'])

    elif audio_choice == "Text to speech":
        # Language selection
        languages = [
            ("English", "en"),
            ("Arabic", "ar"),
            ("Chinese", "zh-CN"),
            ("Dutch", "nl"),
            ("Finnish", "fi"),
            ("Filipino", "tl"),
            ("French", "fr"),
            ("German", "de"),
            ("Greek", "el"),
            ("Hindi", "hi"),
            ("Hungarian", "hu"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
            ("Nepali", "ne"),
            ("Polish", "pl"),
            ("Portuguese", "pt"),
            ("Romanian", "ro"),
            ("Russian", "ru"),
            ("Spanish", "es"),
            ("Swedish", "sv"),
            ("Thai", "th"),
            ("Turkish", "tr"),
            ("Vietnamese", "vi")
        ]
        target_language = st.selectbox("Select the language of your text", [lang[0] for lang in languages])
        user_input_text = st.text_area("Type the text you want to convert to speech:")

        target_lang_code = next((lang[1] for lang in languages if lang[0] == target_language), None)

        if st.button("Generate Speech"):
            tts = gTTS(text=user_input_text, lang=target_lang_code, slow=False)
            byte_io = BytesIO()
            tts.write_to_fp(byte_io)
            st.session_state.driven_audio = BytesIO(byte_io.getvalue())

    if st.session_state.driven_audio:
        st.audio(st.session_state.driven_audio)

    # Sliders and options for settings
    options_expander = st.expander("Additional Options")
    pose_style = options_expander.slider("Pose style", 0, 46, 0)
    size_of_image = options_expander.radio("Face model resolution", [256, 512])
    preprocess_type = options_expander.radio("Preprocess", ['crop', 'resize', 'full', 'extcrop', 'extfull'])
    is_still_mode = options_expander.checkbox("Still Mode (fewer hand motion, works with preprocess 'full')")
    batch_size = options_expander.slider("Batch size in generation", 1, 10, 2)
    enhancer = options_expander.checkbox("GFPGAN as Face enhancer")

    # Send Image and Audio to local server for processing
    if st.button("Generate"):
        if (st.session_state.source_image) is not None and (st.session_state.driven_audio is not None):
            files = {
                'source_image': ("source_image.png",st.session_state.source_image.getvalue()),
                'driven_audio': ("driven_audio.wav",st.session_state.driven_audio.getvalue())
            }
            with st.spinner('Generating animation...'):
                try:
                    response = requests.post(f'{server_url}/talk-sad', files=files, 
                                                data={'pose_style': pose_style, 'size_of_image': size_of_image,
                                                        'preprocess_type': preprocess_type,'is_still_mode': is_still_mode,
                                                        'batch_size': batch_size,'enhancer': enhancer},
                                                timeout=600)
                except requests.exceptions.SSLError as e:
                    st.error(f"SSL Error: {e}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

            # Handle received response from local server
            if response.status_code == 200:
                json_data = response.json()
                if "video_data" in json_data:
                    video_data = base64.b64decode(json_data['video_data'])

                    # Create a video buffer and display it using st.video
                    video_buffer = BytesIO(video_data)
                    st.video(video_buffer)
                elif "error" in json_data:
                    st.error(f"Server error: {json_data['error']}")
            else:
                st.error('Request failed')
        else:
            st.warning("Please upload an image and an audio file!")

if __name__ == "__main__":
    main()