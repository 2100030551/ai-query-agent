import os
from PIL import Image
import speech_recognition as sr
import streamlit as st
from googletrans import LANGUAGES
from streamlit_option_menu import option_menu
from transformers import pipeline  # Import from transformers library
from gemini_utility import load_gemini_pro_model, gemini_pro_response, gemini_pro_vision_response, \
    embeddings_model_response
from text_translate import translate_text  # Import the translate_text function
import pyttsx3  # For text-to-speech

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Function to speak the assistant's response
def speak_response(response_text, speak_enabled=True):
    if speak_enabled:
        engine = pyttsx3.init()
        engine.say(response_text)
        engine.runAndWait()

# Function for sentiment analysis
def analyze_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)
    return result[0]['label']

# Set page configuration
st.set_page_config(
    page_title="Query Agent",
    page_icon="🤖",
    layout="centered"
)

# Sidebar menu options with speech recognition toggle
with st.sidebar:
    selected = option_menu(
        'Query Agent',
        ['ChatBot', 'Image Captioning', 'Embed text', 'Ask me anything', 'Text Translation', 'Sentiment Analysis'],
        menu_icon='robot',
        icons=['chat-fill', 'file-image-fill', 'textarea-t', 'patch-question-fill', 'file-text-fill', 'heart-fill'],  # Assuming 'heart-fill' is your sentiment analysis icon
        default_index=0
    )

    use_speech_recognition = st.checkbox("Enable Speech Recognition", value=False)

# Functionality sections
if selected == 'ChatBot':
    st.title("🤖 ChatBot")
    model = load_gemini_pro_model()

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Option to enable/disable speaking responses
    speak_enabled = st.checkbox("Speak Responses", value=True)

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    if use_speech_recognition:
        # Handle microphone access and speech recognition
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        with microphone as source:
            st.info("Listening...")
            audio = recognizer.listen(source)

        try:
            user_prompt = recognizer.recognize_google(audio)
            st.write("You said:", user_prompt)  # Display recognized text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            user_prompt = None  # Fallback to text input
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            user_prompt = None  # Fallback to text input
    else:
        user_prompt = st.chat_input("Ask Query here...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)
            speak_response(gemini_response.text, speak_enabled)  # Speak the response if enabled

elif selected == "Image Captioning":
    st.title("📷 Instant Captions")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image is not None:
        image = Image.open(uploaded_image)
        resized_img = image.resize((800, 500))

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(resized_img, caption="Uploaded Image", use_column_width=True)
        with col2:
            caption = gemini_pro_vision_response("write a short caption for this image", image)
            st.info(caption)

elif selected == "Embed text":
    st.title("🪬 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response") and user_prompt:
        response = embeddings_model_response(user_prompt)
        st.markdown(response)

elif selected == "Ask me anything":
    st.title("❓ Ask me a question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response") and user_prompt:
        response = gemini_pro_response(user_prompt)
        st.markdown(response)

elif selected == "Text Translation":  # New section for Text Translation
    st.title("🖺 Text Translation")

    # Default option for target language
    target_lang = st.selectbox("Select Target Language", ['None'] + list(LANGUAGES.values()))

    text_to_translate = st.text_area("Enter text to translate")

    if st.button("Translate"):
        if target_lang != 'None' and text_to_translate:
            try:
                translated_text = translate_text(text_to_translate, target_lang)
                st.success(f"Translated text ({target_lang}): {translated_text}")
            except Exception as e:
                st.error(f"Translation Error: {e}")
        else:
            st.warning("Please select a target language and provide text to translate")

elif selected == "Sentiment Analysis":  # New section for Sentiment Analysis
    st.title("🫀 Sentiment Analysis")

    user_input = st.text_area("Enter a sentence to analyze sentiment:")

    if st.button("Analyze Sentiment") and user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
