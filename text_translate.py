from googletrans import Translator, LANGUAGES
import streamlit as st

def translate_text(text, target_lang):
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest=target_lang)
        return translated_text.text
    except Exception as e:
        return f"Translation Error: {e}"

if __name__ == "__main__":
    st.title("Text Translation Example")

    source_lang = st.selectbox("Select Source Language", list(LANGUAGES.values()))
    target_lang = st.selectbox("Select Target Language", list(LANGUAGES.values()))

    text_to_translate = st.text_area("Enter text to translate")

    if st.button("Translate"):
        if source_lang and target_lang and text_to_translate:
            try:
                translated_text = translate_text(text_to_translate, source_lang, target_lang)
                st.success(f"Translated text ({target_lang}): {translated_text}")
            except Exception as e:
                st.error(f"Translation Error: {e}")
        else:
            st.warning("Please fill in all fields")
