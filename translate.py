import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize


def translate(source, ct_model, sp_source_model, sp_target_model, device="cpu"):
    """Use CTranslate model to translate a sentence

    Args:
        source (str): A source sentence to translate
        ct_model (str): The path to the CTranslate model
        sp_source_model (str): The path to the SentencePiece source model
        sp_target_model (str): The path to the SentencePiece target model
        device (str): "cpu" (default) or "cuda"
    Returns:
        Translation of the source text.
    """

    source_sentences = sent_tokenize(source)
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    translations = translator.translate_batch(source_tokenized)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = sp_target_model.decode(translations)
    translation = " ".join(translations_detokenized)

    return translation


# [Modify] File paths here to the CTranslate2 SentencePiece models.
ct_model_path = "/path/to/the/ctranslate/model/directory"
sp_source_model_path = "/path/to/the/sentencepiece/source/model/file"
sp_target_model_path = "/path/to/the/sentencepiece/target/model/file"

# Create objects of CTranslate2 Translator and SentencePieceProcessor to load the models
translator = ctranslate2.Translator(ct_model_path, "cpu")    # or "cuda" for GPU
sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)


# Title for the page and nice icon
st.set_page_config(page_title="NMT", page_icon="🤖")
# Header
st.title("Translate")

# Form to add your items
with st.form("my_form"):
    # Textarea to type the source text.
    user_input = st.text_area("Source Text", max_chars=200)
    # Translate with CTranslate2 model
    translation = translate(user_input, translator, sp_source_model, sp_target_model)

    # Create a button
    submitted = st.form_submit_button("Translate")
    # If the button pressed, print the translation
    # Here, we use "st.info", but you can try "st.write", "st.code", or "st.success".
    if submitted:
        st.write("Translation")
        st.info(translation)


# Optional Style
# Source: https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
