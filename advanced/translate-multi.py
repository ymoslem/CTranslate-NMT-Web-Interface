import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize


def translate(source, translator, sp_source_model, sp_target_model):
    """Use CTranslate model to translate a sentence

    Args:
        source (str): A source sentence to translate
        translator (object): Object of Translator, with the CTranslate2 model
        sp_source_model (object): The path to the SentencePiece source model
        sp_target_model (object): The path to the SentencePiece target model
    Returns:
        Translation of the source text
    """

    source_sentences = sent_tokenize(source)  # split sentences
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    translations = translator.translate_batch(source_tokenized, replace_unknowns=True)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = sp_target_model.decode(translations)

    return translations_detokenized


# [Modify] File paths here to the CTranslate2 and SentencePiece models.
@st.cache(allow_output_mutation=True)
def load_models(lang_pair, device="cpu"):
    """Load CTranslate2 model and SentencePiece models

    Args:
        lang_pair (str): Language pair to load the models for
        device (str): "cpu" (default) or "cuda"
    Returns:
        Model objects of Translator and SentencePieceProcessor
    """
    if lang_pair == "English-to-French":
        ct_model_path = "/path/to/your/ctranslate2/model/"
        sp_source_model_path = "/path/to/your/sp_source.model"
        sp_target_model_path = "/path/to/your/sp_target.model"
    elif lang_pair == "French-to-English":
        ct_model_path = "/path/to/your/ctranslate2/model/"
        sp_source_model_path = "/path/to/your/sp_source.model"
        sp_target_model_path = "/path/to/your/sp_target.model"

    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)
    translator = ctranslate2.Translator(ct_model_path, device)

    return translator, sp_source_model, sp_target_model


# Title for the page and nice icon
st.set_page_config(page_title="NMT", page_icon="🤖")
# Header
st.title("Translate")

# Form to add your items
with st.form("my_form"):

    # Dropdown menu to select a language pair
    lang_pair = st.selectbox("Select Language Pair",
                             ("English-to-French", "French-to-English"))
    # st.write('You selected:', lang_pair)

    # Textarea to type the source text.
    user_input = st.text_area("Source Text", max_chars=200)
    sources = user_input.split("\n")  # split on new line.

    # Load models
    translator, sp_source_model, sp_target_model = load_models(lang_pair, device="cpu")

    # Translate with CTranslate2 model
    translations = [translate(source, translator, sp_source_model, sp_target_model) for source in sources]
    translations = [" ". join(translation) for translation in translations] 

    # Create a button
    submitted = st.form_submit_button("Translate")
    # If the button pressed, print the translation
    if submitted:
        st.write("Translation")
        st.code("\n".join(translations))


# Optional Style
st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .reportview-container .main .block-container{
        padding-top: 0rem;
        padding-right: 0rem;
        padding-left: 0rem;
        padding-bottom: 0rem;
    } </style> """, unsafe_allow_html=True)
