import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize


def tokenize(text, sp_source_model):
    """Use SentencePiece model to tokenize a sentence

    Args:
        text (str or list(str)): A sentence or list of sentences to tokenize
        sp_source_model (str): The path to the SentencePiece source model

    Returns:
        List of tokens of the text, or list of lists (sentences) of tokens
    """

    sp = spm.SentencePieceProcessor(sp_source_model)
    tokens = sp.encode(text, out_type=str)
    return tokens


def detokenize(text, sp_target_model):
    """Use SentencePiece model to detokenize a sentence

    Args:
        text (str or list(str)): A sentence to or list of sentences to detokenize
        sp_target_model (str): The path to the SentencePiece target model

    Returns:
        String of the detokenized text, or list of detokenized sentences
    """

    sp = spm.SentencePieceProcessor(sp_target_model)
    translation = sp.decode(text)
    return translation


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

    translator = ctranslate2.Translator(ct_model, device)
    source_sentences = sent_tokenize(source)  # split sentences
    source_tokenized = tokenize(source_sentences, sp_source_model)
    translations = translator.translate_batch(source_tokenized, replace_unknowns=True)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = detokenize(translations, sp_target_model)

    return translations_detokenized


# [Modify] File paths here to the CTranslate2 model
# and the SentencePiece source and target models.
# try caching @st.cache
def load_models(lang_pair):
    if lang_pair == "English-to-French":
        path = "/Users/yasmin/Documents/Snowball/fr_en/"
        ct_model = path+"model"
        sp_source_model = sp_target_model = path+"sentencepiece.model"
    elif lang_pair == "French-to-English":
        path = "/Users/yasmin/Documents/Snowball/fr_en/"
        ct_model = path+"model"
        sp_source_model = sp_target_model = path+"sentencepiece.model"

    return ct_model, sp_source_model, sp_target_model


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
    ct_model, sp_source_model, sp_target_model = load_models(lang_pair)

    # Translate with CTranslate2 model
    translations = [translate(source, ct_model, sp_source_model, sp_target_model) for source in sources]
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
