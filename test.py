# Simple streamlit test
# To launch your app, run: streamlit run test.py

import streamlit as st

st.title("Upper My Text")

user_input = st.text_input("Type some text and press Enter \
    to convert it to the UPPER case.")

if len(user_input) > 0:
    output = user_input.upper()
    st.info(output)
