import streamlit as st
import sys
import Data

# Hiển thị DataFrame
st.title("Chuyển đổi tập Train - Test  - Validation sang ma trận số")
col1, col2 = st.columns(2)
# Hiển thị DataFrame
with col1:
    st.header("Câu hỏi - train")
    st.dataframe(Data.encoder_inp_train)
with col2:
    st.header("Trả lời - train")
    st.dataframe(Data.decoder_inp_train)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - test")
    st.dataframe(Data.encoder_inp_test)
with col2:
    st.header("Trả lời - test")
    st.dataframe(Data.decoder_inp_test)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - validation")
    st.dataframe(Data.encoder_inp_valid)
with col2:
    st.header("Trả lời - validation")
    st.dataframe(Data.decoder_inp_vaid)