import streamlit as st
import sys
import Data

st.title("Xóa các ký tự đặc biệt trên cột 'Câu hỏi' và 'Trả lời'của tập Train-Test-Validation")
col1, col2 = st.columns(2)
# Hiển thị DataFrame
with col1:
    st.header("Câu hỏi - train")
    st.dataframe(Data.data_questions_clean_train)
with col2:
    st.header("Trả lời - train")
    st.dataframe(Data.data_answers_clean_train)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - test")
    st.dataframe(Data.data_questions_clean_test)
with col2:
    st.header("Trả lời - test")
    st.dataframe(Data.data_answers_clean_test)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - validation")
    st.dataframe(Data.data_questions_clean_valid)
with col2:
    st.header("Trả lời - validation")
    st.dataframe(Data.data_answers_clean_valid)
    
    
#--------------------------Xóa các từ stopword-----------------------------------------------
st.title("Xóa các từ StopWord")
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - train")
    st.dataframe(Data.data_questions_clean_stopword_train)
with col2:
    st.header("Trả lời - train")
    st.dataframe(Data.data_answers_clean_train)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - test")
    st.dataframe(Data.data_questions_clean_stopword_test)
with col2:
    st.header("Trả lời - test")
    st.dataframe(Data.data_answers_clean_test)
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi - validation")
    st.dataframe(Data.data_questions_clean_stopword_valid)
with col2:
    st.header("Trả lời - validation")
    st.dataframe(Data.data_answers_clean_valid)