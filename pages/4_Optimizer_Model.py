import streamlit as st
import Data
st.title("Tham số tốt nhất của mô hình")
st.write(Data.best_params)
st.title("Tóm tắt mô hình tốt nhất")
Data.best_model.summary(line_length=110, print_fn=lambda x: st.text(x))