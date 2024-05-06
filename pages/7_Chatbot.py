import streamlit as st
import Data
import pandas as pd
from keras.preprocessing.text import Tokenizer
st.markdown(
    """
    <style>
    .title {
        font-size: 50px; /* Thay đổi cỡ chữ tiêu đề theo ý muốn */
        color: #FF5733; /* Thay đổi màu sắc của tiêu đề theo ý muốn */
    }
    </style>
    """,
    unsafe_allow_html=True
)
df_tonghop = pd.DataFrame({'Question': Data.data_questions,'Phan Loai':Data.data_phanloai })
question_hoctap_tailieu = df_tonghop[df_tonghop['Phan Loai'] == 'Học tập và Tài Liệu học tập']['Question']
question_hoatdong_ngoaikhoa = df_tonghop[df_tonghop['Phan Loai'] == 'Sinh viên Quốc tế và Hoạt động ngoại khóa']['Question']
question_trocap_CTXH = df_tonghop[df_tonghop['Phan Loai'] == 'Trợ cấp và Công tác xã hội']['Question']
question_tuyensinh = df_tonghop[df_tonghop['Phan Loai'] == 'Tuyển sinh ']['Question']
question_hocphi_chinhsach = df_tonghop[df_tonghop['Phan Loai'] == 'Học phí và Chính sách Tài chính']['Question']

question_hoctap_tailieu_clean = [Data.clean_text(question.lower()) for question in question_hoctap_tailieu]
question_hoatdong_ngoaikhoa_clean = [Data.clean_text(question.lower()) for question in question_hoatdong_ngoaikhoa]
question_trocap_CTXH_clean = [Data.clean_text(question.lower()) for question in question_trocap_CTXH]
question_tuyensinh_clean = [Data.clean_text(question.lower()) for question in question_tuyensinh]
question_hocphi_chinhsach_clean = [Data.clean_text(question.lower()) for question in question_hocphi_chinhsach]

question_hoctap_tailieu_clean_stopword = [Data.preprocess_text(question) for question in question_hoctap_tailieu_clean]
question_hoatdong_ngoaikhoa_clean_stopword = [Data.preprocess_text(question) for question in question_hoatdong_ngoaikhoa_clean]
question_trocap_CTXH_clean_stopword = [Data.preprocess_text(question) for question in question_trocap_CTXH_clean]
question_tuyensinh_clean_stopword = [Data.preprocess_text(question) for question in question_tuyensinh_clean]
question_hocphi_chinhsach_clean_stopword = [Data.preprocess_text(question) for question in question_hocphi_chinhsach_clean]

tokenizer_hoctap_tailieu = Tokenizer(filters='', lower=False)
tokenizer_hoatdong_ngoaikhoa = Tokenizer(filters='', lower=False)
tokenizer_trocap_CTXH = Tokenizer(filters='', lower=False)
tokenizer_tuyensinh = Tokenizer(filters='', lower=False)
tokenizer_hocphi_chinhsach = Tokenizer(filters='', lower=False)


tokenizer_hoctap_tailieu.fit_on_texts(question_hoctap_tailieu_clean_stopword)
tokenizer_hoatdong_ngoaikhoa.fit_on_texts(question_hoatdong_ngoaikhoa_clean_stopword)
tokenizer_trocap_CTXH.fit_on_texts(question_trocap_CTXH_clean_stopword)
tokenizer_tuyensinh.fit_on_texts(question_tuyensinh_clean_stopword)
tokenizer_hocphi_chinhsach.fit_on_texts(question_hocphi_chinhsach_clean_stopword)

vocab_hoctap_tailieu = tokenizer_hoctap_tailieu.word_index
vocab_hoatdong_ngoaikhoa = tokenizer_hoatdong_ngoaikhoa.word_index
vocab_trocap_CTXH = tokenizer_trocap_CTXH.word_index
vocab_tuyensinh = tokenizer_tuyensinh.word_index
vocab_hocphi_chinhsach = tokenizer_hocphi_chinhsach.word_index

st.markdown('<p class="title">Trả lời tự động bằng học sâu</p>', unsafe_allow_html=True)  
selected_vocab = st.selectbox("Lĩnh Vực", ["Học tập và Tài Liệu học tập", "Sinh viên Quốc tế và Hoạt động ngoại khóa", "Trợ cấp và Công tác xã hội", "Tuyển sinh", "Học phí và Chính sách Tài chính"])

if selected_vocab == "Học tập và Tài Liệu học tập":
    selected_word_index = vocab_hoctap_tailieu
elif selected_vocab == "Sinh viên Quốc tế và Hoạt động ngoại khóa":
    selected_word_index = vocab_hoatdong_ngoaikhoa
elif selected_vocab == "Trợ cấp và Công tác xã hội":
    selected_word_index = vocab_trocap_CTXH
elif selected_vocab == "Tuyển sinh":
    selected_word_index = vocab_tuyensinh
elif selected_vocab == "Học phí và Chính sách Tài chính":
    selected_word_index = vocab_hocphi_chinhsach
    
question=st.text_area("Nhập câu hỏi của bạn ở bên dưới")
button=st.button('✉️ Send', key='send')
if question and button:
    input_tokens = Data.preprocess_text(question).split()
    st.write(input_tokens)
    total_tokens = len(input_tokens)
    num_tokens_in_vocab = sum(1 for token in input_tokens if token in selected_word_index)
    st.write(selected_word_index)
    st.write(num_tokens_in_vocab)
    in_vocab = num_tokens_in_vocab / total_tokens
    if in_vocab > 0.4:
        Data.model.load_weights("model_weights.h5")
        with st.spinner("Đợi chút nhen !"):
            reply=Data.chatbot(question)
            st.write("Chatbot: ", reply)
            button2=st.button("Clear")
    else:
        st.write("Chatbot: Tôi chưa học về nó, sẽ cập nhật sau.")