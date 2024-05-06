import streamlit as st
import Data
import time
from sklearn.metrics import accuracy_score
import numpy as np
if st.sidebar.button('🚀 Test Model', key='test'):
    Data.model.load_weights("model_weights.h5")
    st.title("Đánh giá mô hình")
    start_time = time.time()
    predicted_output = Data.model.predict([Data.encoder_inp_test, Data.decoder_inp_test])
    predicted_indices = np.argmax(predicted_output, axis=-1)
    true_indices = np.argmax(Data.decoder_final_output_test, axis=-1)
    non_zero_mask = true_indices != 0
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(true_indices[non_zero_mask], predicted_indices[non_zero_mask])
    # Tính precision
    precision = precision_score(true_indices[non_zero_mask], predicted_indices[non_zero_mask], average='weighted')

    # Tính recall
    recall = recall_score(true_indices[non_zero_mask], predicted_indices[non_zero_mask], average='weighted')

    # Tính F1-score
    f1 = f1_score(true_indices[non_zero_mask], predicted_indices[non_zero_mask], average='weighted')
    end_time = time.time()
    test_time = end_time - start_time
    st.write("Accuracy on test set: ", accuracy)
    st.write("Precision on test set: ", precision)
    st.write("Recall on test set: ", recall)
    st.write("F1-score on test set: ", f1)
    st.text(f"Thời gian test mô hình: {test_time:.2f} giây")