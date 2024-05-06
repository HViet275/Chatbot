import streamlit as st
import Data
import time
import pickle

if st.sidebar.button('🚀 Train Model', key='train'):
    st.title("Huấn luyện mô hình")
    st.write('Training the model...')
    start_time = time.time()
    batch_sizes = [2,4,6,8]
    epochs_values = [15,20,25,30]
    best_params = {'batch_size': None, 'epochs': None}
    best_loss = float('inf')
    for batch_size in batch_sizes:
        for epochs in epochs_values:
            # Đào tạo mô hình với các giá trị siêu tham số hiện tại
            Data.model.fit([Data.encoder_inp_train, Data.decoder_inp_train], Data.decoder_final_output_train, batch_size=batch_size, epochs=epochs, verbose=0)

            # Đánh giá mô hình trên tập kiểm tra
            loss = Data.model.evaluate([Data.encoder_inp_valid, Data.decoder_inp_vaid], Data.decoder_final_output_valid)

            # Kiểm tra nếu giá trị hiện tại tốt hơn giá trị tốt nhất trước đó
            if loss[0] < best_loss:
                best_loss = loss[0]
                best_params['batch_size'] = batch_size
                best_params['epochs'] = epochs
                
                
    Data.model.save_weights("model_weights.h5")
    end_time = time.time()
    training_time = end_time - start_time
    best_params['training_time'] = training_time
    with open("best_params.pickle", "wb") as file:
        pickle.dump(best_params, file)      
    st.write('Training complete')
    st.text(f"Thời gian huấn luyện: {training_time:.2f} giây") 
    

with open("best_params.pickle", "rb") as file:
    best_params = pickle.load(file)
    st.write("Bộ tham số tốt nhất và thời gian training:",best_params)
