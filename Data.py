import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.models import load_model
from keras.layers import Embedding, LSTM, Dense
from keras import Input, Model
from pyvi import ViTokenizer
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kerastuner.tuners import RandomSearch
import streamlit as st
# import Thamso
# import sys
# sys.path.append('pages')
# from pages import Split_Data
#----------------------------------Đọc tập dữ liệu---------------------------------
df = pd.read_csv('Book2.csv')
st.title("TẬP DỮ LIỆU")
st.dataframe(df)
#---------------------------------Chia thành 2 cột Câu hỏi và Sample--------------------------
questions = df['Câu hỏi '].values
samples = df['Sample'].values
phanloai= df['Phân Loại '].values
data_questions = []
for i in range(len(df)):
    question = questions[i] if i < len(questions) else ''
    similar_question = samples[i] if i < len(samples) else ''
    data_questions.append(question)
    data_questions.append(similar_question)
    
data_phanloai = []
for i in range(len(phanloai)):
    data_phanloai.append(phanloai[i])
    data_phanloai.append(phanloai[i])    

answers = df['Trả lời'].values
data_answers = []
for i in range(len(answers)):
    data_answers.append(answers[i])
    data_answers.append(answers[i])

st.title("Tách riêng cột 'Câu hỏi' và 'Trả lời'")
col1, col2 = st.columns(2)
with col1:
    st.header("Câu hỏi")
    st.dataframe(data_questions)
with col2:
    st.header("Trả lời")
    st.dataframe(data_answers)

#-----------------------------------------------Chia tập dữ liệu------------------------------------------------------------------------------
def split_train_test_validation(train_size, val_size, random_state):
    test_size = 1 - train_size - val_size

    train_questions, temp_questions, train_answers, temp_answers = train_test_split(data_questions, data_answers, test_size=val_size + test_size, random_state=random_state)
    val_questions, test_questions, val_answers, test_answers = train_test_split(temp_questions, temp_answers, test_size=test_size / (val_size + test_size), random_state=random_state)
    
    return train_questions, val_questions, test_questions, train_answers, val_answers, test_answers


st.title("Chia tập dữ liệu Train - Test - Validation ")
train_size = st.slider("Train Size:", 0.1, 0.8, 0.8, 0.1)
val_size = st.slider("Validation Size:", 0.1, 0.8, 0.1, 0.1)
test_size = st.slider("Test Size:", 0.1, 0.8, 1 - train_size - val_size, 0.1)

train_questions, val_questions, test_questions, train_answers, val_answers, test_answers = split_train_test_validation(train_size, val_size, random_state=42)


col1, col2 = st.columns(2)
train_data = pd.DataFrame({'Question': train_questions, 'Answer': train_answers})

test_data = pd.DataFrame({'Question': test_questions, 'Answer': test_answers})

validation_data = pd.DataFrame({'Question': val_questions, 'Answer': val_answers})


# Hiển thị DataFrame
with col1:
    st.header("Tập train")
    st.dataframe(train_data)
with col2:
    st.header("Tập test")
    st.dataframe(test_data)
# Hiển thị DataFrame
st.header("Tập Validation")
st.dataframe(validation_data)

st.write(f"Train Size: {len(train_questions)} samples")
st.write(f"Validation Size: {len(val_questions)} samples")
st.write(f"Test Size: {len(test_questions)} samples")

#-------------------------------------------Xóa các ký tự đặc biệt--------------------------------    
def clean_text(sent):
    return re.sub(r'[!“”"#$%&\'()*+,-.:;<=>?@[\]^_`{|}~]', '', sent)

data_questions_clean_train = [clean_text(question.lower()) for question in train_questions]
data_answers_clean_train = [clean_text(answer.lower()) for answer in train_answers]

data_questions_clean_test = [clean_text(question.lower()) for question in test_questions]
data_answers_clean_test = [clean_text(answer.lower()) for answer in test_answers]

data_questions_clean_valid = [clean_text(question.lower()) for question in val_questions]
data_answers_clean_valid = [clean_text(answer.lower()) for answer in val_answers]



#---------------------------------------------Xóa từ Stopword-----------------------------------------    
def clean_and_word_segmentation(sent):
    return word_tokenize(clean_text(sent.lower()), format='text')

def read_stop_words_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        stop_words = [word.strip() for word in file.readlines()]
    return stop_words

def preprocess_text(text):
    stop_words = read_stop_words_from_file('./vietnamese-stopwords.txt')
    cleaned_text = clean_text(text.lower())
    words = ViTokenizer.tokenize(cleaned_text)
    words = words.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data_questions_clean_stopword_train = [preprocess_text(question) for question in data_questions_clean_train]
data_questions_clean_stopword_test = [preprocess_text(question) for question in data_questions_clean_test]
data_questions_clean_stopword_valid = [preprocess_text(question) for question in data_questions_clean_valid]


all_questions = data_questions_clean_stopword_train + data_questions_clean_stopword_test + data_questions_clean_stopword_valid
longest_question = max(all_questions, key=lambda x: len(x.split()))
max_word_count = len(longest_question.split())
def prepare_data(questions, answers):
    count_words_ques = [len(clean_text(ques).split()) for ques in questions]
    counter_words_ques = Counter(count_words_ques)
    list_count_word = []
    list_count_sent = []
    for i in counter_words_ques.items():
        list_count_word.append(i[0])
        list_count_sent.append(i[1])
    sorted_ques = []
    sorted_ans = []
    for i, count in enumerate(count_words_ques):
        if count <= max_word_count:
            sorted_ques.append(questions[i])
            sorted_ans.append(answers[i])
    questions = [clean_and_word_segmentation(ques) for ques in sorted_ques]
    answers = ['<START> ' + clean_and_word_segmentation(answ) + ' <END>' for answ in sorted_ans]

    return questions, answers

questions_pre_train,answers_pre_train = prepare_data(data_questions_clean_stopword_train,data_answers_clean_train)
questions_pre_test,answers_pre_test = prepare_data(data_questions_clean_stopword_test,data_answers_clean_test)
questions_pre_valid,answers_pre_valid = prepare_data(data_questions_clean_stopword_valid,data_answers_clean_valid)
#----------------------------------------------------Chuyển đổi tập train và test sang vecto------------------------------------------
tokenizer = Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(questions_pre_train + answers_pre_train + questions_pre_test + answers_pre_test + questions_pre_valid + answers_pre_valid) 
VOCAB_SIZE = len(tokenizer.word_index) + 1 

# encoder 
tokenized_questions_train = tokenizer.texts_to_sequences(questions_pre_train) 
encoder_inp_train = pad_sequences(tokenized_questions_train,maxlen=max_word_count,padding='post')

tokenized_questions_test = tokenizer.texts_to_sequences(questions_pre_test) 
encoder_inp_test = pad_sequences(tokenized_questions_test,maxlen=max_word_count,padding='post')

tokenized_questions_valid = tokenizer.texts_to_sequences(questions_pre_valid) 
encoder_inp_valid = pad_sequences(tokenized_questions_valid,maxlen=max_word_count,padding='post')

#decoder 
tokenized_answers_train = tokenizer.texts_to_sequences(answers_pre_train) 
maxlen_answers_train = np.max([len(x) for x in tokenized_answers_train])

tokenized_answers_test = tokenizer.texts_to_sequences(answers_pre_test) 
maxlen_answers_test = np.max([len(x) for x in tokenized_answers_test]) 

tokenized_answers_valid = tokenizer.texts_to_sequences(answers_pre_valid) 
maxlen_answers_valid = np.max([len(x) for x in tokenized_answers_valid]) 
 
decoder_inp_train = pad_sequences(tokenized_answers_train, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post')

decoder_inp_test = pad_sequences(tokenized_answers_test, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post')

decoder_inp_vaid = pad_sequences(tokenized_answers_valid, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post')
# Decord_final_output 
for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:] 
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post') 
decoder_final_output_train = to_categorical(padded_answers_train, VOCAB_SIZE)   

for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]

padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post') 
decoder_final_output_test = to_categorical(padded_answers_test, VOCAB_SIZE)

for i in range(len(tokenized_answers_valid)):
    tokenized_answers_valid[i] = tokenized_answers_valid[i][1:] 
    
padded_answers_valid = pad_sequences(tokenized_answers_valid, maxlen=max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test), padding='post') 
decoder_final_output_valid = to_categorical(padded_answers_valid, VOCAB_SIZE)


#--------------------------------------------------------Xây dưng model-----------------------------------------------------
def build_model(hp):
    enc_inputs = Input(shape=(None,),name="encoder_input")
    enc_embedding = Embedding(VOCAB_SIZE, hp.Int('embedding_size', min_value=100, max_value=400, step=50), mask_zero=True)(enc_inputs)
    _, state_h, state_c = LSTM(hp.Int('lstm_units', min_value=100, max_value=400, step=50), return_state=True)(enc_embedding)
    enc_states = [state_h, state_c]
        
    dec_inputs = Input(shape=(None,),name="decoder_input")
    dec_embedding = Embedding(VOCAB_SIZE, hp.Int('embedding_size', min_value=100, max_value=400, step=50), mask_zero=True)(dec_inputs)
    dec_lstm = LSTM(hp.Int('lstm_units', min_value=100, max_value=400, step=50), return_state=True, return_sequences=True)
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
        
    dec_dense = Dense(VOCAB_SIZE, activation='softmax')
    output = dec_dense(dec_outputs)
        
    model = Model([enc_inputs, dec_inputs], output)
        
    hp_optimizer = hp.Choice('optimizer', values=['Adam'])
        
    model.compile(optimizer=hp_optimizer, loss='categorical_crossentropy', metrics='accuracy')
        
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=64, 
    directory='my_tuner_dir',
    project_name='my_seq2seq_tuner'
)

tuner.search([encoder_inp_train, decoder_inp_train], decoder_final_output_train, epochs=10,validation_data=([encoder_inp_valid, decoder_inp_vaid], decoder_final_output_valid))
best_model = tuner.get_best_models(num_models=1)[0] 
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_params = best_trial.hyperparameters.values  


enc_inputs = Input(shape=(None,),name="encoder_input")
enc_embedding = Embedding(VOCAB_SIZE, best_params['embedding_size'], mask_zero=True)(enc_inputs)
_, state_h, state_c = LSTM(best_params['lstm_units'], return_state=True)(enc_embedding)
enc_states = [state_h, state_c]
dec_inputs = Input(shape=(None,),name="decoder_input") 
dec_embedding = Embedding(VOCAB_SIZE, best_params['embedding_size'], mask_zero=True)(dec_inputs) 
dec_lstm = LSTM(best_params['lstm_units'], return_state=True, return_sequences=True) 

dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states) 
dec_dense = Dense(VOCAB_SIZE, activation='softmax')
output = dec_dense(dec_outputs)

model = Model([enc_inputs, dec_inputs], output) 
model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics = 'accuracy')   

def make_inference_models():
    dec_state_input_h = Input(shape=(400,))
    dec_state_input_c = Input(shape=(400,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                            initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)            
    dec_model = Model(                              
        inputs=[dec_inputs] + dec_states_inputs,
        outputs = [dec_outputs] + dec_states)
    # dec_model.summary()
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
    # enc_model.summary() 
    return enc_model, dec_model 

def str_to_tokens(sentence):
    words = clean_and_word_segmentation(sentence).split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word)
        if result is not None:
            tokens_list.append(result)
    return pad_sequences([tokens_list], maxlen=max_word_count,padding='post')

enc_model, dec_model = make_inference_models()
def chatbot(input_question):
    preprocessed_input_question = preprocess_text(input_question)

    states_values = enc_model.predict(str_to_tokens(preprocessed_input_question))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['<START>']
    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None

        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != '<END>':
                    decoded_translation += f'{word} '
                sampled_word = word

        if sampled_word == '<END>' or len(decoded_translation.split()) > max(maxlen_answers_valid,maxlen_answers_train,maxlen_answers_test):
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
        decoded_translation = re.sub(r'_', ' ', decoded_translation)
        # st.write('Bot answer:', decoded_translation)
    return decoded_translation
