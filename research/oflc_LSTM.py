from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


abel_encoder= LabelEncoder()
df['label'] = label_encoder.fit_transform(df['case_status'])
y = to_categorical(df['label'])


max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
# Create a 'text' column by concatenating relevant columns
df['text'] = df[['continent', 'education_of_employee', 'has_job_experience', 
				 'requires_job_training', 'region_of_employment', 
				 'unit_of_wage', 'full_time_position', 'case_status']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=max_len)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='sigmoid'))  # Use 'sigmoid' for multi-label classification

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)