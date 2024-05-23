import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import confusion_matrix
# from keras.utils.np_utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv('possible_datasets/cyber-threat-intelligence_all.csv')

print(df.head())

print(df.info())

# Drop the Unnamed: 0 column
df = df.drop(labels='Unnamed: 0', axis=1)

# in label column, raplace the NaN values with benign
df['label'].fillna(value='benign', inplace=True)

# in entire data frame, replace the NaN values with 0
df=df.fillna(0)
print(df.info())

print(df['label'].unique())

data = df[['text','label']]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(text)

    tokens = [token.lower() for token in tokens if token not in punctuations]

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in stop_words]

    processed_text = ' '.join(tokens)
    
    return processed_text

data['text_new'] = data['text'].apply(preprocess_text)
print(data.head())

# drop the text column
data = data.drop(labels=['text'],axis=1)

# remove special characters
data['text_new'] = data['text_new'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

print(data.head())

# change any label listed below to NEED_ATTENTION
data.label.loc[
    (data['label']=="MD5") | 
    (data['label']=="REGISTRYKEY") | 
    (data['label']=="EMAIL") |  
    (data['label']=="Infrastucture") | 
    (data['label']=="DOMAIN") | 
    (data['label']=="SHA1") | 
    (data['label']=="IPV4") | 
    (data['label']=="campaign") | 
    (data['label']=="URL") | 
    (data['label']=="SHA2") | 
    (data['label']=="vulnerability") | 
    (data['label']=="FILEPATH") | 
    (data['label']=="tools") | 
    (data['label']=="TIME") | 
    (data['label']=="url") | 
    (data['label']=="hash")
] = "NeedChecking"

print(data.head())

print(data['label'].value_counts())

# print(data['text_new'][0])

max_length = data['text_new'].str.len().max()
print(max_length)

max_length_tokens = data['text_new'].apply(lambda x: len(x.split())).max()
max_length_characters = data['text_new'].apply(lambda x: len(x)).max()
print('Maximum Sequence Length (Tokens):', max_length_tokens)
print('Maximum Sequence Length (Characters):', max_length_characters)

# max number of words in the vocabulary (common words)
MAX_NB_WORDS = 50000
# max length of input sequence (max sentence length)
MAX_SEQ_LENGTH = 450
# number of vectors in vector representation of word/sentence
EMBEDDING_DIM=100

#rename column from text_new to text
data = data.rename(columns={'text_new':'text'})

# build vocabulary from the text by seperating words
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%()*+,-./:;<=>?@[\]^_`{|}',lower=True)
# each unique word is assigned a unique number
tokenizer.fit_on_texts(data['text'].values)
word_index = tokenizer.word_index
print('Number of words:', len(word_index))

# convert text to sequence of numbers
X = tokenizer.texts_to_sequences(data['text'].values)
#print(X[0:8])
# pad sequences to make them of equal length
X = pad_sequences(X, maxlen=MAX_SEQ_LENGTH)
#print(X[0:8])
print('Shape of data tensor X:', X.shape)
# print('X:', X)

# encode the labels into boolean values where the length of each is equal to the number of unique labels
Y = pd.get_dummies(data['label']).values
# print('Y:', Y)

from sklearn.preprocessing import LabelBinarizer
# instantiate the label binarizer class
lb = LabelBinarizer()
# encode the labels into binary values where the length of each is equal to the number of unique labels
Y = lb.fit_transform(data['label'])
print('Y:', Y)
print('Shape of data tensor Y:', Y.shape)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

# load model
from tensorflow.keras.models import load_model
model = load_model('models/model.h5')

print(model.summary())

# model evaluation
accr = model.evaluate(x_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# check 5 of the model predictions
predictions = model.predict(x_test)
for i in range(5):
    print('Prediction: {}'.format(predictions[i]))
    print('Actual: {}'.format(y_test[i]))

# predicts probabilities of each label for each row in the test set
# predictions = model.predict(x_test)
# print('predictions:', predictions)
# assigns the label with the highest probability to y_pred
y_pred = np.argmax(model.predict(x_test), axis=1)
print('Y_pred:', y_pred)

# confusion matrix to evaluate the model labeling capability
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class_labels = lb.classes_
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)

print('y_true:', y_true)
print('y_pred:', y_pred)
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:', cm)

classes = np.arange(len(cm))

plt.figure(figsize=(14, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
