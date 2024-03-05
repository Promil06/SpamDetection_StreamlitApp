import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import RandomizedSearchCV

# Load the data
# List of potential encodings to try
encodings = ['utf-8', 'latin-1', 'ISO-8859-1']

# Try reading the file with different encodings
for encoding in encodings:
    try:
        data = pd.read_csv(r"C:\Users\DELL\Desktop\Codebays\SpamFilteration\spam.csv", encoding=encoding)
        print(f"File successfully read with encoding: {encoding}")
        break  # Break out of the loop if successful
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
data.columns = ['kind', 'message']


# Preprocessing function

stop_words = set(stopwords.words('english'))
stemmer = LancasterStemmer()
def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic|S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])

    words = nltk.word_tokenize(text)
    words = [i for i in words if i not in stop_words and len(i) > 2]

    text = " ".join(words)
    text = re.sub(r'\s+', " ", text).strip()

    return text


# Apply preprocessing
data['CleanMessage'] = data['message'].apply(cleaning_data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['CleanMessage'])

# One hot encoding for 'kind'
encoder = OneHotEncoder(sparse=False, drop='first')
Y = encoder.fit_transform(data['kind'].values.reshape(-1, 1))

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models with their parameters
svm = SVC()
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()

# Define parameter grids for randomized search
svm_params = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [100, 50]}
dt_params = {'max_depth': np.arange(2, 10, 2)}
rf_params = {'n_estimators': np.arange(20, 101, 10), 'max_depth': np.arange(2, 17, 2)}

# Train models with randomized search
svm_rs = RandomizedSearchCV(svm, param_distributions=svm_params, cv=3, random_state=42)
svm_rs.fit(x_train, y_train)

dt_classifier_rs = RandomizedSearchCV(dt_classifier, param_distributions=dt_params, random_state=42)
dt_classifier_rs.fit(x_train, y_train)

rf_classifier_rs = RandomizedSearchCV(rf_classifier, param_distributions=rf_params, random_state=42)
rf_classifier_rs.fit(x_train, y_train)

# Evaluate models based on accuracy and precision scores
svm_accuracy = accuracy_score(y_true=y_test, y_pred=svm_rs.predict(x_test))
svm_precision = precision_score(y_true=y_test, y_pred=svm_rs.predict(x_test))

dt_accuracy = accuracy_score(y_true=y_test, y_pred=dt_classifier_rs.predict(x_test))
dt_precision = precision_score(y_true=y_test, y_pred=dt_classifier_rs.predict(x_test))

rf_accuracy = accuracy_score(y_true=y_test, y_pred=rf_classifier_rs.predict(x_test))
rf_precision = precision_score(y_true=y_test, y_pred=rf_classifier_rs.predict(x_test))

# Choose the best model based on accuracy and precision scores
best_model = None
if (svm_accuracy + svm_precision) > (dt_accuracy + dt_precision) and (svm_accuracy + svm_precision) > (
        rf_accuracy + rf_precision):
    best_model = svm_rs
elif (dt_accuracy + dt_precision) > (rf_accuracy + rf_precision):
    best_model = dt_classifier_rs
else:
    best_model = rf_classifier_rs

# Streamlit app
st.title('Spam Message Classification')

# Input text box for user message
message = st.text_area('Enter the message:', '')

if st.button('prediction'):
    # Preprocess the input message
    cleaned_message = cleaning_data(message)

    # Vectorize the cleaned message
    vectorized_message = vectorizer.transform([cleaned_message])

    # Make prediction using the best model
    prediction = best_model.predict(vectorized_message)

    # Display prediction result
    if prediction[0] == 1:
        st.error('Spam Message!')
    elif prediction[0] == 0:
        st.success('Not a Spam Message')

