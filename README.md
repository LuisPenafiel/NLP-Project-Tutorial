# NLP Project: Spam Email Detection

## Overview

This project focuses on building a machine learning model to classify emails as either spam or not spam (ham). The project involves data preprocessing, text analysis, and the implementation of various machine learning models to achieve accurate classification. The dataset used in this project contains URLs and corresponding labels indicating whether the URL is associated with spam.

## Project Structure

The project is organized into several key steps:

1. **Installation of Required Libraries**: The project requires several Python libraries, including `regex`, `nltk`, `wordcloud`, `gensim`, `tensorflow`, and `keras`. These libraries are essential for text processing, visualization, and model building.

2. **Data Loading and Preprocessing**:
   - The dataset is loaded from a CSV file containing URLs and their corresponding spam labels.
   - The text data is preprocessed by converting it to lowercase, removing special characters, and lemmatizing the words.
   - Stopwords are removed to reduce noise in the text data.

3. **Exploratory Data Analysis (EDA)**:
   - The distribution of spam and non-spam emails is visualized using a pie chart.
   - A word cloud is generated to visualize the most common words in the dataset.

4. **Model Building**:
   - The text data is transformed into numerical features using TF-IDF vectorization.
   - The dataset is split into training and testing sets.
   - A Support Vector Machine (SVM) model is trained and evaluated based on accuracy, F1 score, and ROC-AUC score.

5. **Advanced Model (Optional)**:
   - A more complex model using LSTM (Long Short-Term Memory) networks is implemented for better performance on text classification tasks.

## Installation

To set up the environment for this project, you need to install the required Python libraries. You can do this by running the following commands:

```bash
pip install regex nltk wordcloud gensim tensorflow keras
```

Additionally, you may need to download specific NLTK datasets:

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

## Usage

1. **Data Loading**:
   - The dataset is loaded from a CSV file using `pandas`.

   ```python
   total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")
   ```

2. **Data Preprocessing**:
   - The text data is preprocessed to remove special characters, convert to lowercase, and lemmatize words.

   ```python
   def preprocess_text(text):
       text = re.sub(r'[^a-z ]', " ", text)
       text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
       text = re.sub(r'\^[a-zA-Z]\s+', " ", text)
       text = re.sub(r'\s+', " ", text.lower())
       text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
       return text.split()
   ```

3. **Exploratory Data Analysis**:
   - The distribution of spam and non-spam emails is visualized.

   ```python
   total_data["is_spam"].value_counts().plot.pie()
   plt.legend(["Not Spam", "Spam"])
   plt.show()
   ```

4. **Model Training**:
   - The text data is transformed using TF-IDF vectorization, and an SVM model is trained.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.svm import SVC
   from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(total_data['url'])
   y = total_data['is_spam']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = SVC()
   model.fit(X_train, y_train)

   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("F1 Score:", f1_score(y_test, y_pred))
   print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
   ```

5. **Advanced Model (LSTM)**:
   - An LSTM model is implemented using TensorFlow and Keras for more complex text classification.

   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   from tensorflow.keras.utils import to_categorical
   from tensorflow.keras.layers import Embedding, Bidirectional, Input, LSTM, Dense, Dropout
   from tensorflow.keras.models import Model, load_model
   from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(total_data['url'])
   sequences = tokenizer.texts_to_sequences(total_data['url'])
   padded_sequences = pad_sequences(sequences, maxlen=100)

   model = Sequential()
   model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=100))
   model.add(Bidirectional(LSTM(64)))
   model.add(Dense(64, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(1, activation='sigmoid'))

   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(padded_sequences, total_data['is_spam'], epochs=10, validation_split=0.2)
   ```

## Conclusion

This project demonstrates the process of building a spam email detection system using natural language processing techniques and machine learning models. The project covers data preprocessing, exploratory data analysis, and the implementation of both traditional and advanced machine learning models. The final model can be used to classify emails as spam or not spam with high accuracy.

## Future Work

- Experiment with different machine learning models and compare their performance.
- Explore more advanced text preprocessing techniques, such as word embeddings.
- Deploy the model as a web application for real-time spam detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.