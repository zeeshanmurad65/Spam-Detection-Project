import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Download necessary NLTK data (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

# --- Text Preprocessing Functions ---
stop_words = set(stopwords.words('english'))

def remove_pun(txt):
    return txt.translate(str.maketrans("", "", string.punctuation))

def remove_emojis(txt):
    return ''.join(char for char in txt if char.isascii())

def remove_stopwords(txt):
    words = txt.split()
    clean_words = [word for word in words if word not in stop_words]
    return ' '.join(clean_words)

# --- Model Training ---
print("Training the model...")

# 1. Load the dataset
dataset = pd.read_csv('SPAM text message 20170820 - Data.csv')

# 2. Map categories to numbers
dataset['Category'] = dataset['Category'].map({'ham': 0, 'spam': 1})

# 3. Apply all preprocessing steps to the 'Message' column
dataset['Message'] = dataset['Message'].apply(lambda x: x.lower())
dataset['Message'] = dataset['Message'].apply(remove_pun)
dataset['Message'] = dataset['Message'].apply(remove_emojis)
dataset['Message'] = dataset['Message'].apply(remove_stopwords)

# 4. Define the final pipeline with the best parameters found
# These parameters are based on your grid search results
final_pipeline = Pipeline([
    ('Tf_vectorizer', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
    ('smote', SMOTE(random_state=42)),
    ('lr', LogisticRegression(C=10, penalty='l2', random_state=42, solver='saga', max_iter=1000))
])

# 5. Train the final model on the ENTIRE dataset
X = dataset['Message']
y = dataset['Category']
final_pipeline.fit(X, y)

# 6. Save the trained pipeline to a file
joblib.dump(final_pipeline, 'spam_model.joblib')

print("Model trained and saved as spam_model.joblib")