import nltk
nltk.download(['stopwords', 'punkt'])


import pandas as pd

# Load IMDb dataset (or replace this with your own dataset)
df = pd.read_csv("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", compression="gzip")

# Preview the dataset
df.head()


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean the text
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove non-alphabetic characters (remove digits, punctuations, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply the function to clean the reviews
df['cleaned_review'] = df['review'].apply(clean_text)

# Check the cleaned data
df[['review', 'cleaned_review']].head()
