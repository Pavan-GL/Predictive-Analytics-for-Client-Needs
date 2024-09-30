import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('D:\Predictive Analytics for Client Needs\data\client_interactions.csv')

# Clean and preprocess text data
data['cleaned_text'] = data['text'].str.lower().str.replace('[^a-z\s]', '')

# Encode labels to numerical format
label_encoder = LabelEncoder()
data['encoded_labels'] = label_encoder.fit_transform(data['labels'])

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(list(data['cleaned_text']), padding=True, truncation=True, return_tensors="pt")

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    inputs, 
    data['encoded_labels'],  # Use encoded labels
    test_size=0.2,
    random_state=42
)
