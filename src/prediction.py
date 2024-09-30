import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function for predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

# Example prediction
text_to_predict = "Example client interaction text"
print(predict(text_to_predict))
