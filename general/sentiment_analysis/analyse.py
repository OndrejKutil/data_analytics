from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Correct label mapping used during training
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}

# Load tokenizer
model_path = './general/sentiment_analysis/model'
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load model with correct label mapping
model = BertForSequenceClassification.from_pretrained(
    model_path,
    id2label=id2label,
    label2id=label2id
)

# Create inference pipeline
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Prediction function
def predict_sentiment(text: str) -> dict:
    result = classifier(text)[0]
    return {
        "text": text,
        "sentiment": result['label'],  # Now returns 'positive', 'neutral', etc.
        "confidence": round(result['score'], 4)
    }

# Test
print(predict_sentiment("This is so good"))
print(predict_sentiment("This is so bad"))
print(predict_sentiment("This is ok"))
