import pandas as pd
import json
from collections import Counter
from attacut import tokenize, Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_json("thai_sentiment_dataset.json")

# Tokenize all sentences
def tokenize_sentences(sentences):
    atta = Tokenizer(model="attacut-sc")
    return [atta.tokenize(sentence) for sentence in sentences]

df['tokens'] = tokenize_sentences(df.iloc[:, 0])

# List of positive and negative words
positive_words = ["สวยงาม", "ดี", "ยินดี", "อร่อย", "เชื่อถือได้", "สุข", "รัก", "ความสุข", "เยี่ยม", "สนุก", "อบอุ่น", "ดีใจ"]
negative_words = ["ข่มขืน", "ฆ่า", "ไม่", "ฆาตกรรม", "เสี่ยง", "มีผลกระทบ", "ปัญหา", "แย่", "เสียใจ", "โง่"]

# Count occurrences of positive and negative words
def count_word_occurrences(tokens, words):
    token_counts = Counter(tokens)
    return sum(token_counts[word] for word in words)

# Add columns for positive and negative word occurrences
df['positive_word_count'] = df['tokens'].apply(lambda tokens: count_word_occurrences(tokens, positive_words))
df['negative_word_count'] = df['tokens'].apply(lambda tokens: count_word_occurrences(tokens, negative_words))

# Prepare data for classification
x = df[['positive_word_count', 'negative_word_count']]
y = df['sentiment']

# Split data into train and test sets
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=7)

# Train Decision Tree model
model = DecisionTreeClassifier(criterion="gini")
model.fit(x_train, y_train)

# Evaluate model
predicted = model.predict(x_test)
accuracy = accuracy_score(y_test, predicted)
print("Accuracy:", accuracy)

df.to_csv('data_yaimakmak.csv', index=False)