import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv("spam_demo_dataset.py.csv", encoding='latin1')  # Your uploaded file

# Step 2: Explore the data
print(df.head())

# Step 3: Rename columns (if needed)
# Assume format: 'label', 'message' or similar
df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})

# Convert labels: ham → 0, spam → 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Train-Test Split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text to vectors (Bag of Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Prediction & Evaluation
y_pred = model.predict(X_test_vec)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
