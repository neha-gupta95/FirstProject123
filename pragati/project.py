import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and clean data
df = pd.read_csv("movies.csv")
df.columns = df.columns.str.strip()  # remove column name spaces

# Step 2: Fill missing values
for feature in ['Genre', 'Director', 'Actors']:
    df[feature] = df[feature].fillna('')

# Step 3: Create combined genre-based vectors
cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(df['Genre'])
similarity = cosine_similarity(vectors)

# Step 4: Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    df['movie_lower'] = df['movie'].str.lower().str.strip()

    if movie_name not in df['movie_lower'].values:
        print(f" Movie '{movie_name}' not found.")
        print("\n Available movies:")
        print(df['movie'].tolist())
        return

    index = df[df['movie_lower'] == movie_name].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\n Movies similar to '{movie_name.title()}':")
    for i in movie_list:
        print("", df.iloc[i[0]]['movie'])

# Step 5: Example call
recommend("taare zameen par")