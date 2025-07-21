import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("movies.csv")
df.columns=df.columns.str.strip()
for feature in['Genre','Director','Actors']:
 df['feature']=df['feature'].fillna('')
cv=CountVectorizer(stop_words='english')
vectors=cv.fit_transform(df['Genre'])
similarity=cosine_similarity(vectors)
def recommend(movie_name):
    movie_name=movie_name.lower().strip()
    if movie_name not in df['Tare Zameen Per'].str.lower().values:
        print(f"Movie '{movie_name}'not found.")
        return
    index=df[df['Tare Zameen Per'].str.lower()==movie_name].index[0]
    distance=similarity[index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    print(f"Movies similar to'{movie_name}':")
    for i in movie_list:
        print(df.iloc[i[0]]['movie'])
# example 
recommend('Tare Zameen Per')