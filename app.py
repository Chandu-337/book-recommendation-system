
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
books = pd.read_csv('books.csv')
books.dropna(inplace=True)  # Remove rows with missing values
books.drop_duplicates(inplace=True)  # Remove duplicate rows

# Preprocess and prepare data
popular_books = books[books['ratings_count'] >= 1000]  # Adjust this threshold as necessary
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(popular_books['title'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_books(book_title, top_n=5):
    if book_title not in popular_books['title'].values:
        return "Book not found."

    # Reset the index to match with the similarity matrix
    popular_books_reset = popular_books.reset_index(drop=True)

    # Get the index of the book
    idx = popular_books_reset[popular_books_reset['title'] == book_title].index[0]

    # Check if the index is valid
    if idx >= similarity_matrix.shape[0]:
        return "Index out of bounds."

    # Calculate similarity scores
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in similarity_scores]

    # Return the recommended books
    return popular_books.iloc[book_indices][['title', 'authors', 'average_rating']]

# Streamlit UI
st.title("Book Recommendation System")
book_input = st.text_input("Enter Book Title")
if st.button("Recommend"):
    recommendations = recommend_books(book_input)
    st.write(recommendations)

    