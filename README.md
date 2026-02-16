### EX6 Information Retrieval Using Vector Space Model in Python
### DATE:16/02/2026 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from tabulate import tabulate

nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)

preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Vectorizers
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# Term Frequency Table
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(row) for i, row in enumerate(tf_table)], headers="firstrow", tablefmt="grid"))

# Document Frequency (DF) and IDF Table
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(df_idf_table, headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"], tablefmt="grid"))

# TF-IDF Table
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(map(lambda x: round(x, 4), row)) for i, row in enumerate(tfidf_table)], headers="firstrow", tablefmt="grid"))

# Manual Cosine Similarity calculation
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0.0
    return dot_product, norm_vec1, norm_vec2, similarity

# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]
    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]
        dot, norm_q, norm_d, sim = cosine_similarity_manual(query_vector, doc_vector)
        results.append([doc_id, doc_text, round(dot, 4), round(norm_q, 4), round(norm_d, 4), round(sim, 4)])
    
    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# Input from user
query = input("\nEnter your query: ")

# Perform search
results_table, query_vector = search(query, tfidf_matrix, tfidf_vectorizer)

# Display Cosine Similarity Table
print("\n--- Search Results and Cosine Similarity ---\n")
headers = ["Doc ID", "Document", "Dot Product", "Query Magnitude", "Doc Magnitude", "Cosine Similarity"]
print(tabulate(results_table, headers=headers, tablefmt="grid"))
# Display Query TF-IDF Weights
print("\n--- Query TF-IDF Weights ---\n")
query_weights = [(terms[i], round(query_vector[i], 4)) for i in range(len(terms)) if query_vector[i] > 0]
print(tabulate(query_weights, headers=["Term", "Query TF-IDF Weight"], tablefmt="grid"))

# Display Ranking
print("\n--- Ranked Documents ---\n")
ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([idx, res[0], res[1], res[5]])

print(tabulate(ranked_docs, headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"], tablefmt="grid"))
# Find the document with the highest cosine similarity
highest_doc = max(results_table, key=lambda x: x[5])  # x[5] is the cosine similarity
highest_doc_id = highest_doc[0]
highest_doc_text = highest_doc[1]
highest_score = highest_doc[5]

print(f"\nThe highest rank cosine score is: {highest_score} (Document ID: {highest_doc_id})")
###### Sample documents stored in a dictionary
    documents = {
        "doc1": "This is the first document.",
        "doc2": "This document is the second document.",
        "doc3": "And this is the third one.",
        "doc4": "Is this the first document?",
    }

###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in               string.punctuation]
        return " ".join(tokens)

###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

###### Construct TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

###### Calculate cosine similarity between query and documents
    def search(query, tfidf_matrix, tfidf_vectorizer):
        //TYPE YOUR CODE HERE

###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = search(query, tfidf_matrix, tfidf_vectorizer)

###### Display search results
    print("Query:", query)
    for i, result in enumerate(search_results, start=1):
        print(f"\nRank: {i}")
        print("Document ID:", result[0])
        print("Document:", result[1])
        print("Similarity Score:", result[2])
        print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("The highest rank cosine score is:", highest_rank_score)
```

### Output:

<img width="1870" height="801" alt="Screenshot 2025-09-26 114040" src="https://github.com/user-attachments/assets/db2cb0f7-70ae-4d83-91c0-d6359b031f81" />


<img width="1885" height="800" alt="Screenshot 2025-09-26 114059" src="https://github.com/user-attachments/assets/33aa1810-a871-4f01-b690-78fadffc78e7" />


<img width="1442" height="602" alt="Screenshot 2025-09-26 114111" src="https://github.com/user-attachments/assets/8f044d17-e61f-48d0-a988-5171bdb9c812" />


<img width="1627" height="580" alt="Screenshot 2025-09-26 114124" src="https://github.com/user-attachments/assets/6c6f9d52-fd24-4d98-bc87-da60f7a9f673" />


### Result:
implemented Information Retrieval Using Vector Space Model in Python.
