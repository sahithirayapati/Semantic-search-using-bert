import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\hp\\Downloads\\news.csv")


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    if pd.isnull(text):
        return "" 
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)


df['processed_text'] = df['Book'].apply(preprocess_text)


user_query = input("Enter your query: ")


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


document_embeddings = model.encode(df['processed_text'].tolist(), convert_to_tensor=True)
user_query_embedding = model.encode(user_query, convert_to_tensor=True)


similarities = util.pytorch_cos_sim(user_query_embedding, document_embeddings).cpu().detach().numpy().flatten()

threshold = 0.5
query_similar_books = df[similarities > threshold].copy()

query_similar_books['BERT_similarity'] = similarities[similarities > threshold]
query_similar_books = query_similar_books.sort_values(by='BERT_similarity', ascending=False).head(10)


print("Books similar to the query term:")
print(query_similar_books[['Book', 'Author(s)', 'Genre', 'BERT_similarity']])


plt.figure(figsize=(3, 3))
plt.bar(query_similar_books['Book'], query_similar_books['BERT_similarity'], color='blue')
plt.xlabel('Books')
plt.ylabel('Similarity')
plt.title(' BERT')
plt.xticks(rotation=90) 
plt.ylim(0, 1) 
plt.show()
