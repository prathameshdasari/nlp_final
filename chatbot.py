import json
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ dataset
def load_faq():
    with open("faq.json", "r") as file:
        data = json.load(file)
    return data




# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    # text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    # print('preprocessed text : ',text,end="")  # Remove punctuation
    return text

# Create FAQ knowledge base
def prepare_faq_data(faq_data):
    questions = []
    answers = []

    for category_data in faq_data:  # Iterate over list
        faqs = category_data.get("faqs", [])  
        for faq in faqs:
            # questions.extend(faq.get("question", []))
            q = faq.get("question", [])
            if isinstance(q, list):  
                questions.extend(q)  # Extend if it's a list
            else:
                questions.append(q) 
            answers.append(faq.get("answer", ""))
    # print("Questions: ",questions)
    return questions, answers


# Find the best answer using TF-IDF & Cosine Similarity
def get_best_answer(user_queryy, questions, answers):
    user_query = preprocess_text(user_queryy)
    
    # Preprocess all FAQ questions
    preprocessed_questions = [preprocess_text(q) for q in questions]

    # Combine preprocessed FAQ questions with user query for vectorization
    all_texts = preprocessed_questions + [user_query]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # print("Similarity Scores:", similarity_scores)  # Debugging line

    best_match_idx = np.argmax(similarity_scores)
    best_score = similarity_scores[0, best_match_idx]

    # print(f"Best Match Index: {best_match_idx}, Score: {best_score}")

    # Threshold to decide if a match is relevant
    if best_score > 0.2:  
        return answers[best_match_idx]
    else:
        return "I'm sorry, I couldn't find an answer. Please contact the admissions office."

# Load FAQ dataset
faq_data = load_faq()

questions, answers = prepare_faq_data(faq_data)

# Function to get chatbot response
def chatbot_response(user_query):
    return get_best_answer(user_query, questions, answers)

# Test the chatbot
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)
