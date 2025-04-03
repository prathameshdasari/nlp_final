from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the BERT model using Hugging Face's pipeline
embedding_pipeline = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_embedding(text):
    """Generate BERT embeddings for a given text."""
    embedding = embedding_pipeline(text)
    return np.array(embedding).squeeze(0).mean(axis=0)  # Get sentence-level embedding

def prepare_faq_data(faq_data):
    """Preprocess FAQs and generate embeddings."""
    questions = []
    answers = []
    question_embeddings = []

    for category_data in faq_data:
        for faq in category_data.get("faqs", []):
            question_texts = faq.get("question", [])
            answer_text = faq.get("answer", "")

            if isinstance(question_texts, str):  # If it's a single string, convert to list
                question_texts = [question_texts]

            for question in question_texts:
                questions.append(question)
                answers.append(answer_text)
                question_embeddings.append(get_embedding(question))

    return questions, answers, np.array(question_embeddings)

def get_best_answer(user_query, questions, answers, question_embeddings):
    """Find the best-matching answer using cosine similarity."""
    query_embedding = get_embedding(user_query)
    similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return answers[best_match_idx]

# Example Usage
faq_data = [
    {"category": "Admissions", "faqs": [
        {"question": ["How can I apply?", "What is the admission process?"], "answer": "You can apply online via our portal."}
    ]}
]

questions, answers, question_embeddings = prepare_faq_data(faq_data)
user_query = "How do I enroll?"
best_answer = get_best_answer(user_query, questions, answers, question_embeddings)

print("User Query:", user_query)
print("Best Answer:", best_answer)
