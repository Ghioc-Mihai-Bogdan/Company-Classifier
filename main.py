import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Incarcare date
company_df = pd.read_csv("ml_insurance_challenge.csv")
taxonomy_df = pd.read_excel("insurance_taxonomy.xlsx")

# Preprocesare text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

# Combinare coloane relevante intr-un singur camp
def combine_company_text(row):
    components = [
        row.get("company_description", ""),
        row.get("business_tags", ""),
        row.get("sector", ""),
        row.get("category", ""),
        row.get("niche", "")
    ]
    components = [preprocess_text(str(comp)) for comp in components]
    return " ".join(components)

company_df["combined_text"] = company_df.apply(combine_company_text, axis=1)

# Pregatire text taxonomie
if "label_description" in taxonomy_df.columns:
    taxonomy_df["combined_text"] = taxonomy_df["label"].astype(str) + " " + taxonomy_df["label_description"].astype(str)
else:
    taxonomy_df["combined_text"] = taxonomy_df["label"].astype(str)

taxonomy_df["combined_text"] = taxonomy_df["combined_text"].apply(preprocess_text)

# TF-IDF vectorizare
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
all_text = pd.concat([company_df["combined_text"], taxonomy_df["combined_text"]])
vectorizer.fit(all_text)

company_vectors = vectorizer.transform(company_df["combined_text"])
taxonomy_vectors = vectorizer.transform(taxonomy_df["combined_text"])

# Calculare similaritate cosine
similarity_matrix = cosine_similarity(company_vectors, taxonomy_vectors)

# Atribuire etichete
SIMILARITY_THRESHOLD = 0.2  # Prag

def assign_labels(sim_scores, taxonomy_labels, threshold=SIMILARITY_THRESHOLD):
    indices = np.where(sim_scores >= threshold)[0]
    if len(indices) == 0:
        indices = [np.argmax(sim_scores)]
    return ", ".join(taxonomy_labels.iloc[indices].values)

company_df["insurance_label"] = [
    assign_labels(sim_scores, taxonomy_df["label"])
    for sim_scores in similarity_matrix
]

# Salvare rezultate
company_df.to_csv("ml_insurance_challenge_with_labels.csv", index=False)
print("Clasificare completa. Rezultate salvate in ml_insurance_challenge_with_labels.csv.")
