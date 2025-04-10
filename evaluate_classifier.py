# evaluate_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Incarca rezultatele cu etichete
def load_classification_results(filename="ml_insurance_challenge_with_labels.csv"):
    return pd.read_csv(filename)

# Analizeaza distributia etichetelor
def analyze_label_distribution(df):
    all_labels = []
    for labels in df['insurance_label']:
        labels_list = [label.strip() for label in labels.split(",")]
        all_labels.extend(labels_list)
    label_counts = Counter(all_labels)
    print("Distributia etichetelor:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
    plt.xticks(rotation=45)
    plt.title("Frecventa etichetelor")
    plt.xlabel("Eticheta")
    plt.ylabel("Frecventa")
    plt.tight_layout()
    plt.show()

# Analizeaza scorurile de similaritate
def evaluate_similarity_scores(df, similarity_scores_file="similarity_scores.npy"):
    try:
        sim_scores = np.load(similarity_scores_file)
        max_scores = sim_scores.max(axis=1)
        print("Statistici scoruri maxime:")
        print(f"Medie: {np.mean(max_scores):.4f}")
        print(f"Mediana: {np.median(max_scores):.4f}")
        print(f"Std: {np.std(max_scores):.4f}")
        plt.figure(figsize=(10, 6))
        plt.hist(max_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title("Scoruri maxime de similaritate")
        plt.xlabel("Scor")
        plt.ylabel("Numar companii")
        plt.show()
    except FileNotFoundError:
        print(f"Fisierul {similarity_scores_file} nu a fost gasit.")

# Revizuire manuala (eșantion)
def manual_sample_review(df, sample_size=10):
    sample_df = df.sample(n=sample_size, random_state=42)
    for _, row in sample_df.iterrows():
        print("------------------------------------------------------")
        print(f"Nume companie: {row.get('company_name', 'Necunoscut')}")
        print(f"Descriere: {row.get('company_description', 'N/A')}")
        print(f"Etichete: {row.get('insurance_label')}")
        print("------------------------------------------------------\n")

# Evaluare optionala cu ground truth (daca exista)
def optional_ground_truth_evaluation(df, ground_truth_file="ground_truth.csv"):
    try:
        gt_df = pd.read_csv(ground_truth_file)
        merged_df = pd.merge(df, gt_df, on="company_name", how="inner")
        if merged_df.empty:
            print("Nu s-au gasit potriviri.")
            return
        y_true = merged_df["true_label"].apply(lambda x: x.split(",")[0].strip())
        y_pred = merged_df["insurance_label"].apply(lambda x: x.split(",")[0].strip())
        # Decomentati pentru scoruri clasice
        # from sklearn.metrics import accuracy_score, f1_score
        # acc = accuracy_score(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred, average="weighted")
        # print(f"Acuratete: {acc:.4f}")
        # print(f"F1: {f1:.4f}")
        print("Evaluare pe ground truth (activati scorurile daca e cazul)")
    except FileNotFoundError:
        print(f"Fisierul {ground_truth_file} nu a fost gasit.")

# Rulare principala
def main():
    df = load_classification_results()
    analyze_label_distribution(df)
    evaluate_similarity_scores(df)
    print("\nRevizuire manuala:")
    manual_sample_review(df, sample_size=5)
    optional_ground_truth_evaluation(df)

if __name__ == "__main__":
    main()
