import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Load a transformer model for embeddings (using a lightweight model for speed)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def calculate_diversity_score(abstracts):
    """
    Calculate semantic similarity between all abstracts using sentence embeddings.
    Returns average similarity and a qualitative diversity label.
    """
    if len(abstracts) < 2:
        return 1.0, 'Low Diversity (Insufficient Data)'
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(abstracts)
    sim_matrix = cosine_similarity(embeddings)
    # Take upper triangle without diagonal
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    avg_similarity = np.mean(upper_tri)
    # Interpret diversity
    if avg_similarity > 0.75:
        diversity = 'Low Diversity (Focused)'
    elif avg_similarity > 0.55:
        diversity = 'Medium Diversity'
    else:
        diversity = 'High Diversity (Broad)'
    return avg_similarity, diversity


def extract_top_themes(abstracts, top_n=5):
    """
    Use TF-IDF to extract top research themes from combined abstracts.
    """
    combined_text = ' '.join(abstracts)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return list(top_keywords)


def generate_wordcloud(abstracts, researcher_name, out_dir='wordclouds'):
    """
    Generate and save a word cloud image from combined abstracts.
    """
    os.makedirs(out_dir, exist_ok=True)
    combined_text = ' '.join(abstracts)
    wc = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    img_path = os.path.join(out_dir, f"{researcher_name}_wordcloud.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    return img_path


if __name__ == "__main__":
    # Example usage: expects a DataFrame with columns ['S.No', 'Researcher Name', 'Title', 'Abstract'] for each researcher
    # You can integrate this into your workflow by loading each researcher's sheet as a DataFrame
    excel_path = 'researcher_analysis.xlsx'
    xl = pd.ExcelFile(excel_path)
    summary_profiles = []
    summary_diversity = []
    print("\n--- Sheet Columns Diagnostic ---")
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, nrows=1)  # Only load first row for speed
        print(f"Sheet: {sheet} | Columns: {list(df.columns)}")
    print("--- End Diagnostic ---\n")
    for sheet in xl.sheet_names:
        if sheet.lower() in ['author_profiles', 'author_diversity', 'summary']:
            continue
        df = xl.parse(sheet)
        # Accept both lower/upper case column names
        col_map = {c.lower(): c for c in df.columns}
        required_cols = ['abstract', 'researcher name']
        if not all(col in col_map for col in required_cols):
            print(f"[WARNING] Sheet '{sheet}' skipped: missing columns {required_cols}")
            continue
        abstracts = df[col_map['abstract']].dropna().astype(str).tolist()
        researcher_name = df[col_map['researcher name']].iloc[0] if not df.empty else sheet
        # Diversity score
        avg_sim, diversity_label = calculate_diversity_score(abstracts)
        # Top research themes
        top_themes = extract_top_themes(abstracts)
        # Word cloud
        wc_path = generate_wordcloud(abstracts, researcher_name)
        # Collect for summary
        summary_profiles.append({
            'Researcher': researcher_name,
            'Top Research Themes': ', '.join(top_themes),
            'Wordcloud Path': wc_path
        })
        summary_diversity.append({
            'Researcher': researcher_name,
            'Average Similarity': avg_sim,
            'Diversity Score': diversity_label
        })
        print(f"Processed {researcher_name}: Diversity={diversity_label}, Themes={top_themes}")
    # Write summary sheets
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        pd.DataFrame(summary_profiles).to_excel(writer, sheet_name='Author_Profiles', index=False)
        pd.DataFrame(summary_diversity).to_excel(writer, sheet_name='Author_diversity', index=False)
    print("Summary sheets written to Excel.")
