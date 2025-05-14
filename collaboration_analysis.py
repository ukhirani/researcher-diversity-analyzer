import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def load_analysis_data():
    """Load data from the correct researcher analysis Excel file."""
    try:
        themes_df = pd.read_excel('researcher_analysis.xlsx', sheet_name='Author_Profiles')
        diversity_df = pd.read_excel('researcher_analysis.xlsx', sheet_name='Author_diversity')
        return themes_df, diversity_df
    except FileNotFoundError:
        print("Error: Could not find researcher_analysis_20250514_102153.xlsx. Please run the main analysis first.")
        return None, None

def extract_themes(text):
    """Extract individual themes from the formatted theme string."""
    if pd.isna(text) or text == "No data available":
        return []
    # Split by semicolon and then by comma
    theme_groups = text.split(';')
    themes = []
    for group in theme_groups:
        themes.extend([t.strip() for t in group.split(',')])
    return [t for t in themes if t]

def calculate_similarity_matrix(themes_df):
    """Calculate similarity between researchers based on their research themes."""
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create theme vectors for each researcher
    researcher_vectors = {}
    for _, row in themes_df.iterrows():
        researcher = row['Researcher']
        themes = extract_themes(row['Top Research Themes'])
        # Create a combined text of all themes
        theme_text = ' '.join(themes)
        # Get embedding for the combined themes
        vector = model.encode(theme_text)
        researcher_vectors[researcher] = vector
    
    # Calculate similarity matrix
    researchers = list(researcher_vectors.keys())
    similarity_matrix = np.zeros((len(researchers), len(researchers)))
    
    for i, res1 in enumerate(researchers):
        for j, res2 in enumerate(researchers):
            if i != j:
                similarity = cosine_similarity(
                    [researcher_vectors[res1]], 
                    [researcher_vectors[res2]]
                )[0][0]
                similarity_matrix[i][j] = similarity
    
    return researchers, similarity_matrix

def find_most_similar_pair(researchers, similarity_matrix):
    """Find the most similar researchers."""
    # Get the highest similarity pair
    max_sim = -1
    most_similar_pair = None
    
    for i in range(len(researchers)):
        for j in range(i+1, len(researchers)):
            if similarity_matrix[i][j] > max_sim:
                max_sim = similarity_matrix[i][j]
                most_similar_pair = (researchers[i], researchers[j], max_sim)
    
    return most_similar_pair

def suggest_collaborations(themes_df, diversity_df, researchers, similarity_matrix):
    """Suggest interdisciplinary collaboration opportunities."""
    # Prepare diversity lookup
    diversity_lookup = dict(zip(diversity_df['Researcher'], diversity_df['Diversity Score']))
    theme_lookup = dict(zip(themes_df['Researcher'], themes_df['Top Research Themes']))
    suggestions = []
    used_pairs = set()
    # 1. High-Low Diversity
    high = [r for r in researchers if diversity_lookup.get(r) == 'High Diversity']
    low = [r for r in researchers if diversity_lookup.get(r) == 'Low Diversity']
    if high and low:
        for h in high:
            for l in low:
                if h != l and (h, l) not in used_pairs and (l, h) not in used_pairs:
                    h_themes = set(extract_themes(theme_lookup[h]))
                    l_themes = set(extract_themes(theme_lookup[l]))
                    complementary = h_themes - l_themes
                    if complementary:
                        suggestions.append({
                            'Type': 'High-Low Diversity',
                            'Researchers': f'{h} & {l}',
                            'Justification': f"{h} (high diversity) brings breadth in {', '.join(list(complementary)[:3])}, complementing {l}'s focused expertise in {', '.join(list(l_themes)[:3])}."
                        })
                        used_pairs.add((h, l))
                        break
            if len(suggestions) >= 1:
                break
    # 2. Medium-Medium Diversity
    medium = [r for r in researchers if diversity_lookup.get(r) == 'Medium Diversity']
    if len(medium) >= 2:
        for i in range(len(medium)):
            for j in range(i+1, len(medium)):
                r1, r2 = medium[i], medium[j]
                if (r1, r2) not in used_pairs and (r2, r1) not in used_pairs:
                    t1 = set(extract_themes(theme_lookup[r1]))
                    t2 = set(extract_themes(theme_lookup[r2]))
                    overlap = t1 & t2
                    complementary = (t1 | t2) - overlap
                    if complementary:
                        suggestions.append({
                            'Type': 'Medium-Medium Diversity',
                            'Researchers': f'{r1} & {r2}',
                            'Justification': f"{r1} and {r2} share interest in {', '.join(list(overlap)[:2])} and can collaborate on complementary areas like {', '.join(list(complementary)[:3])}."
                        })
                        used_pairs.add((r1, r2))
                        break
            if len(suggestions) >= 2:
                break
    # 3. Most Interdisciplinary (lowest similarity, different diversity)
    min_sim = 2
    min_pair = None
    for i in range(len(researchers)):
        for j in range(i+1, len(researchers)):
            r1, r2 = researchers[i], researchers[j]
            if diversity_lookup.get(r1) != diversity_lookup.get(r2):
                if similarity_matrix[i][j] < min_sim:
                    min_sim = similarity_matrix[i][j]
                    min_pair = (r1, r2)
    if min_pair:
        r1, r2 = min_pair
        t1 = set(extract_themes(theme_lookup[r1]))
        t2 = set(extract_themes(theme_lookup[r2]))
        unique1 = t1 - t2
        unique2 = t2 - t1
        suggestions.append({
            'Type': 'Interdisciplinary (Low Similarity)',
            'Researchers': f'{r1} & {r2}',
            'Justification': f"{r1} and {r2} have the most distinct research themes ({', '.join(list(unique1)[:2])} vs {', '.join(list(unique2)[:2])}), offering a unique opportunity for cross-domain collaboration."
        })
    return suggestions[:3]

def main():
    # Load the analysis data
    themes_df, diversity_df = load_analysis_data()
    if themes_df is None or diversity_df is None:
        return
    
    # Find similar researchers
    researchers, similarity_matrix = calculate_similarity_matrix(themes_df)
    similar_pair = find_most_similar_pair(researchers, similarity_matrix)
    
    print("\n=== Most Similar Researchers ===")
    if similar_pair:
        res1, res2, similarity = similar_pair
        print(f"\n{res1} and {res2}")
        print(f"Similarity Score: {similarity:.2f}")
        print("\nShared Research Themes:")
        themes1 = set(extract_themes(themes_df[themes_df['Researcher'] == res1]['Top Research Themes'].iloc[0]))
        themes2 = set(extract_themes(themes_df[themes_df['Researcher'] == res2]['Top Research Themes'].iloc[0]))
        shared_themes = themes1 & themes2
        print(", ".join(shared_themes))
    
    # Suggest collaborations
    print("\n=== Suggested Collaboration Opportunities ===")
    collaborations = suggest_collaborations(themes_df, diversity_df, researchers, similarity_matrix)
    
    for i, collab in enumerate(collaborations, 1):
        print(f"\n{i}. {collab['Type']} Collaboration:")
        print(f"Researchers: {collab['Researchers']}")
        print(f"Justification: {collab['Justification']}")

    # Save to Excel
    out = pd.ExcelWriter('collaboration_opportunities.xlsx')
    pd.DataFrame([{
        'Researcher 1': res1,
        'Researcher 2': res2,
        'Similarity Score': similarity,
        'Shared Themes': ', '.join(shared_themes)
    }]).to_excel(out, sheet_name='Most Similar Researchers', index=False)
    pd.DataFrame(collaborations).to_excel(out, sheet_name='Collaboration Opportunities', index=False)
    out.close()
    print('Collaboration analysis saved to collaboration_opportunities.xlsx')

if __name__ == "_main_":
    main()