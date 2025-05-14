import pandas as pd
from scholarly import scholarly
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import os
import random
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def extract_user_id(url):
    """Extract user ID from Google Scholar profile URL."""
    match = re.search(r'user=([^&]+)', url)
    return match.group(1) if match else None

def get_researcher_publications(profile_url):
    """Fetch publications for a researcher from Google Scholar using their profile URL."""
    try:
        # Add random delay between requests
        time.sleep(random.uniform(2, 5))
        
        # Extract user ID from URL
        user_id = extract_user_id(profile_url)
        if not user_id:
            print(f"Could not extract user ID from URL: {profile_url}")
            return []
        
        # Get author profile using user ID
        author = scholarly.search_author_id(user_id)
        if not author:
            print(f"Could not find author with ID: {user_id}")
            return []
        
        # Get detailed author info
        author = scholarly.fill(author)
        
        # Get publications
        publications = []
        for pub in author['publications'][:20]:  # Get top 20 publications
            try:
                pub = scholarly.fill(pub)
                publications.append({
                    'title': pub['bib'].get('title', ''),
                    'abstract': pub['bib'].get('abstract', ''),
                    'year': pub['bib'].get('pub_year', ''),
                    'citations': pub['num_citations'],
                    'venue': pub['bib'].get('venue', ''),
                    'journal': pub['bib'].get('journal', '')
                })
                time.sleep(random.uniform(1, 3))  # Random delay between publications
            except Exception as e:
                print(f"Error fetching publication: {str(e)}")
                continue
        
        return publications, author['name']
    except Exception as e:
        print(f"Error fetching data from URL {profile_url}: {str(e)}")
        return [], None

def analyze_research_themes(abstracts):
    """Analyze research themes using TF-IDF."""
    if not abstracts:
        return []
        
    # Combine all abstracts
    combined_text = ' '.join([abs for abs in abstracts if abs])
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    
    # Get top keywords
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = [feature_names[i] for i in tfidf_matrix.toarray()[0].argsort()[::-1][:10]]
    
    return top_keywords

def calculate_diversity_score(abstracts):
    """Calculate semantic similarity between abstracts."""
    if not abstracts:
        return 0, "No Data"
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings for all abstracts
    embeddings = model.encode(abstracts)
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(similarity)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities) if similarities else 0
    
    # Determine diversity level
    if avg_similarity > 0.8:
        diversity = "Low Diversity"
    elif avg_similarity > 0.5:
        diversity = "Medium Diversity"
    else:
        diversity = "High Diversity"
    
    return avg_similarity, diversity



def main():
    # List of Google Scholar profile URLs
    profile_urls = [
    "https://scholar.google.com/citations?user=nUlanA8AAAAJ",
    "https://scholar.google.com/citations?user=zeET7_QAAAAJ",
    "https://scholar.google.co.in/citations?user=eAM84HQAAAAJ",
]
    
    output_file = 'researcher_analysis_2.xlsx'
    
    # Initialize summary DataFrame
    summary_df = pd.DataFrame(columns=['Researcher', 'Profile URL', 'Average Similarity', 'Diversity Score', 'Top Research Themes', 'Number of Publications', 'Total Citations'])
    
    # Create Excel writer
    with pd.ExcelWriter(output_file) as writer:
        # Process each researcher
        for profile_url in profile_urls:
            print(f"Processing profile: {profile_url}")
            
            # Get publications
            publications, researcher_name = get_researcher_publications(profile_url)
            
            if not publications or not researcher_name:
                print(f"No publications found for profile: {profile_url}")
                continue
            
            print(f"Found {len(publications)} publications for {researcher_name}")
            
            # Create DataFrame for publications
            pub_df = pd.DataFrame(publications)
            pub_df.insert(0, 'S.No', range(1, len(pub_df) + 1))
            pub_df.insert(1, 'Researcher Name', researcher_name)
            
            # Save to Excel sheet
            sheet_name = researcher_name # Excel sheet names limited to 31 chars
            pub_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Analyze research themes
            abstracts = [pub['abstract'] for pub in publications if pub['abstract']]
            if abstracts:
                                
                # Calculate diversity score
                avg_similarity, diversity = calculate_diversity_score(abstracts)
                
                # Get top research themes
                top_themes = analyze_research_themes(abstracts)
                
                # Add analysis results to summary DataFrame
                summary_data = {
                    'Researcher': [researcher_name],
                    'Profile URL': [profile_url],
                    'Average Similarity': [avg_similarity],
                    'Diversity Score': [diversity],
                    'Top Research Themes': [', '.join(top_themes)],
                    'Number of Publications': [len(publications)],
                    'Total Citations': [sum(pub['citations'] for pub in publications)]
                }
                summary_df = pd.concat([summary_df, pd.DataFrame(summary_data)], ignore_index=True)
            
            print(f"Completed processing {researcher_name}")
        
        # Write summary DataFrame to Excel
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

if __name__ == "__main__":
    main() 