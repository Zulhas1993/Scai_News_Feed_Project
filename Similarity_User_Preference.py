import json
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from langid import classify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
import nltk
import string
from janome.tokenizer import Tokenizer
from Scrap_Data_From_Json import get_news_details

# Authenticate Text Analytics client
def authenticate_client():
    endpoint = "https://laboblogtextanalytics.cognitiveservices.azure.com/"
    key = "09b7c88cfff44bac95c83a2256f31907"
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# Preprocess text data
#The stopwords are common words in a language that are often filtered out from text data because they are considered to be of little value in terms of meaning. These words include articles, prepositions, conjunctions, and other common words that do not carry significant information about the content of the text.
#In natural language processing and text mining tasks, removing stopwords is a common preprocessing step. The purpose is to focus on the more meaningful words that contribute to the overall meaning of the text.
nltk.download('stopwords')
def preprocess_text(text, language='english'):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    #Punctuation characters include symbols like dot, periods, commas, exclamation marks, etc
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove stop words based on the language
    if language == 'english':
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    elif language == 'japanese':
        # Tokenize Japanese text
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        text = ' '.join([token.surface for token in tokens])

    # Add more preprocessing steps as needed

    return text

# Extract key phrases using Azure Text Analytics API
def extract_key_phrases(text, text_analytics_client):
    response = text_analytics_client.extract_key_phrases([text])
    if response.errors:
        print("Error in Text Analytics API response:", response.errors)
        return []
    return response[0].key_phrases

# Perform LDA topic modeling
#The perform_lda function in the code uses Latent Dirichlet Allocation (LDA),
# which is a generative probabilistic model commonly used for topic modeling.
def perform_lda(texts, num_topics=5):
   # CountVectorizer is a text vectorization technique that converts a collection of text documents to a matrix of token counts.
    #corresponds to a document, and each column corresponds to a unique word (token) in the corpus.
    #The values in the matrix represent the count of each word in the respective documents.
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    #The perform_lda function takes a list of texts (documents), converts them into a document-term matrix using CountVectorizer, and then applies LDA to identify latent topics in the collection of texts. The function returns the trained LDA model, which can later be used to extract topics from new documents or analyze the distribution of topics in the given set of texts

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    return lda

def perform_lda_multilingual(texts, num_topics=5):
    # Separate texts based on their identified language
    english_texts = [text for text in texts if classify(text)[0] == 'en']
    japanese_texts = [text for text in texts if classify(text)[0] == 'ja']

    # Apply CountVectorizer for English texts
    vectorizer_en = CountVectorizer(stop_words='english')
    X_en = vectorizer_en.fit_transform(english_texts)

    # Apply CountVectorizer for Japanese texts
    vectorizer_ja = CountVectorizer(stop_words='japanese')  # Replace with Japanese stop words
    X_ja = vectorizer_ja.fit_transform(japanese_texts)

    # Apply LatentDirichletAllocation for English texts
    lda_en = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_en.fit(X_en)

    # Apply LatentDirichletAllocation for Japanese texts
    lda_ja = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_ja.fit(X_ja)

    return lda_en, lda_ja





# Match topics and key phrases
def match_user_interest_with_news(user_title, user_request, article_urls, text_analytics_client):
    max_similarity = -1
    selected_url = None

    for article_url in article_urls:
        url_details = get_news_details(article_url)  # Implement the get_news_details function

        if url_details is None:
            continue  # Skip to the next URL if fetching details fails

        # Combine title, request, and description for user and news details
        user_text = f"{user_title} {user_request}"
        news_text = f"{url_details['title']} {url_details['news_details']}"

        # Extract key phrases
        user_key_phrases = extract_key_phrases(user_text, text_analytics_client)
        news_key_phrases = extract_key_phrases(news_text, text_analytics_client)

        # Perform LDA on user and news texts
        lda_user = perform_lda([user_text])
        lda_news = perform_lda([news_text])

        # Calculate similarity between user and news key phrases or topics
        key_phrases_similarity = calculate_similarity(user_key_phrases, news_key_phrases)
        topics_similarity = calculate_similarity(lda_user.components_, lda_news.components_)

        # Calculate an overall similarity score (you can customize this)
        overall_similarity = 0.8 * key_phrases_similarity + 0.6 * topics_similarity

        if overall_similarity > max_similarity:
            max_similarity = overall_similarity
            selected_url = article_url

    return selected_url

    #return key_phrases_similarity, topics_similarity

# Calculate similarity (example: cosine similarity)
def calculate_similarity(vec1, vec2):
    # Assuming vec1 and vec2 are numpy arrays representing vectors
    # Calculate cosine similarity using sklearn's cosine_similarity function
    # Cosine similarity measures the cosine of the angle between two non-zero vectors
    # It ranges from -1 (completely dissimilar) to 1 (completely similar)
    
    # Ensure both vectors are 2D arrays
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0, 0]

        # Implement your similarity calculation method (e.g., cosine similarity using numpy)
        # This is just a placeholder, you might need to implement a proper similarity metric
    # similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # return similarity

    return similarity


def calculate_similarity(vec1, vec2):
    # Your similarity calculation method (e.g., cosine similarity)

# User Profile Information
 user_profile = {
    "Occupation": "your_occupation",
    "Hobby": "your_hobby",
    "Gender": "your_gender",
    "Age": "your_age",
    "Country": "your_country",
    "Religion": "your_religion"
}

# News Details from URL (replace with actual data)
news_details_text = "news details text extracted from URL"

# Authenticate Text Analytics client
text_analytics_client = authenticate_client()

# Extract key phrases from user profile and news details
user_profile_key_phrases = extract_key_phrases(" ".join(user_profile.values()), text_analytics_client)
news_details_key_phrases = extract_key_phrases(news_details_text, text_analytics_client)

# Perform LDA on user profile and news details key phrases
lda_user_profile = perform_lda(user_profile_key_phrases)
lda_news_details = perform_lda(news_details_key_phrases)

# Calculate similarity
topics_similarity = calculate_similarity(lda_user_profile.components_, lda_news_details.components_)

# Print or use the similarity score as needed
print("Topics Similarity:", topics_similarity)






def calculate_similarity(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = words1.intersection(words2)
    similarity = len(intersection) / (len(words1) + len(words2) - len(intersection))
    return similarity

def match_and_Selected_return_url(user_title, user_request, article_urls, text_analytics_client):
    max_similarity = -1
    selected_url = None

    for article_url in article_urls:
        url_details = get_news_details(text_analytics_client, article_url)

        if url_details is None:
            continue  # Skip to the next URL if fetching details fails

        # Calculate similarity between user title, user request, and URL details
        title_similarity = calculate_similarity(user_title, url_details["title"])
        details_similarity = calculate_similarity(user_request, url_details["details"])
        overall_similarity = 0.8 * title_similarity + 0.6 * details_similarity

        if overall_similarity > max_similarity:
            max_similarity = overall_similarity
            selected_url = article_url

    return selected_url
if __name__ == "__main__":
    # Example usage
    user_input = "User's interest title, request, or information"
    news_details = "News details from the link"
    
    # Authenticate Text Analytics client
    text_analytics_client = authenticate_client()

    # Match user interest with news details
    key_phrases_similarity, topics_similarity = match_user_interest_with_news(
        user_input, news_details, text_analytics_client
    )


    # Provide recommendations based on the similarity scores
    print("Key Phrases Similarity:", key_phrases_similarity)
    print("Topics Similarity:", topics_similarity)
    
    # Based on the similarity scores, provide recommendations or further actions
