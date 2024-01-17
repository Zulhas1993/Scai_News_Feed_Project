from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import json

# Import the get_news_details function from Scrap_Data_From_Json
from Scrap_Data_From_Json import get_news_details, extract_news_info

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-16') as file:
        data = json.load(file)
    return data

file_path = 'object_list.json'

# Read JSON file
data = read_json(file_path)

# Extract news information
news_list = extract_news_info(data)

# Now 'news_list' contains a list of dictionaries, each representing a news entry
# Iterate through each news entry to get news details
for news in news_list:
    link = news['link']
    details_news = get_news_details(link)
    
    # Now you can use 'details_news' as needed
    print(details_news)

def extract_keywords(details_news, languages=None):
    endpoint = "https://laboblogtextanalytics.cognitiveservices.azure.com/"
    key = "09b7c88cfff44bac95c83a2256f31907"

    # Initialize the Text Analytics client
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint, credential)

    # Detect language
    language_detection_response = text_analytics_client.detect_language([details_news])
    detected_languages = language_detection_response[0].detected_languages

    # Use detected language or default to English
    detected_language = detected_languages[0].iso6391_name if detected_languages else "en"

    # Call the Text Analytics API to analyze the document
    response = text_analytics_client.extract_key_phrases([details_news], language=detected_language)

    # Check for errors in the response
    if response.errors:
        # Handle errors
        print("Error in Text Analytics API response:", response.errors)
        return []

    # Extract keywords from the API response
    keywords = response[0].key_phrases

    # Limit the number of keywords (adjust as needed)
    max_keywords = 400
    keywords = keywords[:max_keywords]
    return keywords


def get_user_input():
    title = input("Enter your interest title: ")
    request = input("Enter your interest request or information: ")
    
    return title, request

def extract_User_Theme_keywords(text, text_analytics_client):
    # Call the Text Analytics API to analyze the document and extract key phrases
    response = text_analytics_client.extract_key_phrases([text])
    
    # Check for errors in the response
    if response.errors:
        print("Error in Text Analytics API response:", response.errors)
        return []
    
    # Extract keywords from the API response
    keywords = response[0].key_phrases
    
    return keywords



# Example usage:
# details_news = "This is a sample news article. It discusses various topics."
# keywords = extract_keywords(details_news)
# print("Extracted Keywords:", keywords)
