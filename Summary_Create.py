import json
from bs4 import BeautifulSoup
from docx import Document 
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Import the get_news_details function from Scrap_Data_From_Json
from Scrap_Data_From_Json import get_news_details, get_ex_links, extract_news_info

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-16') as file:
        data = json.load(file)
    return data

def authenticate_client():
    endpoint = "https://laboblogtextanalytics.cognitiveservices.azure.com/"
    key = "09b7c88cfff44bac95c83a2256f31907"
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def generate_abstractive_summary(document, text_analytics_client):
    try:
        file_path = 'object_list.json'
        
        # Read JSON file
        data = read_json(file_path)
        
        # Extract news information
        news_list = extract_news_info(data)
        
        # Get links 
        get_ex_links(news_list, output_file='Ex_link.docx')
        
        summaries = []
        for news in news_list:
            link = news['link']
            try:
                # Fetch details for each link
                news_details = get_news_details(link)
                
                if news_details:
                    document_to_summarize = news_details.get('content', '')
                    
                    # Begin abstractive summarization
                    poller = text_analytics_client.begin_abstract_summary(documents=[{"id": "1", "text": document_to_summarize}])
                    abstract_summary_results = poller.result()

                    for result in abstract_summary_results:
                        if result.kind == "AbstractiveSummarization":
                            summaries.append([summary.text for summary in result.summaries])
                        elif result.is_error is True:
                            print(f"Error with code '{result.error.code}' and message '{result.error.message}'")
                            # Print the full error details for more information
                            print(result.error)
                            return None
            except Exception as e:
                print(f"An error occurred while processing news details: {e}")

        return summaries

    except Exception as e:
        print(f"An error occurred during abstractive summarization: {e}")
        return None

if __name__ == "__main__":
    text_analytics_client = authenticate_client()
    document = "Your document goes here."  # Replace with your actual document
    summaries = generate_abstractive_summary(document, text_analytics_client)

    if summaries:
        print("Summaries abstracted from news details:")
        [print(summary) for summary in summaries]
    else:
        print("Abstractive summarization failed.")
