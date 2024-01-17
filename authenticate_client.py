
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def authenticate_client():
    # Replace the following values with your Azure Text Analytics API endpoint and key
    endpoint = "https://laboblogtextanalytics.cognitiveservices.azure.com/"
    key = "09b7c88cfff44bac95c83a2256f31907"

    # Create an AzureKeyCredential object using your API key
    credential = AzureKeyCredential(key)

    # Create a TextAnalyticsClient using the endpoint and credential
    text_analytics_client = TextAnalyticsClient(endpoint, credential)

    # Return the authenticated TextAnalyticsClient
    return text_analytics_client