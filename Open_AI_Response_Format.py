import os
from langchain.callbacks.manager import get_openai_callback
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Set up Azure OpenAI API credentials
os.environ["AZURE_OPENAI_API_KEY"] = "5e1835fa2e784d549bb1b2f6bd6ed69f"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://labo-azure-openai-swedencentral.openai.azure.com/"

def __call_chat_api(messages: list) -> AzureChatOpenAI:
    # Initialize AzureChatOpenAI model
    model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment="labo-azure-openai-gpt-4-turbo",
    )
    
    # Use the get_openai_callback context manager
    with get_openai_callback():
        return model(messages)

def format_questionnaire_json(questionnaire_content: str) -> dict:
    if isinstance(questionnaire_content, str):
        # Split the questionnaire content into lines and filter out empty lines
        questions_and_options = [line.strip() for line in questionnaire_content.split('\n\n') if line.strip()]
        formatted_data = {"questions": []}

        for question_and_options in questions_and_options[1:]:
            # Skip the first line as it contains a sample introduction
            question_lines = question_and_options.strip().split('\n', 1)
            if len(question_lines) == 2:
                question_number, question_text = question_lines[0].split('.', 1)
                options = [line.strip()[3:] for line in question_lines[1].split('\n')]
                if options:
                    formatted_data["questions"].append({
                        "question": question_text.strip(),
                        "options": options
                    })
            else:
                # Handle the case where there is no dot ('.') on the first line
                question_text = question_lines[0]
                formatted_data["questions"].append({
                    "question": question_text.strip(),
                    "options": []
                })
        return formatted_data
    else:
        raise ValueError("Invalid content type. Expected string.")

def extract_keywords(text, languages=None):
    endpoint = "https://laboblogtextanalytics.cognitiveservices.azure.com/"
    key = "09b7c88cfff44bac95c83a2256f31907"

    # Initialize the Text Analytics client
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint, credential)
    # Detect language
    language_detection_response = text_analytics_client.detect_language([text])
    
    # Get the detected language
    detected_language = language_detection_response[0].primary_language.iso6391_name

    # Call the Text Analytics API to analyze the document
    response = text_analytics_client.extract_key_phrases([text], language=detected_language)

    # Check for errors in each element of the response list
    for result in response:
        if result.is_error:
            # Handle errors
            print("Error in Text Analytics API response:", result.error)
            return []

    # Extract keywords from the API response
    keywords = response[0].key_phrases
    # Limit the number of keywords (adjust as needed)
    max_keywords = 4000
    keywords = keywords[:max_keywords]
    return keywords

def analysis_and_recommendation():
    # Initial request messages
    request_messages = [SystemMessage(content="Please answer in English")]

    # User's personal information
    personal_information = "I am adnan. I am 29 years old. I am Muslim. I have completed my graduation from CUET in 2017. I have 4 members in my family. I live in Dhaka, Bangladesh. I am working for a private software company as a Senior Software Engineer."

    # User's personal interests
    personal_interest = """
        ("Relaxation and adventure motivate me to travel, adventure and budget traveller, beaches and mountains are my travel destinations, travel agent will plan for my travel, planning for international travel",
        "Artificial Intelligence. Recent news for AI. AI documentations. AI machines. Microsoft AI services")
    """

    # Feed containing details
    details_feed = """
        ("Historic books are portals to bygone eras, preserving the collective wisdom and tales of civilizations past. Through the weathered pages, one glimpses the evolution of ideas, cultures, and societies. These literary treasures bridge the gap between epochs, offering timeless insights that illuminate the rich tapestry of human history.",
        "Mobile phones have evolved from communication tools to indispensable companions. These pocket-sized marvels connect the world, offering instant communication, entertainment, and productivity. With sleek designs and powerful capabilities, they redefine daily life, providing access to information and connecting people across the globe in the palm of our hands.",
        "Web development is the art of crafting digital experiences, seamlessly blending creativity and technology. It encompasses designing and coding websites, ensuring functionality and aesthetic appeal. From front-end interfaces to back-end databases, web developers navigate the ever-evolving landscape of programming languages and frameworks, shaping the online world's dynamic presence.")
    """

    # User's request for information about a trip
    request_messages.extend([
        HumanMessage(content=f"If you are looking for information about a trip, please list 20 things you need")
    ])

    # Make the API call to get a response for trip information
    response = __call_chat_api(request_messages)

    # Convert content_from_api to string
    content_from_api = response.content if isinstance(response, AIMessage) else str(response)
    content_str = str(content_from_api)

    # Extend the request_messages with the response and a new request for a questionnaire
    request_messages.extend([
        AIMessage(content=f"{content_str}"),
        HumanMessage(content="Make a questionnaire based on the above things with 4 options")
    ])

    # Make another API call with updated messages to get a questionnaire
    response = __call_chat_api(request_messages).content

    # Ensure response is a string
    response_content = response.content if isinstance(response, AIMessage) else str(response)

    # Format the questionnaire JSON
    formatted_response = format_questionnaire_json(response_content)

    # Remove questions with empty options
    formatted_response["questions"] = [q for q in formatted_response["questions"] if q["options"]]

    # Modify the formatted response to match the expected result format
    modified_response = {
        "questions": formatted_response["questions"]
    }

    # Extract questions and options
    questions_and_options_list = []
    for question_data in modified_response["questions"]:
        question_text = question_data["question"]
        options = question_data["options"]

        # Store the question and options in a dictionary
        question_dict = {
            "question": question_text,
            "options": options
        }

        # Append the dictionary to the list
        questions_and_options_list.append(question_dict)

    # Print or save the extracted data
    #print("Extracted Data:")
    #print(questions_and_options_list)

    # Generate a summary article based on personal information and interests
    summary_article = generate_summary_article(personal_information, personal_interest)
    # print("\nSummary Article:")
    # print(summary_article)
    print(modified_response)
    
# Function to generate a summary article
def generate_summary_article(personal_information, personal_interest):
    # Create a summary article based on personal information and interests
    summary_article = f"""
    **Personal Information:**
    {personal_information}

    **Personal Interests:**
    {personal_interest}
    """

    return summary_article

# Call the main function
analysis_and_recommendation()
