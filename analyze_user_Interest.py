import os
from langchain.callbacks.manager import get_openai_callback
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["AZURE_OPENAI_API_KEY"] = "5e1835fa2e784d549bb1b2f6bd6ed69f"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://labo-azure-openai-swedencentral.openai.azure.com/"

def __call_chat_api(messages: list) -> AzureChatOpenAI:
    model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment="labo-azure-openai-gpt-4-turbo",
    )
    with get_openai_callback():
        return model(messages)

def format_questionnaire_json(questionnaire_content: str) -> dict:
    if isinstance(questionnaire_content, str):
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



def analyze_user_content(user_content):
    request_messages = [
        SystemMessage(content="Please analyze the following content"),
        HumanMessage(content=user_content),
    ]

    # Make the API call
    response = __call_chat_api(request_messages)

    # Convert content_from_api to string
    content_from_api = response.content if isinstance(response, AIMessage) else str(response)
    return content_from_api

def analyze_user_content(user_content, num_sentences=3):
    request_messages = [
        SystemMessage(content=f"Please summarize the following content in {num_sentences} sentences"),
        HumanMessage(content=user_content),
    ]

    # Make the API call
    response = __call_chat_api(request_messages)

    # Convert content_from_api to string
    content_from_api = response.content if isinstance(response, AIMessage) else str(response)

    # Split the content into sentences and return the first 'num_sentences'
    return content_from_api.split('. ')[:num_sentences]


def analysis_and_recommendation():
    personal_information = "I am adnan. I am 29 years old. I am Muslim. I have completed my graduation from CUET in 2017. I have 4 members in my family. I live in Dhaka, Bangladesh. I am working for a private software company as a Senior Software Engineer."

    personal_interest = """
        ("Relaxation and adventure motivate me to travel, adventure and budget traveller, beaches and mountains are my travel destinations, travel agent will plan for my travel, planning for international travel",
        "Artificial Intelligence. Recent news for AI. AI documentations. AI machines. Microsoft AI services")
    """

    # Analyze personal_information
    analyzed_information = analyze_user_content(personal_information)
    print("Analyzed Personal Information:", analyzed_information)

    # Analyze personal_interest
    analyzed_interest = analyze_user_content(personal_interest)
    print("Analyzed Personal Interest:", analyzed_interest)

    # Analyze personal_information
    analyzed_information = analyze_user_content(personal_information, num_sentences=3)
    print("Analyzed Personal Information:", analyzed_information)

    # Analyze personal_interest
    analyzed_interest = analyze_user_content(personal_interest, num_sentences=3)
    print("Analyzed Personal Interest:", analyzed_interest)
    
# Call the main function
analysis_and_recommendation()
