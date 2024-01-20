import os
from langchain.callbacks.manager import get_openai_callback
#from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "5e1835fa2e784d549bb1b2f6bd6ed69f"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://labo-azure-openai-swedencentral.openai.azure.com/"

def __call_chat_api(messages: list) -> AzureChatOpenAI:
    model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment="labo-azure-openai-gpt-4-turbo",
    )
    with get_openai_callback():
        return model.invoke(messages)

def extract_keywords_and_topics(content: str) -> list:
    # Use OpenAI to extract keywords and topics
    # Modify this part based on OpenAI's specific capabilities for keyword extraction
    # You may use OpenAI's gpt-3.5-turbo for prompt-based extraction or other methods

    # Example:
    request_messages = [
        AIMessage(content=content),
        HumanMessage(content="Extract key topics,information,user interest, user needs and key sentence based on information"),
    ]
    response = __call_chat_api(request_messages).content
    # Assuming response is a string
    extracted_data = str(response)
    return extracted_data.split('\n')  # Adjust this based on the actual output format

def find_most_relevant_details_feed(user_interest, user_information, details_feed_list):
    # Extract key topics and information from user interest and information
    user_interest_keywords = extract_keywords_and_topics(user_interest)
    user_information_keywords = extract_keywords_and_topics(user_information)

    # Combine user interest and information keywords
    combined_keywords = user_interest_keywords + user_information_keywords

    # Initialize variables to store the most relevant details feed
    max_relevance_score = 0
    most_relevant_feed = ""

    # Iterate through details_feed_list
    for feed in details_feed_list:
        relevance_scores = []
        # Iterate through sentences in the feed
        for sentence in feed:
            sentence_keywords = extract_keywords_and_topics(sentence)
            # Calculate relevance score based on the intersection of keywords
            relevance_score = len(set(combined_keywords) & set(sentence_keywords))
            relevance_scores.append(relevance_score)

        # Calculate the cumulative relevance score for the entire feed
        cumulative_relevance_score = sum(relevance_scores)

        # Update the most relevant details feed if the current one has a higher cumulative relevance score
        if cumulative_relevance_score > max_relevance_score:
            max_relevance_score = cumulative_relevance_score
            most_relevant_feed = feed

    return most_relevant_feed

def analysis_and_recommendation():
    request_messages = [
        SystemMessage(content="Please answer in English"),
    ]

    personal_information = "I am adnan. I am 29 years old. I am Muslim. I have completed my graduation from CUET in 2017. I have 4 members in my family. I live in Dhaka, Bangladesh. I am working for a private software company as a Senior Software Engineer."

    personal_interest = """
        ("Relaxation and adventure motivate me to travel, adventure and budget traveller, beaches and mountains are my travel destinations, travel agent will plan for my travel, planning for international travel",
        "Artificial Intelligence. Recent news for AI. AI documentations. AI machines. Microsoft AI services")
    """

    details_feed_list = [
         """
        ("In the intricate tapestry of international politics, a constant dance unfolds, where nations navigate alliances, conflicts, and global dynamics. Leaders engage in a diplomatic ballet, seeking equilibrium between cooperation and competition. Power dynamics shift like tectonic plates, shaping the geopolitical landscape. ",
        "Summits and negotiations become stages for both collaboration and contention, while issues of human rights, climate change, and security echo across borders. The United Nations, a global forum, attempts to harmonize diverse voices. ",
        "Yet, beneath the surface, tensions simmer, and strategic moves reshape the chessboard. The interplay of ideologies, economic interests, and cultural nuances defines this ever-evolving saga, illustrating the intricate interdependence of nations on the grand stage of international politics.")
        """,
        """
        ("Adnan, a 29-year-old Muslim residing in Dhaka, Bangladesh, graduated from CUET in 2017. He is a Senior Software Engineer at a private software company, and he has a family of four.",
        "Adnan finds motivation in relaxation and adventure, particularly enjoying budget travel to beaches and mountains. He relies on a travel agent for trip planning and is currently considering international travel",
        "Additionally, Adnan is deeply interested in Artificial Intelligence, keeping up with recent news, AI documentations, and Microsoft AI services.")
        """,
        """
        ("Zulhas, a 29-year-old Muslim residing in Dhaka, Bangladesh, completed his graduation from JNU in 2017. With a family of six, he embraces a strong sense of community. ",
        "Zulhas is not just academically inclined; he is also an avid cricketer, showcasing his passion for the sport. Beyond the cricket field, he finds joy in exploring the dynamic world of programming. ",
        "Zulhas's diverse interests reflect a well-rounded individual who appreciates both the exhilarating competitiveness of cricket and the intricate challenges of programming, creating a unique blend of athleticism and intellectual curiosity in his pursuits.")
        """,
      
        """
        ("In the recent chapters of Bangladeshi politics, the nation experienced a dynamic narrative marked by democratic strides and social transformations. Vibrant political dialogues unfolded as diverse voices sought representation. ",
        "The government's initiatives aimed at economic development garnered attention, yet challenges persisted. Elections became a canvas where citizens painted their aspirations, emphasizing transparency and accountability. Social movements echoed demands for justice and equality, reflecting a society in flux.",
        "Amidst the political mosaic, Bangladesh grappled with the dual forces of progress and tradition, navigating a path toward inclusive governance. The resilience of its people and the evolving political landscape hinted at a nation scripting its destiny, embracing both the lessons of history and the promise of a progressive future.")
        """
    ]

    request_messages.extend([
        HumanMessage(content=f"""
        If you are looking for information about a trip, please list 20 things you need
        """
        )
    ])

    # Make the API call
    response = __call_chat_api(request_messages)

    # Convert content_from_api to string
    content_from_api = response.content if isinstance(response, AIMessage) else str(response)
    content_str = str(content_from_api)

    request_messages.extend([
        AIMessage(content=f"{content_str}"),
        HumanMessage(content="""
                     make a questionnaire based on the above things with 4 options
                     """)
    ])

    # Make another API call with updated messages
    #response = __call_chat_api(request_messages).content
    #print(response)
    # Ensure response is a string
    #response_content = response.content if isinstance(response, AIMessage) else str(response)

    # Analyze user interest and information
    user_interest = """
        ("Relaxation and adventure motivate me to travel, adventure and budget traveller, beaches and mountains are my travel destinations, travel agent will plan for my travel, planning for international travel",
        "Artificial Intelligence. Recent news for AI. AI documentations. AI machines. Microsoft AI services")
    """
    user_information = "I am adnan. I am 29 years old. I am Muslim. I have completed my graduation from CUET in 2017."

    # Find the most relevant details_feeds
    relevant_feed = find_most_relevant_details_feed(user_interest, user_information, details_feed_list)
    print("Most Relevant Details_feed:")
    print(relevant_feed)

# Call the main function
analysis_and_recommendation()
