from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def pick_llm(level: str):
    """
    Pick an LLM based on the specified level.
    Args:
        level(str): The level of the LLM to pick, can be "low", "medium", or "high".

    Returns:
        str: The name of selected LLM based on the specified level.
    """ 
    if level.lower() == 'low':
        llm = ChatOpenAI(model='gpt-5.6-luna')
    elif level.lower() == 'medium':
        llm = ChatOpenAI(model='gpt-5.6-terra')
    elif level.lower() == 'high':
        llm = ChatOpenAI(model='gpt-5.6-sol')
    else:
        raise ValueError("Invalid level. Please choose 'low', 'medium', or 'high'.")

    return llm