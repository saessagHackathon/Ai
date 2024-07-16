from langchain_community.chat_models import ChatOpenAI
from prompt import burgerking_prompt, mega_coffee_prompt


API_KEY = ''

llm = ChatOpenAI(
    model = 'gpt-4o',
    api_key=API_KEY
)


