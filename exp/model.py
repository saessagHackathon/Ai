from langchain_community.chat_models import ChatOpenAI
from prompt import question_answering_prompt

from langchain.chains.combine_documents import create_stuff_documents_chain
API_KEY = ''

llm = ChatOpenAI(
    model = 'gpt-4o',
    api_key=API_KEY
)


document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

