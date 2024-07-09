from langchain.memory import ChatMessageHistory

import re
import json
from model import llm
from prompt import question_answering_prompt
from retriever import retriever
from model import document_chain





demo_ephemeral_chat_history = ChatMessageHistory()


user = '안녕하세요 무엇을 도와드릴까요?'
print(user)

while user != 'exit':
    user = input()
    demo_ephemeral_chat_history.add_user_message(user)
    docs = retriever.invoke(user)
    content = document_chain.invoke(
        {
            "messages": demo_ephemeral_chat_history.messages,
            "context": docs,
        }
    )
    demo_ephemeral_chat_history.add_ai_message(content)
    
    if 'json' not in content:
        print(content)
    else:
        json_str = re.search(r'```json\n(.*)```', content, re.DOTALL).group(1)
        print('주문이 완료되었습니다!')
        order_dict = json.loads(json_str)
        break
