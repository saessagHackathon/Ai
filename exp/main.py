from langchain.memory import ChatMessageHistory
import re
import json
from model import llm
from shop_data import shop_data
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from retriever import make_retriever
import prompt

def convert_to_json(content):
    text_split =  content[8:].replace('`', "").strip()
    text_json = json.loads(text_split)
    
    return text_json

# 가게 이름 불러오기 
shop_name = 'burgerking'

# prompt
shop_prompt = shop_data[shop_name]['prompt']

# db
shop_db = shop_data[shop_name]['db']

# 메세지 대화 저장
demo_ephemeral_chat_history = ChatMessageHistory()

# 모델 정의 
document_chain = create_stuff_documents_chain(llm, shop_prompt)

# retriever 정의
loader = TextLoader(shop_db)
data = loader.load()
rag = make_retriever(data)

# 메신저 대화 시작
user = '안녕하세요 무엇을 도와드릴까요?'
print(user)

# 주문이 완료되기 전까지 챗봇 활성화
while user != 'exit':
    
    # 유저 입력
    user = input()
    
    # 유저 메세지 저장
    demo_ephemeral_chat_history.add_user_message(user)
    
    # RAG docs 참조
    docs = rag.invoke(user)
    
    # AI 메세지 생성
    content = document_chain.invoke(
        {
            "messages": demo_ephemeral_chat_history.messages,
            "context": docs
        }
    )
    
    # AI 메세지 저장
    demo_ephemeral_chat_history.add_ai_message(content)
    
    # str > json    
    text = convert_to_json(content)    
    
    # AI 메세지 출력
    print(text['content'])
    
    # 주문 완료시 주문 완료 메세지와 함께 채팅 종료
    if text['order'] == 'complete':
        print('주문이 완료되었습니다. 즐거운 시간되세요!')
        break
    
