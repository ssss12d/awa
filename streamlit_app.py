import openai
import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import pandas as pd
from typing import List, Dict
import json
import os
from datetime import datetime
from urllib.parse import urlparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedLearningAI:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.messages = []
        self.knowledge_base = self.load_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.update_vectors()
    
    def load_knowledge_base(self) -> dict:
        """加载知识库"""
        try:
            with open('advanced_knowledge_base.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "web": {},
                "file": {},
                "manual": {}
            }
    
    def save_knowledge_base(self):
        """保存知识库"""
        with open('advanced_knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        self.update_vectors()
    
    def update_vectors(self):
        """更新知识向量"""
        if not any(self.knowledge_base.values()):
            return
        
        all_content = []
        for category in self.knowledge_base.values():
            for item in category.values():
                all_content.append(item['content'])
        
        if all_content:
            self.vectors = self.vectorizer.fit_transform(all_content)
    
    def learn_from_web(self, url: str) -> str:
        """从网页学习"""
        try:
            # 验证URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return "无效的URL"
            
            # 获取网页内容
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # 解析网页
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取主要内容
            main_content = ' '.join([p.text for p in soup.find_all(['p', 'article', 'section'])])
            title = soup.title.string if soup.title else url
            
            # 保存到知识库
            self.knowledge_base['web'][url] = {
                'title': title,
                'content': main_content,
                'learned_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_knowledge_base()
            
            return f"成功学习了网页内容：{title}"
            
        except Exception as e:
            return f"学习网页时发生错误：{str(e)}"
    
    def learn_from_file(self, file) -> str:
        """从文件学习"""
        try:
            file_type = file.type
            content = ""
            file_name = file.name
            
            if "pdf" in file_type.lower():
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                content = " ".join([page.extract_text() for page in pdf_reader.pages])
            
            elif "text" in file_type.lower():
                content = file.getvalue().decode("utf-8")
            
            elif "excel" in file_type.lower() or "spreadsheet" in file_type:
                df = pd.read_excel(file)
                content = df.to_string()
            
            else:
                return "不支持的文件类型"
            
            self.knowledge_base['file'][file_name] = {
                'content': content,
                'type': file_type,
                'learned_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_knowledge_base()
            
            return f"成功学习了文件：{file_name}"
            
        except Exception as e:
            return f"处理文件时发生错误：{str(e)}"
    
    def find_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关知识"""
        if not self.vectors:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        relevant_docs = []
        for idx in similarities.argsort()[-top_k:][::-1]:
            if similarities[idx] > 0.1:  # 相似度阈值
                for category in self.knowledge_base.values():
                    for info in category.values():
                        if info['content'] == self.vectorizer.inverse_transform(self.vectors[idx])[0]:
                            relevant_docs.append(info['content'])
        
        return relevant_docs
    
    def generate_response(self, user_input: str) -> str:
        try:
            # 获取相关知识
            relevant_knowledge = self.find_relevant_knowledge(user_input)
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能助手，可以学习和利用各种来源的知识。"
                }
            ]
            
            if relevant_knowledge:
                knowledge_prompt = "根据以下相关知识回答：\n" + "\n".join(relevant_knowledge)
                messages.append({"role": "system", "content": knowledge_prompt})
            
            messages.extend(self.messages)
            messages.append({"role": "user", "content": user_input})
            
            # 生成回答
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            return f"生成回答时发生错误：{str(e)}"

def main():
    st.set_page_config(page_title="智能学习助手", page_icon="🤖", layout="wide")
    st.title("智能学习助手 🤖")
    
    # 初始化会话状态
    if "ai" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        st.session_state.ai = AdvancedLearningAI(api_key)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 创建三列布局
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # 主对话区域
    with col1:
        st.subheader("对话区")
        user_input = st.text_input("请输入您的问题：")
        
        if user_input:
            with st.spinner("AI思考中..."):
                response = st.session_state.ai.generate_response(user_input)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # 显示对话历史
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.write(f"👤 您：{msg['content']}")
            else:
                st.write(f"🤖 AI：{msg['content']}")
            st.write("---")
    
    # 网页学习区域
    with col2:
        st.subheader("从网页学习")
        url = st.text_input("输入网页URL：")
        if st.button("学习网页"):
            with st.spinner("正在学习网页内容..."):
                result = st.session_state.ai.learn_from_web(url)
                st.write(result)
    
    # 文件学习区域
    with col3:
        st.subheader("从文件学习")
        uploaded_file = st.file_uploader(
            "上传文件（支持PDF、TXT、Excel）",
            type=["pdf", "txt", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            if st.button("学习文件"):
                with st.spinner("正在学习文件内容..."):
                    result = st.session_state.ai.learn_from_file(uploaded_file)
                    st.write(result)
    
    # 知识库显示
    st.sidebar.title("知识库")
    for category, items in st.session_state.ai.knowledge_base.items():
        if items:
            st.sidebar.subheader(f"{category.title()}知识")
            for source, info in items.items():
                with st.sidebar.expander(f"📚 {source}"):
                    st.write(f"学习时间: {info['learned_at']}")
                    st.write(f"内容预览: {info['content'][:200]}...")

if __name__ == "__main__":
    main()
