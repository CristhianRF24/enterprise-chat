from pathlib import Path 
import re
import spacy
import unidecode 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader

class PDFProcessingPipeline:
    def __init__(self, file_path, chunk_size=2000, chunk_overlap=500):
        self.loader = PyPDFLoader(file_path)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = None 
        self.nlp = spacy.load("en_core_web_sm")  
        self.file_path = file_path

    def load(self):
        documents = self.loader.load()
        return documents 

    def clean_text(self, text):
        text = text.lower()
        text = text.replace('ñ', 'n')
        text = unidecode.unidecode(text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', ' ', text)  
        doc = self.nlp(text)
        lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(lemmatized_words)

    def process_texts(self):
        documents = self.load() 
        chunks = self.splitter.split_documents(documents)
        cleaned_texts = [self.clean_text(chunk.page_content) for chunk in chunks]
        return cleaned_texts  

    def embed_texts(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings

    def create_vector_store(self, texts):
        embeddings = self.embed_texts(texts)
    
        all_lemmatized_words = set()
        for text in texts:
            lemmatized_words = text.split()  
            all_lemmatized_words.update(lemmatized_words)
    
        words_list = sorted(list(all_lemmatized_words)) 
    
        vector_store_data = []
        
        for i, embedding in enumerate(embeddings):
            lemmatized_words = texts[i].split()  
            
            word_embedding = {word: 0 for word in words_list}
            
            for j, word in enumerate(lemmatized_words):
                if word in word_embedding and j < len(embedding):
                    word_embedding[word] = float(embedding[j])
    
            vector_store_data.append(word_embedding)
    
        vector_store_json = vector_store_data
        self.vector_store = vector_store_json
        return self.vector_store


    def run_pipeline(self):
        cleaned_texts = self.process_texts()
        vector_store = self.create_vector_store(cleaned_texts)
        return vector_store