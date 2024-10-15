import re
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader

class PDFProcessingPipeline:
    def __init__(self, file_path, chunk_size=2000, chunk_overlap=500):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = None 
        self.nlp = spacy.load("en_core_web_sm")  
        self.file_path = file_path

    def load(self):
        return PyPDFLoader(self.file_path).load()
    
    def clean_text(self, text):
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return ' '.join(token.lemma_ for token in self.nlp(cleaned) 
                        if not token.is_stop and token.is_alpha)
    
    def process_texts(self):
        return [self.clean_text(chunk.page_content) for chunk in self.splitter.split_documents(self.load())]

    def embed_texts(self, texts):
        return self.model.encode(texts, clean_up_tokenization_spaces=True)

    def create_vector_store(self, texts):
        embeddings = self.embed_texts(texts)
        words_list = sorted(set(word for text in texts for word in text.split()))
        vector_store_data = []

        for i, embedding in enumerate(embeddings):
            lemmatized_words = texts[i].split()
            word_embedding = {word: 0.0 for word in words_list}
            
            for j, word in enumerate(lemmatized_words):
                if word in word_embedding and j < len(embedding):
                    word_embedding[word] = float(embedding[j])
            
            vector_store_data.append(word_embedding)

        self.vector_store = vector_store_data
        return self.vector_store
    
    def run_pipeline(self):
        return self.create_vector_store(self.process_texts())