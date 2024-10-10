import unittest
from app.pdf_processing_pipeline import PDFProcessingPipeline
from sklearn.metrics.pairwise import cosine_similarity

class TestPDFProcessingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = PDFProcessingPipeline("uploaded_files/CV file.pdf")

    # Test that the load method returns a list of documents from the PDF file.
    def test_load(self):
        documents = self.pipeline.load()
        self.assertIsInstance(documents, list) 

    # Test the clean_text method to ensure it correctly removes special characters
    def test_clean_text(self):
        sample_text = "This is a test text with special characters!!!"
        clean_text = self.pipeline.clean_text(sample_text)
        self.assertEqual(clean_text, "test text special character")  

    # Test the embed_texts method to verify that it returns an embedding for each input text
    def test_embed_texts(self):
        sample_texts = ["text one", "text two"]
        embeddings = self.pipeline.embed_texts(sample_texts)
        self.assertEqual(len(embeddings), len(sample_texts)) 
    
    #Test that the similarity between embeddings of similar texts is greater than that of 
    # embeddings of different texts, ensuring the embedding model captures semantic meaning.
    def test_embedding_similarity(self):
        sample_texts_similar = ["text one", "text one is great"]
        sample_texts_different = ["text one", "completely different text"]

        embeddings_similar = self.pipeline.embed_texts(sample_texts_similar)
        embeddings_different = self.pipeline.embed_texts(sample_texts_different)

        similarity_similar = cosine_similarity([embeddings_similar[0]], [embeddings_similar[1]])[0][0]
        similarity_different = cosine_similarity([embeddings_different[0]], [embeddings_different[1]])[0][0]

        self.assertGreater(similarity_similar, similarity_different)

    # Test the create_vector_store method to verify it returns a list
    def test_create_vector_store(self):
        sample_texts = ["text one", "text two"]
        vector_store = self.pipeline.create_vector_store(sample_texts)
        self.assertIsInstance(vector_store, list) 

if __name__ == '__main__':
    unittest.main()
