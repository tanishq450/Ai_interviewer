from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import loguru






class ModelLoader:
    def __init__(self):
        self.logger = loguru.logger
        
    def load_embedding_model(self):
        try:
            self.logger.info("Loading embedding model")
            embedding_model = OllamaEmbedding(model="qwen3-embedding:4b")
            self.logger.info("Embedding model loaded successfully")
            return embedding_model
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            return None 
    
    def load_llm(self):
        try:
            self.logger.info("Loading LLM")
            llm = Ollama(model="kimi-k2-thinking:cloud")
            self.logger.info("LLM loaded successfully")
            return llm
        except Exception as e:
            self.logger.error(f"Error loading LLM: {e}")
            return None 