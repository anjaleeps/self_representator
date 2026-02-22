import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb.config import Settings as ChromaSettings
from chromadb import Client
from data_loaders import DataLoader, GitHubDataLoader, MediumDataLoader, LocalDocumentLoader

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Indexer:

    def __init__(self, vector_store: BasePydanticVectorStore, embed_model: BaseEmbedding, nodes: List[BaseNode]):
        self.vector_store = vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index(nodes, self.storage_context, embed_model)
        
    def index(self, nodes: BaseNode, storage_context: StorageContext, embed_model: BaseEmbedding):
        self.vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)


class IngestionPipeline:

    def __init__(
            self, 
            data_loaders: List[DataLoader],
            vector_store: BasePydanticVectorStore, 
            embed_model: BaseEmbedding):
        self._data_loaders = data_loaders
        self._vector_store = vector_store
        self._embed_model = embed_model

    def run(self):
        logger.info("Starting self-representator pipeline")
        all_nodes = self._load_data_nodes()
        self.index = Indexer(self._vector_store, self._embed_model, all_nodes)
        logger.info("Index built with %d nodes", len(all_nodes))   

    def _load_data_nodes(self):
        all_nodes = []
        for loader in self._data_loaders:
            try:
                nodes = loader.load()
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error("Loader %s failed: %s", loader.__class__.__name__, e)
        return all_nodes
    

class QueryEngine:

    def __init__(self, vector_store: BasePydanticVectorStore, embed_model: BaseEmbedding):
        self.retriever = self._initialize_query_engine(vector_store, embed_model)
        self.system_prompt = self._build_chat_system_prompt()

    def _initialize_query_engine(self, vector_store: BasePydanticVectorStore, embed_model: BaseEmbedding):
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
        return index.as_retriever(similarity_top_k=5)

    def _build_chat_system_prompt(self):
        name = os.getenv("USER_FULL_NAME")
        return (
            f"You are acting as {name}. Answer the user's question about {name} "
            f"using only the retrieved context provided in the conversation. "
            f"Be professional and engaging, as if speaking to a potential client or employer on {name}'s website. "
            f"Only use the parts of the context that are relavant to the user's query when forming answers."
            f"If the provided context doesn't include enough information to provide an answer, say that you don't know honestly. "
            f"Stay in character as {name} at all times."
        ) 
    
    def _build_query_enhancement_prompt(self, query: str, history: List[Dict]) -> str:
        return (
            "You are a query refiner. Given the user's latest question and the chat history, "
            "rewrite the question to maximize retrieval quality. "
            "Goals: (1) keep the user’s intent, (2) add relevant context and keywords implied by history, "
            "and (3) if the question is complex, break it into clear sub-questions.\n"
            "Chat history:\n"
            f"{history}\n"
            "User question:\n"
            f"{query}\n\n"
            "Produce:\n"
            "- A single enhanced query OR a short numbered list of decomposed sub-queries.\n"
            "- Avoid adding facts not supported by the history.\n"
            "- Keep it concise and information-dense."
        )
    
    def retrieve(self, prompt: str) -> List[NodeWithScore]:
        nodes = self.retriever.retrieve(prompt)
        scores = [(node.get_content(), node.node.get_metadata_str(), node.get_score()) for node in nodes]
        
        results_msg = "\n========Retrieved Context Scores======\n\n"
        for idx, score in enumerate(scores):
            results_msg += f"{idx}.\n Content: {score[0][:200]}...\nMetadata: {score[1]}\nScore: {score[2]}\n\n"

        logger.info(results_msg)

        return nodes

    def query(self, prompt: str, history: List[Dict], llm_client: OpenAI, model: str) -> str:
        history = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        enhanced_prompt = self._enhance_prompt(prompt, history, llm_client, model)
        logger.info(f"Enhanced user prompt: {enhanced_prompt}")
        
        nodes = self.retrieve(enhanced_prompt)
        context = "\n\n".join(n.node.get_content() for n in nodes)
        full_prompt = f"User message: {prompt} \n Retrieved Context: {context}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": full_prompt}
        ]
        response = self._call_llm(llm_client, model, messages)
        return response
    
    def _enhance_prompt(self, prompt: str, history: List[Dict], llm_client: OpenAI, model: str) -> str:
        messages = [
            {"role": "user", "content": self._build_query_enhancement_prompt(prompt, history)}
        ]
        response = self._call_llm(llm_client, model, messages)
        return response
    
    def _call_llm(self, llm_client: OpenAI, model: str, messages: List[Dict],) -> str:
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise ValueError("LLM call failed") from e


class SelfRepresentator:

    def __init__(self):
        self._vector_store = self._initialize_vector_store()
        self._embed_model = self._initialize_embed_model()
        self._llm_client = self._initialize_llm_client()

        self._ingestion_pipeline = IngestionPipeline(
            data_loaders=[ LocalDocumentLoader(), GitHubDataLoader(), MediumDataLoader()],
            vector_store=self._vector_store, 
            embed_model=self._embed_model
        )
        self._query_engine = None

    def start(self):
        self._ingestion_pipeline.run()

    def query(self, prompt: str, history: List[Dict]):
        if not self._query_engine:
            try: 
                self._query_engine = QueryEngine(self._vector_store, self._embed_model)
                logger.info("Started query engine")
            except Exception as e:
                logger.error(f"Error starting the query engine: {str(e)}")
                raise ValueError("Error starting the query engine! Try running start before using the querying")
        return self._query_engine.query(prompt, history, self._llm_client, os.getenv("LLM_MODEL"))
    
    def update_knowledge(self):
        self._ingestion_pipeline.run()

    def _initialize_vector_store(self):
        client = Client(ChromaSettings(is_persistent=True, persist_directory="chromadb"))
        chroma_collection = client.get_or_create_collection(os.getenv("VECTOR_DB_NAME"))
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logger.info("Connected to Chroma collection %s", chroma_collection.name)
        return vector_store
    
    def _initialize_embed_model(self):
        return OpenAIEmbedding(model=os.getenv("EMBED_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

    def _initialize_llm_client(self):
        return OpenAI(base_url=os.getenv("GROQ_BASE_URL"), api_key=os.getenv("GROQ_API_KEY"))    


if __name__ == "__main__":
    rep = SelfRepresentator()
    rep.start()
    answer = rep.query("Have you worked with ai frameworks?", [])
