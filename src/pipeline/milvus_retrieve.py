import logging 
from typing import Optional, List
import time
from dataclasses import asdict
from src.configs.schema import (
    MilvusConfig,
    OtherConfig,
    CrossEmbeddingConfig,
    AsymRerankConfig
)

from src.core.storage_context import StorageContext
from src.core.service_context import ServiceContext
from src.engine.retrieve_engine import RetriverEngine
from src.vector_stores.milvus import MilvusVectorStore
from src.callbacks.callback_manager import CallbackManager
from src.embeddings.huggingface import CrossEncoder
from src.retriever.dense import VectorIndexRetriever
from src.reranker.asymmetric_reranker import AsymRanker
from src.retriever.types import QueryBundle

logger = logging.getLogger(__name__)

class MilvusRetrieverPipeline:
    def __init__(self,
        milvus_config: MilvusConfig,
        encoder_config: CrossEmbeddingConfig,
        other_config: OtherConfig,
        asym_config: Optional[AsymRerankConfig] = None
    ) -> None:
        self.milvus_config = milvus_config
        # Ensure not building index from scratch
        if self.milvus_config.overwrite == True:
            self.milvus_config.overwrite = False
        self.asym_config = asym_config
        self.other_config = other_config

        # callback manager
        self.callback_manager = CallbackManager()

        # encoder
        emb_model = self.get_encoder(config=encoder_config)

        self.service_context = ServiceContext.from_defaults(
            embed_model=emb_model,
            callback_manager=self.callback_manager
        )


    def main(self, query: str):

        """Wrap time"""
        start_build_collection_index = int(round(time.time() * 1000))

        #TODO: Connect to milvus to access collection and index
        milvus_vector_store = MilvusVectorStore(**asdict(self.milvus_config))
        # construct index and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=milvus_vector_store
        )
        
        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        """Wrap for loading retriever and search"""
        start_build_retrieve_search = int(round(time.time() * 1000))

        #TODO: build retriever
        retriever = VectorIndexRetriever(
            vector_store=milvus_vector_store,
            similarity_top_k=self.other_config.similarity_top_k,
            vector_store_query_mode=self.other_config.vector_store_query_mode,
            callback_manager=self.callback_manager,
            service_context=self.service_context,
            storage_context=storage_context
        )

        #TODO: rerank
        rerank = None
        if self.asym_config is not None:
            rerank = AsymRanker(
                model_name_or_path= self.asym_config.model_name_or_path,
                token= self.asym_config.token,
                device= [0],
            )

        #TODO: query
        engine = RetriverEngine(
            callback_manager=self.callback_manager,
            retriever=retriever,
            reranker=rerank
        )
        
        #TODO: query
        nodes, rerank_nodes = engine.search(QueryBundle(query))

        end_build_retrieve_search = int(round(time.time() * 1000))
        print(f"Time for build retriever and search: {end_build_retrieve_search - start_build_retrieve_search} ms")

        return nodes, rerank_nodes

    def apply_rerank():
        pass

    def get_encoder(self, config: CrossEmbeddingConfig) -> CrossEncoder:
        """create instace for cross embedding"""
        emb_model = CrossEncoder(
            qry_model_name=config.qry_model_name,
            psg_model_name=config.psg_model_name,
            token=config.token,
            device=config.device,
        )
        return emb_model
    