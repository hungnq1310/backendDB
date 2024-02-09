from pathlib import Path
import time

from transformers import AutoTokenizer, AutoModel

from src.pipeline.create_db import InitializeDatabase
from src.config.configuration import ConfigurationManager
from src.core.service_context import ServiceContext
from src.callbacks import CallbackManager
from src.node_parser.text.sentence import SentenceSplitter
from src.embeddings.huggingface import CrossEncoder


# tạo database 
# input data

from src.reader.dir_reader import DirectoryReader
import glob

startTime_load = int(round(time.time() * 1000))

pdf_files = glob.glob("/home/hungnq/hungnq_2/backend_db/backendDB/data/*.docx")
print("pdf_files: ", pdf_files)
reader = DirectoryReader(
    input_files=pdf_files
)
print("reader: ", reader.__dict__)
pdf_documents = reader.load_data()
if pdf_documents:
    print("Load pdf success")

# ------------------

# config
# params
# service context
# synthesizer
# node processor

manager = ConfigurationManager(
    config_filepath=Path("/home/hungnq/hungnq_2/backend_db/backendDB/configs/config.yaml"),
    param_filepath=Path("/home/hungnq/hungnq_2/backend_db/backendDB/configs/params.yaml")
)
node_parser_config = manager.get_node_parser_config()
node_parser_params = manager.get_node_parser_params()
milvus_config = manager.get_milvus_config()
milvus_params = manager.get_milvus_params()
embed_params = manager.get_embed_params()
index_retriver_params = manager.get_index_retriever_params()

#callback manager
callback_manager = CallbackManager()

# node parser
tokenizer = AutoTokenizer.from_pretrained(node_parser_params.model_name_tokenizer)

node_parser = SentenceSplitter(
    separator=node_parser_params.separator,
    chunk_size=node_parser_params.chunk_size,
    chunk_overlap=node_parser_params.chunk_overlap,
    tokenizer=tokenizer.encode,
    paragraph_separator="\n\n\n",
    secondary_chunking_regex=node_parser_params.secondary_chunking_regex,
    callback_manager=callback_manager,
)

# emb_model
emb_model = CrossEncoder(
    qry_model_name="pythera/mbert-retrieve-qry-base",
    psg_model_name="pythera/mbert-retrieve-ctx-base",
    token="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",
    device=[0],
)

service_context = ServiceContext.from_defaults(
    embed_model=emb_model,
    node_parser=node_parser,
    callback_manager=callback_manager,
    )

# cần 2 params and config: của milvus và index -> done
pipeline = InitializeDatabase(
    milvus_config=milvus_config,
    milvus_params=milvus_params,
    index_params=index_retriver_params,
    service_context=service_context,
    callback_manager=callback_manager
)
endTime_load = int(round(time.time() * 1000))
print(f"Time for load pipeline: {endTime_load - startTime_load} ms")


startTime_run= int(round(time.time() * 1000))
pipeline.main(documents=pdf_documents)
endTime_run= int(round(time.time() * 1000))
print(f"Time for load pipeline: {endTime_run - startTime_run} ms")