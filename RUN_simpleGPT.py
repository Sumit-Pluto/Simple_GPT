#1) PACKAGES
###################################################################

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts.prompts import SimpleInputPrompt
import torch
import gradio




#2) LLAMA-INDEX CONFIG
###################################################################
############# CONFIG. FOR LLM #################
system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")    

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="microsoft/phi-2",
    model_name="microsoft/phi-2",
    device_map="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16}          

)

########### CONFIG. FOR RAG ##################
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)




#3) LLAMA-INDEX ENGINE
###################################################################

documents = SimpleDirectoryReader("/content/Data").load_data()         #----------------------> INPUT

index = VectorStoreIndex.from_documents(documents, service_context=service_context)    

query_engine = index.as_query_engine()

def predict(input, history):
  response = query_engine.query(input)                                 #----------------------> OUTPUT
  return str(response)




#4) WEB-UI
###################################################################

gradio.ChatInterface(predict).launch(share=True)
