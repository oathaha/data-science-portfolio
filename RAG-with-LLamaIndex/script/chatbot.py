import streamlit as st

from llama_index.core import  VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_pipeline import InputComponent, QueryPipeline
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor,  MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Show title and description.
st.title("💬 Australian visa assistance")
st.header("Ask me anything about Australian visa.")

openai_api_key = st.secrets["OPENAI_API_KEY"]


@st.cache_resource
def prepare_pipeline():

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    client = qdrant_client.QdrantClient(
        url = st.secrets["QDRANT_URL"],
        api_key = st.secrets["QDRANT_API_KEY"]
    )

    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="visa-info",
        enable_hybrid=True
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    ## prepare pipeline component

    input_comp = InputComponent()

    response_synthesizer = get_response_synthesizer(response_mode="compact")

    similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

    metadata_replacement_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window_context",
    )

    post_processors = [similarity_postprocessor, metadata_replacement_postprocessor]


    retriever = VectorIndexRetriever(
        index = index,
        similarity_top_k = 10,
        vector_store_query_mode = 'hybrid',
        alpha = 0.8, 
        hybrid_top_k = 10
    )

    query_engine = RetrieverQueryEngine(
        retriever = retriever,
        node_postprocessors = post_processors,
        response_synthesizer = response_synthesizer
    )


    ## define custom prompt

    custom_prompt = """You are an assistant for question-answering tasks related to visa application in Australia.

    Use the following pieces of retrieved context to answer the user's query:

    ---------------------\n
    {context_str}\n
    ---------------------\n

    Query: {query_str}
    """

    custom_prompt_template = PromptTemplate(custom_prompt)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt_template})



    ## create pipeline

    pipeline = QueryPipeline(
        chain = [input_comp, query_engine, llm],
        verbose=False
    )

    st.success('Chatbot is ready. You can ask questions now.')

    return pipeline

pipeline = prepare_pipeline()


# Initialize the chat message history
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. I am an Australian visa assistance. You can ask me about visa in Australia."}
    ]

# Prompt for user input and save to chat history

prompt = st.chat_input("Please enter your question here.")

if prompt: 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            ## get response from pipeline
            response = pipeline.run(input=prompt)

            response_str = str(response)

            if 'assistant: ' in response_str:
                response_str = response_str.replace('assistant: ', '')
                
            st.write(response_str)
            message = {"role": "assistant", "content": response_str}
            st.session_state.messages.append(message) # Add response to message history
