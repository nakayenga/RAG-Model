import mlflow
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch, DatabricksEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# define a retriever function that will be used in the chain
def vector_search_as_retriever(persist_dir=None):
    vectorstore = DatabricksVectorSearch(vs_index_table_name)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# define template for prompt
prompt_template = PromptTemplate.from_template(
    """
    You are a hand-crafted product design expert and your task is to creative products that are very good and can be sold online.

    Write a product title and description that is similar to the following product title and item details.

    Maximum 300 words.

    Use the following product title and details as example;

    <context>
    {context}
    </context>

    Question: {question}

    Answer:
    """
)

# define foundation model for generating responses
model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 500, temperature=0.8)

# RAG chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt_template
    | model
    | StrOutputParser()
)

# let's give it a try:
input_example = {"messages": [ {"role": "user", "content": "handmade lamp with remote control"}]}
answer = chain.invoke(input_example)
print(answer)
