from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from supabase import create_client
import os

app = FastAPI()

SUPABASE_URL = "https://kjwdegxainpuekfwajre.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtqd2RlZ3hhaW5wdWVrZndhanJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjYxOTY0MSwiZXhwIjoyMDYyMTk1NjQxfQ.8tOq83Bp68oSXslYWtrnYrLAJjTZZ0esH2A5uioDqOk"
OPENAI_API_KEY = "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"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 🔗 Conectar com Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table_name="documentos_embeddings",
    query_name="match_documents_embeddings"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=retriever,
    return_source_documents=True
)

class Pergunta(BaseModel):
    pergunta: str

@app.post("/responder")
async def responder(p: Pergunta):
    resultado = qa_chain.invoke({"query": p.pergunta})
    return {"resposta": resultado}
