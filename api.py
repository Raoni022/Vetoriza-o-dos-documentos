from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from supabase import create_client
import os
import docx
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table_name="documents_embeddings",
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
    try:
        resultado = qa_chain.invoke({"query": p.pergunta})
        return {
            "resposta": resultado["result"],
            "fontes": [doc.metadata for doc in resultado["source_documents"]]
        }
    except Exception as e:
        return {"erro": str(e)}

@app.post("/melhorar_documento")
async def melhorar_documento(file: UploadFile = File(...)):
    try:
        doc_bytes = await file.read()
        doc = docx.Document(BytesIO(doc_bytes))
        texto_original = "\n".join([p.text for p in doc.paragraphs])

        prompt = f"Melhore o texto a seguir mantendo estilo e estrutura: \n\n{texto_original}"
        resposta = ChatOpenAI(model="gpt-4", temperature=0).invoke(prompt)

        for i, p in enumerate(doc.paragraphs):
            if i < len(resposta.content.split("\n")):
                p.text = resposta.content.split("\n")[i]

        output_stream = BytesIO()
        doc.save(output_stream)
        output_stream.seek(0)
        return {
            "mensagem": "Documento melhorado com sucesso.",
            "arquivo_base64": output_stream.getvalue().hex()
        }
    except Exception as e:
        return {"erro": str(e)}

@app.post("/conversar")
async def conversar(p: Pergunta):
    try:
        contexto = f"""
Usuário está prestes a enviar um documento para ser melhorado. 
Explique como o sistema funciona e peça para ele confirmar o envio.

Pergunta: {p.pergunta}
        """
        resposta = ChatOpenAI(model="gpt-4", temperature=0).invoke(contexto)
        return {"resposta": resposta.content}
    except Exception as e:
        return {"erro": str(e)}
