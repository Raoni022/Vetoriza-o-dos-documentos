from fastapi import FastAPI, UploadFile, File
from docx import Document
from openai import OpenAI
import os
import tempfile

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/melhorar-documento")
async def melhorar_documento(file: UploadFile = File(...)):
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    temp_in.write(await file.read())
    temp_in.close()

    doc = Document(temp_in.name)
    for para in doc.paragraphs:
        if para.text.strip():
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Melhore o texto mantendo o mesmo sentido, tom formal e sem alterar o layout."},
                    {"role": "user", "content": para.text}
                ]
            )
            para.text = response.choices[0].message.content.strip()

    output_path = temp_in.name.replace(".docx", "_melhorado.docx")
    doc.save(output_path)

    return {"mensagem": "Documento melhorado com sucesso.", "arquivo": output_path}