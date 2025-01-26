from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

# Configurar modelo (adaptado de tu código)
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "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"  # Sustituye con tu clave

model = ChatGroq(model="llama-3.3-70B-versatile", max_tokens=8000)

# Cargar datos
data_arco = pd.read_excel("2024 11 01 - Base Original.xlsx")
data_arco = data_arco[['Nombre del Instrumento', 'Descripción']].dropna()

# Configurar embeddings y vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(data_arco['Descripción'].tolist(), embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=vector_store.as_retriever())

# Función de recomendación
def recomendar_instrumentos(query):
    """
    Recomienda instrumentos basados en la consulta y filtra solo los instrumentos mencionados en el resumen.
    """
    # Generar respuestas iniciales
    resultados = qa_chain.run(query)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data_arco['Descripción'])

    # Crear una lista inicial de instrumentos relevantes
    instrumentos_recomendados = []
    for descripcion in resultados.split("\n"):
        descripcion = descripcion.strip()
        if descripcion:
            query_vector = tfidf.transform([descripcion])
            similitudes = cosine_similarity(query_vector, tfidf_matrix)
            indices_similares = similitudes.argsort()[0][-1:][::-1]  # Solo el más similar
            for idx in indices_similares:
                instrumento = {
                    "Nombre del Instrumento": data_arco.iloc[idx]['Nombre del Instrumento'],
                    "Descripción": data_arco.iloc[idx]['Descripción'][:300]
                }
                instrumentos_recomendados.append(instrumento)

    # Generar el resumen final
    nombres_instrumentos = [i["Nombre del Instrumento"] for i in instrumentos_recomendados]
    prompt_resumen = f"Genera un párrafo breve que explique cómo estos instrumentos: {', '.join(nombres_instrumentos)} se complementan para atender la necesidad: {query}"
    parrafo_resumen = model.invoke(prompt_resumen).content

    # Filtrar instrumentos mencionados en el resumen
    instrumentos_filtrados = [
        instrumento for instrumento in instrumentos_recomendados
        if instrumento["Nombre del Instrumento"] in parrafo_resumen
    ]

    return {
        "instrumentos": instrumentos_filtrados,
        "resumen": parrafo_resumen
    }


# Configurar Flask
app = Flask(__name__)
CORS(app) 

@app.route("/recomendar", methods=["POST"])
def recomendar():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "La consulta está vacía"}), 400
    resultado = recomendar_instrumentos(query)
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
