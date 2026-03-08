# RAG Embedding Lab — React Frontend + FastAPI

En snygg, modern frontend för examensarbetet. Ersätter/kompletterar Streamlit-appen med ett fullstack React-gränssnitt.

---

## Struktur

```
examensarbete-rag-embedding/
├── src/                    # Befintlig Python-kod (oförändrad)
│   ├── rag.py
│   ├── embeddings.py
│   ├── ingest.py
│   └── evaluate.py
├── frontend/               # NY — React-app
│   ├── App.jsx             # Hela frontend-appen (en fil)
│   ├── src/main.jsx        # Entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── main.py                 # NY — FastAPI backend
└── ...
```

---

## Setup

### 1. FastAPI backend

```bash
# Installera extra dependencies
pip install fastapi uvicorn

# Starta backend (från projektets rot)
uvicorn main:app --reload --port 8000
```

API:et körs på `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

### 2. React frontend

```bash
# Gå till frontend-mappen
cd frontend

# Installera Node-dependencies
npm install

# Starta dev-server
npm run dev
```

Appen körs på `http://localhost:3000`

---

## Vyer

| Vy | Beskrivning |
|---|---|
| **A/B Compare** | Ställ en fråga — se svar från alla 3 modeller sida vid sida |
| **Benchmark** | Interaktiv dashboard med dina evaluation-resultat |
| **Embedding Space** | Visualisering av retrieval quality-distribution |
| **Presentation** | Slide-deck för redovisningen (piltangenter för navigation) |

---

## Tips

- Benchmark-vyn och Embedding Space kräver att `data/evaluation.json` finns → kör `python -m src.evaluate` först
- A/B Compare kräver byggda Chroma-index → kör `python -m src.ingest --all-models` först
- Presentation-läget fungerar fristående med hårdkodad fallback-data

---

## Produktion

```bash
# Bygg React-appen
cd frontend && npm run build

# Servera static files via FastAPI (lägg till i main.py):
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```