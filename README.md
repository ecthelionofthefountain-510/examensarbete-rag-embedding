# RAG Embedding Lab

> **Examensarbete** — Jämförelse av embedding-modeller i RAG-system  
> NBI/Handelsakademin

En fullstack-applikation för att utvärdera och jämföra hur olika embedding-modeller påverkar retrieval-kvaliteten i ett RAG-system (Retrieval-Augmented Generation). Projektet använder Tolkien-lore som test-domän.

## ✨ Features

- **Jämför embedding-modeller i realtid** – ställ en fråga och se hur tre olika modeller svarar sida vid sida
- **Visualiserad utvärdering** – radar-diagram och stapeldiagram för att jämföra precision, recall och svarstid
- **Modern React-frontend** – snygg, mörkt tema med live-resultat
- **FastAPI-backend** – RESTful API som wrappar RAG-logiken
- **CLI-verktyg** – bygg index, kör utvärdering och chatta direkt i terminalen

---

## 🏗️ Arkitektur

```
examensarbete-rag-embedding/
├── main.py                 # FastAPI backend
├── frontend/               # React + Vite frontend
│   ├── App.jsx             # Hela frontend (single-file)
│   ├── src/main.jsx
│   └── ...
├── src/                    # Python RAG-logik
│   ├── embeddings.py       # Factory för embedding-modeller
│   ├── evaluate.py         # Utvärderingsskript
│   ├── ingest.py           # Bygg vektorindex
│   ├── rag.py              # RAG-pipeline
│   ├── chat.py             # Terminal-chat
│   └── streamlit.py        # Alternativ Webb-UI
├── data/
│   ├── raw/                # Källtexter (.txt)
│   ├── chroma/             # Vektordatabaser (en per modell)
│   └── evaluation.json     # Testfrågor för utvärdering
└── results/                # Utvärderingsresultat
```

---

## 🚀 Snabbstart

### 1. Backend-setup

```bash
# Skapa och aktivera venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installera dependencies
pip install -r requirements.txt

# Skapa .env med din API-nyckel
echo "OPENAI_API_KEY=din_nyckel_här" > .env

# Bygg index för alla modeller
python -m src.ingest --rebuild --all-models

# Starta API:et
uvicorn main:app --reload --port 8000
```

API:et körs på `http://localhost:8000`  
Swagger-docs: `http://localhost:8000/docs`

### 2. Frontend-setup

```bash
cd frontend
npm install
npm run dev
```

Öppna `http://localhost:5173` i webbläsaren.

---

## 📊 Embedding-modeller

| Modell                   | Typ         | Dimension | Beskrivning                             |
| ------------------------ | ----------- | --------- | --------------------------------------- |
| `text-embedding-3-small` | OpenAI      | 1536      | Baseline – bra balans kostnad/prestanda |
| `all-MiniLM-L6-v2`       | HuggingFace | 384       | Snabb och populär för RAG               |
| `multilingual-e5-base`   | HuggingFace | 768       | Bra för flerspråkigt innehåll           |

```bash
# Lista alla stödda modeller
python -m src.ingest --list-models
```

---

## 🔧 CLI-kommandon

### Bygg vektorindex

```bash
# En specifik modell
python -m src.ingest --rebuild --embedding-model text-embedding-3-small

# Flera modeller
python -m src.ingest --rebuild --models text-embedding-3-small all-MiniLM-L6-v2

# Alla modeller
python -m src.ingest --rebuild --all-models
```

### Kör utvärdering

```bash
python -m src.evaluate --models text-embedding-3-small all-MiniLM-L6-v2 multilingual-e5-base
```

Resultaten sparas i `results/evaluation.json`.

### Terminal-chat

```bash
python -m src.chat --embedding-model text-embedding-3-small
```

### Streamlit (alternativ UI)

```bash
streamlit run src/streamlit.py
```

---

## 📡 API-endpoints

| Metod  | Endpoint      | Beskrivning                          |
| ------ | ------------- | ------------------------------------ |
| `GET`  | `/models`     | Lista alla modeller med metadata     |
| `POST` | `/chat`       | RAG-fråga med en modell              |
| `POST` | `/compare`    | Jämför flera modeller på samma fråga |
| `GET`  | `/evaluation` | Hämta sparade utvärderingsresultat   |
| `GET`  | `/health`     | Hälsokontroll                        |

---

## 📈 Mätvärden

- **Source Hit Rate** – Andel frågor där rätt källa hämtades
- **Source Precision** – Andel av hämtade källor som var relevanta
- **Keyword Recall** – Andel förväntade nyckelord som hittades
- **Average Top Score** – Genomsnittlig relevans-score
- **Retrieval Time** – Tid för sökning (ms)

---

## 🎓 Forskningsfrågor

1. Hur skiljer sig retrieval-precisionen mellan OpenAI:s embedding-modell och open-source-alternativ?
2. Vilka trade-offs finns mellan prestanda, kostnad och svarstid vid val av embedding-modell?

---

## 📚 Ursprung

Vidareutveckling av [tolkien-rag-chatbot](https://github.com/ecthelionofthefountain-510/tolkien-rag-chatbot) från kursen _AI – teori och tillämpning_.
