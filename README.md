# Examensarbete: Jämförelse av Embedding-modeller i RAG-system

Detta projekt är en vidareutveckling av [tolkien-rag-chatbot](https://github.com/ecthelionofthefountain-510/tolkien-rag-chatbot) för examensarbete på NBI/Handelsakademin.

## Syfte

Undersöka hur valet av embedding-modell påverkar retrieval-kvaliteten i ett Retrieval-Augmented Generation-system, med fokus på precision, svarstid och kostnad.

## Frågeställningar

1. Hur skiljer sig retrieval-precisionen mellan OpenAI:s embedding-modell och open-source-alternativ?
2. Vilka trade-offs finns mellan prestanda, kostnad och svarstid vid val av embedding-modell?

---

## Setup

### 1. Skapa och aktivera en venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Installera dependencies

```bash
pip install -r requirements.txt
```

### 3. Skapa `.env`

```env
OPENAI_API_KEY=din_api_nyckel
```

---

## Embedding-modeller som jämförs

| Modell | Typ | Dimension | Beskrivning |
|--------|-----|-----------|-------------|
| `text-embedding-3-small` | OpenAI (betald) | 1536 | Baseline, bra balans kostnad/prestanda |
| `all-MiniLM-L6-v2` | HuggingFace (gratis) | 384 | Snabb, populär för RAG |
| `multilingual-e5-base` | HuggingFace (gratis) | 768 | Bra på flerspråkigt innehåll |

### Lista alla stödda modeller

```bash
python -m src.ingest --list-models
```

---

## Bygg index

### En modell (som tidigare)

```bash
python -m src.ingest --rebuild --embedding-model text-embedding-3-small
```

### Flera modeller för jämförelse

```bash
python -m src.ingest --rebuild --models text-embedding-3-small all-MiniLM-L6-v2 multilingual-e5-base
```

### Alla stödda modeller

```bash
python -m src.ingest --rebuild --all-models
```

---

## Kör utvärdering

Kör test-datasetet mot olika modeller och samla in mätvärden:

```bash
python -m src.evaluate --models text-embedding-3-small all-MiniLM-L6-v2
```

### Output

Resultaten sparas i `results/evaluation.json` och skrivs ut i terminalen:

```
================================================================
COMPARISON RESULTS
================================================================

Model                          Type         Hit Rate   Precision  Keywords   Avg Time
--------------------------------------------------------------------------------
text-embedding-3-small         openai         85.0%      72.0%      68.0%     45.2ms
all-MiniLM-L6-v2               huggingface    78.0%      65.0%      62.0%     12.3ms
```

---

## Starta chatten

### Terminal

```bash
python -m src.chat --embedding-model text-embedding-3-small
```

### Streamlit (webb)

```bash
streamlit run src/streamlit.py
```

---

## Projektstruktur

```
├── src/
│   ├── embeddings.py    # NY: Factory för embedding-modeller
│   ├── evaluate.py      # NY: Utvärderingsskript
│   ├── ingest.py        # Uppdaterad: Stödjer flera modeller
│   ├── rag.py           # Uppdaterad: Använder embeddings.py
│   ├── chat.py          # Terminalchat
│   └── streamlit.py     # Webb-UI
├── data/
│   ├── raw/             # Källtexter (.txt)
│   └── chroma/          # Vektordatabaser (en per modell)
├── results/             # Utvärderingsresultat
└── requirements.txt
```

---

## Mätvärden som samlas in

- **Source Hit Rate**: Andel frågor där rätt källa hämtades
- **Source Precision**: Andel av hämtade källor som var relevanta
- **Keyword Recall**: Andel av förväntade nyckelord som hittades
- **Average Top Score**: Genomsnittlig relevans-score
- **Retrieval Time**: Tid för sökning (ms)

---

## Nästa steg (TODO)

- [ ] Bygga index för alla modeller
- [ ] Köra fullständig utvärdering
- [ ] Analysera resultat
- [ ] Skriva rapport

---

## Ursprungligt projekt

Baserat på kunskapskontroll i AI – teori och tillämpning:
- [tolkien-rag-chatbot](https://github.com/ecthelionofthefountain-510/tolkien-rag-chatbot)