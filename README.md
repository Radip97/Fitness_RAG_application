# Fitness RAG Application 

A comprehensive Retrieval-Augmented Generation (RAG) pipeline and web application designed to act as your personalized AI fitness coach. The application uses a local LLM to answer fitness-related questions based on curated datasets, extracts and tracks user progression over time, and provides a modern web interface for interactions and analytics.

## Features 
- **Conversational AI Coach**: Ask fitness, workout, and nutrition questions. The engine uses RAG to pull the most relevant information from local fitness datasets.
- **Automatic Stat Tracking**: The AI extracts stats like weight, body fat percentage, and personal records (e.g., bench press) directly from your conversation and saves them to a local SQLite database.
- **Progress Dashboard**: View your fitness journey and tracked metrics through an interactive web dashboard powered by Chart.js.
- **Local Inference**: Runs completely locally using HuggingFace models, LangChain, and ChromaDB for maximum privacy and control.

## Tech Stack 
- **Backend/Web**: Python, Flask
- **Machine Learning / AI**: LangChain, Transformers (HuggingFace)
- **Language Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Embeddings**: `BAAI/bge-base-en-v1.5`
- **Vector Store**: ChromaDB
- **Database**: SQLite
- **Frontend**: HTML, Vanilla CSS, JavaScript, Chart.js

## Project Structure 
- `app.py`: The main Flask web application containing routing and AI engine initialization.
- `Fitness_App.py`: Core RAG logic, model loading, LangChain setups, and user state management.
- `vectorize.py`: Script used to build and populate the ChromaDB vector store from raw JSON datasets.
- `dataset/`: Curated JSON files (`cardio.json`, `strength.json`, `flexibility.json`) providing the knowledge base.


## Setup & Installation 

### 1. Clone the repository
```bash
git clone https://github.com/Radip97/Fitness_RAG_application.git
cd Fitness_RAG_application
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the Vector Database
Before running the application, you need to embed the dataset into ChromaDB:
```bash
python vectorize.py
```

### 5. Run the Web Application
```bash
python app.py
```
> **Important Note on Models**: The first time you run `vectorize.py` and `app.py`, the system will automatically download the required Hugging Face models (`Qwen/Qwen2.5-3B-Instruct` for the LLM and `BAAI/bge-base-en-v1.5` for embeddings). This requires an active internet connection and will download several gigabytes of data.
> 
> If you wish to use a different Language Model or Embedding Model, you must update the `LLM_MODEL` and `EMBEDDING_MODEL` variables at the top of the `Fitness_App.py` file before running the application.

Access the application in your browser at: `http://localhost:5000`

## Contributing 🤝
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
