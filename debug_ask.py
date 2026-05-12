"""Quick diagnostic - tests each component in isolation."""
import logging, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

print("\n=== STEP 1: Imports ===")
try:
    from Fitness_App import (EMBEDDING_MODEL, CHROMA_DIR, LLM_MODEL, DB_PATH, 
                              DEFAULT_USER, TOP_K, build_llm, build_rag_chain, 
                              UserProfileManager, clean_answer)
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    print("  OK")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== STEP 2: Embeddings ===")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    print("  OK")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== STEP 3: ChromaDB ===")
try:
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    print(f"  OK - {vectorstore._collection.count()} vectors loaded")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== STEP 4: LLM ===")
try:
    llm = build_llm()
    print("  OK")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== STEP 5: RAG Chain ===")
try:
    db_manager = UserProfileManager(DB_PATH)
    chain, _ = build_rag_chain(llm, retriever, db_manager)
    print("  OK")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== STEP 6: Test Query ===")
try:
    answer = chain.invoke("hello")
    print(f"  Answer: {clean_answer(answer)}")
except Exception as e:
    traceback.print_exc()
    exit(1)

print("\n=== ALL TESTS PASSED ===")
