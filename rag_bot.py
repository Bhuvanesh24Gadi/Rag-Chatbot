import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


os.environ["GOOGLE_API_KEY"] = "U"

def create_gemini_rag():
    print("🤖 Initializing Gemini RAG Chatbot...")

    # 1. LOAD DATA
    if not os.path.exists("knowledge.txt"):
        print("❌ Error: 'knowledge.txt' file not found.")
        return

    loader = TextLoader("knowledge.txt", encoding='utf-8')
    documents = loader.load()

    # 2. SPLIT TEXT
    # Gemini has a large context window, but splitting helps finding specific facts
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 3. CREATE EMBEDDINGS
    # We use Google's specific embedding model
    print("   ...Creating embeddings with Google models...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Create the vector database (Chroma)
    db = Chroma.from_documents(texts, embeddings)

    # 4. CREATE RETRIEVER
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # 5. CREATE THE CHAIN
    # We use 'gemini-1.5-flash' because it is fast and cheap/free
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    print("\n✅ Gemini Bot is ready! (Type 'quit' to stop)")
    print("--------------------------------------------------")

    # 6. CHAT LOOP
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            # Ask the question
            result = qa_chain.invoke({"query": query})
            
            # Print the Answer
            print(f"Gemini: {result['result']}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_gemini_rag()
