import os
from dotenv import load_dotenv
from pinecone import Pinecone
import time

def reset_entire_index():
    """
    WARNING: This will delete ALL vectors from the specified Pinecone index.
    This action cannot be undone.
    """
    load_dotenv()
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY not found in .env file.")
        return

    index_name = "rag-lab-index" 

    try:
        print(f"Connecting to Pinecone...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist. Nothing to clear.")
            return

        index = pc.Index(index_name)
        
        print(f"Fetching stats for index '{index_name}' before deletion...")
        stats_before = index.describe_index_stats()
        print(stats_before)
        
        if stats_before.get('total_vector_count', 0) == 0:
            print("Index is already empty.")
            return

        confirm = input(f"Are you sure you want to delete ALL {stats_before.get('total_vector_count', 0)} vectors from the index '{index_name}'? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return

        print(f"Deleting all vectors from index '{index_name}'...")
        index.delete(delete_all=True)
        
        print("Waiting for deletion to complete...")
        while True:
            stats_after = index.describe_index_stats()
            if stats_after.get('total_vector_count', 0) == 0:
                break
            time.sleep(5)

        print("\nSuccessfully cleared the index.")
        print("Current index stats:")
        print(index.describe_index_stats())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    reset_entire_index()