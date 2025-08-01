from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from model import embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "Chinook")

mysql_uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

class DatabaseConnect:
  def __init__(self, uri=mysql_uri):
    self.db = SQLDatabase.from_uri(uri)
        
  def get_db(self):
    return self.db
  
  def get_schema(self):
    return self.db.get_table_info()

  def get_table_names(self):
    return self.db.get_table_names()

  def get_columns(self, table_name: str):
    return self.db.get_columns(table_name)

  def execute_query(self, query: str):
    return self.db.execute_query(query)
      
  def run_query(self, query: str):
    return self.db.run(query)
  
class VectorDBConnect:
  def __init__(self):
    self.vector_store = Chroma(
      collection_name="MultiAgent",
      embedding_function=embedding,
      persist_directory="./chroma_db"
    )
    
  def text_split(self, data: str):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap = 200,
      add_start_index = True,
    )
    doc = [Document(page_content=data)]
    return text_splitter.split_documents(doc)
  
  def add_document(self, data: str):
    doc = self.text_split(data)
    self.vector_store.add_documents(documents=doc)
    
  def get_similar_content(self, query: str):
    return self.vector_store.similarity_search(query=query, k=5)
  
  