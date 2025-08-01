from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import re

class SQLQueryChain:
  def __init__(self, db, llm):
    self.db = db
    self.llm = llm
    self.query_template = """
    Based on the table schema below, write a SQL query that would answer the user's question.
    Return ONLY the SQL query without any markdown formatting, explanations, or additional text.

    {schema}

    Question: {question}
    
    SQL Query:"""

    self.prompt = ChatPromptTemplate.from_template(self.query_template)
  
  def clean_sql_query(self, query: str) -> str:
    """Clean the SQL query by removing markdown formatting and extra whitespace."""
    query = re.sub(r'```sql\n?', '', query)
    query = re.sub(r'```\n?', '', query)
    query = query.strip()
    
    query = ' '.join(query.split())
    
    return query
    
  def get_chain(self):
    return (
      RunnablePassthrough.assign(schema=lambda x: self.db.get_schema())
      | self.prompt
      | self.llm.bind(stop=["\nSQLResult:", "```"])
      | StrOutputParser()
      | self.clean_sql_query
    )

class SQLChain(SQLQueryChain):
  def __init__(self, db, llm):
    super().__init__(db, llm)
    self.template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    Answer:"""
    self.prompt_response = ChatPromptTemplate.from_template(self.template)

  def get_chain(self):
    sql_chain = super().get_chain()
    
    return (
      RunnablePassthrough.assign(query=sql_chain)
      .assign(
        schema=lambda x: self.db.get_schema(),
        response=lambda x: self.db.run_query(x["query"]),
      )
      | self.prompt_response
      | self.llm.bind(stop=["\nResponse:"])
      | StrOutputParser()
    )

  def invoke(self, inputs):
    chain = self.get_chain()
    return chain.invoke(inputs)
  
def main():
  """CLI mode for testing"""
  try:
    db = DatabaseConnect()
    
    print("Database connected successfully!")
    
    chain = SQLChain(db, llm)
    while True:
      question = input("Enter your question (or type 'exit' to quit): ")
      if question.lower() == 'exit':
        break
      
      if not question.strip():
        print("Please enter a valid question.")
        continue
      
      response = chain.invoke({"question": question})
      print(f"Response: {response}")
  except Exception as e:
    print(f"Error: {e}")
    
if __name__ == "__main__":
  from model import *
  from db_connection import *
  main()