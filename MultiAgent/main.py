from agents import WorkflowManager
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):
  user_query: str
  
class UserResponse(BaseModel):
  response: str

manager = WorkflowManager()

@app.post("/chat", response_model=UserResponse)
def chatbot(req: UserRequest):
  response = manager.run(req.user_query)
  return UserResponse(response=response["data"]["result"])


if __name__ == "__main__":
  while True:
    user_query = input("Enter your query('exit' to quit): ")
    
    if user_query == 'exit':
      break
    
    manager = WorkflowManager()
    response = manager.run(user_query)
    
    print(response)