from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass
from typing import Union, List, Annotated, TypedDict, Any, Dict
from abc import ABC, abstractmethod

@dataclass
class WorkflowState:
  user_message: str = ""
  messages: List[str] = None
  current_state: str = "Start"
  data: Dict[str, Any] = None
  
class BaseAgent(ABC):
  def __init__(self, name):
    self.name = name
    self.role = "BaseAgent"
    
  @abstractmethod
  def process(self, state: WorkflowState)->WorkflowState:
    pass
  
  def add_message(self, state: WorkflowState, msg: str)->None:
    msg = f"{self.name}: {msg}"
    state.messages.append(msg)
  
class ReaderAgent(BaseAgent):
  def __init__(self):
    super().__init__("Reader")
    self.role = "ReaderAgent"
  
  
  def process(self, state: WorkflowState)->WorkflowState:
    user_msg = state.user_message.lower()
    print(f"Reader Agent....User query: {user_msg}")
    if "weather" in user_msg:
      query_type = "weather_query"
    elif "joke" in user_msg or "funny" in user_msg:
      query_type = "joke_query"
    else:
      query_type = "general_query"
    
    state.data['query_type'] = query_type
    state.current_state = "Processing State"
    self.add_message(state, f"User Message- {user_msg}")
    self.add_message(state, f"Query Type- {query_type}")
    return state
  
class ProcessAgent(BaseAgent):
  def __init__(self):
    super().__init__("Process")
    self.role = "ReaderAgent"
  
  def process(self, state: WorkflowState)->WorkflowState:
    query_type = state.data.get("query_type", "general_query")
    print(f"Processing Agent...Processing {query_type}")
    
    if query_type == "weather_query":
      state.data["result"] = "Today's weather is awesome"
    elif query_type == "joke_query":
      state.data["result"] = "I failed math so many times at school, I can't even count."
    else:
      state.data["result"] = "This is General service, how can I help."
    
    self.add_message(state, f"Result- {state.data.get("result", "No result found")}")
    state.current_state = "Response"
    return state
  
class RespondAgent(BaseAgent):
  def __init__(self):
    super().__init__("Response")
    self.role = "RespondAgent"
  
  
  def process(self, state: WorkflowState)->WorkflowState:
    result = state.data.get("result", "No result available")
    formatted_response = f"""
    ✅ **Task Completed Successfully!**
    
    📝 Your request: {state.user_message}
    🎯 Result: {result}
    
    Thank you for using our agent system!
    """
    
    state.data["final_response"] = formatted_response
    state.current_state = "complete"
    
    self.add_message(state, "Response - Response formatted and ready!")
    return state
  
class WorkflowManager:
  def __init__(self):
    self.reader = ReaderAgent()
    self.process = ProcessAgent()
    self.respond = RespondAgent()
    
    self.workflow = self._build_graph()
  
  
  def _build_graph(self):
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("reader", self._reader_node)
    workflow.add_node("process", self._process_node)
    workflow.add_node("respond", self._respond_node)
    
    workflow.add_edge(START, "reader")
    workflow.add_edge("reader", "process")
    workflow.add_edge("process", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()


  def _reader_node(self, state: WorkflowState)->WorkflowState:
    return self.reader.process(state)
  
  def _process_node(self, state: WorkflowState)->WorkflowState:
    return self.process.process(state)
  
  def _respond_node(self, state: WorkflowState)->WorkflowState:
    return self.respond.process(state)
  
  def run(self, user_query: str)->WorkflowState:
    print("Starting Simple Workflow Agent....")
    
    initial_state = WorkflowState(
      user_message= user_query,
      messages= [],
      current_state= "Start",
      data= {}
    )
    
    result = self.workflow.invoke(initial_state)
    print("Workflow completed successfully!")
    return result
  
if __name__ == "__main__":
  user_query = "What's the weather like today?"
  manager = WorkflowManager()
  final_state = manager.run(user_query)
  
  print("Final State:")
  print(final_state)
  
  print("Messages:")
  for msg in final_state['messages']:
    print(msg)
    
  print("Final Response:")
  print(final_state['data'].get('final_response', 'No response available'))