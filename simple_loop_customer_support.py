from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class WorkflowState:
  user_message: str = ""
  messages: List[Any] = None
  current_state: str = ""
  data: Dict[str, Any] = None
  iterationCounter: int = 0


class BaseAgent(ABC):
  def __init__(self, name):
    self.name = name
    self.role = "BaseAgent"
  
  @abstractmethod
  def process(self,state: WorkflowState)->WorkflowState:
    pass
  
  def add_message(self, state: WorkflowState, msg: str):
    msg = f"{self.name}: {msg}"
    state.messages.append(msg)

class OrchestrationAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "OrchestrationAgent"
  
  
  def process(self, state: WorkflowState)->WorkflowState:
    user_msg = state.user_message.lower()
    print("User Query: ", user_msg)
    
    billing_key = ["billing", "payment", "invoice", "charge", "calculate", "bill", "cost", "price", "amount", "refund", "transaction", "receipt", "subscription", "plan", "fee", "credit", "debit", "statement", "balance", "due", "refund", "fund", "money"]
    
    technical_key = ["technical", "issue", "problem", "error", "bug", "glitch", "malfunction", "failure", "support", "help", "assist", "troubleshoot", "diagnose", "fix", "repair", "solution"]
    
    
    if any(key in user_msg for key in billing_key):
      state.current_state = "Billing"
    elif any(key in user_msg for key in technical_key):
      state.current_state = "Technical"
    else:
      state.current_state = "General"
      
    self.add_message(state, f"Next state/agent to go {state.current_state}")
    return state

class GeneralAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "GeneralAgent"
    
  def process(self, state: WorkflowState) -> WorkflowState:
    user_msg = state.user_message.lower()
    print(f"General Agent processing...{user_msg}")
    
    response = "This is General service, we will solve your problem in no time."
    state.data["result"] = response
    self.add_message(state, f"Response, {response}")
    return state

class BillingAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "BillingAgent"
    
  def process(self, state: WorkflowState) -> WorkflowState:
    user_msg = state.user_message.lower()
    print(f"Billing Agent processing...{user_msg}")
    
    response = "This is Billing service, we will solve your problem in no time."
    state.data["result"] = response
    self.add_message(state, f"Response, {response}")
    return state

class TechnicalAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "TechnicalAgent"
    
  def process(self, state: WorkflowState) -> WorkflowState:
    user_msg = state.user_message.lower()
    print(f"Technical Agent processing...{user_msg}")
    
    response = "This is Technical service, we will solve your problem in no time."
    state.data["result"] = response
    self.add_message(state, f"Response, {response}")
    return state

class RespondAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "RespondAgent"
  
  def process(self, state: WorkflowState) -> WorkflowState:
    result = state.data.get("result", "No result available")
    formatted_response = f"Response: {result}"
    self.add_message(state, formatted_response)
    
    print(f"Final Response: {formatted_response}")
    state.current_state = "Completed"
    return state
  
class ValidationAgent(BaseAgent):
  def __init__(self, name):
    super().__init__(name)
    self.role = "ValidationAgent"
  
  def process(self, state: WorkflowState) -> WorkflowState:
    result = state.data.get("result", "No result available")
    if not result or "no result available" in result.lower():
      error_message = "Validation Error: No valid result found."
      self.add_message(state, error_message)
      state.current_state = "orchestration"
      print(f"Validation Error: {error_message}")
    else:
      success_message = "Validation successful, proceeding to respond."
      self.add_message(state, success_message)
      print(f"Validation Success: {success_message}")
      state.current_state = "respond"
      
    state.iterationCounter += 1
    if state.iterationCounter > 5:
      error_message = "Validation Error: Too many iterations, stopping workflow."
      self.add_message(state, error_message)
      state.current_state = "respond"
      state.data["result"] = "Workflow stopped due to too many iterations."
      print(f"Validation Error: {error_message}")
    return state
    


class WorkflowManager:
  def __init__(self):
    self.orchestration = OrchestrationAgent("OrchestrationAgent")
    self.general = GeneralAgent("GeneralAgent")
    self.billing = BillingAgent("BillingAgent")
    self.technical = TechnicalAgent("TechnicalAgent")
    self.respond = RespondAgent("RespondAgent")
    self.validation = ValidationAgent("ValidationAgent")
    self.workflow = self._build_workflow()
  
  def _build_workflow(self):
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("orchestration", self._orchestration_node)
    workflow.add_node("billing", self._billing_node)
    workflow.add_node("general", self._general_node)
    workflow.add_node("technical", self._technical_node)
    workflow.add_node("respond", self._respond_node)
    workflow.add_node("validation", self._validation_node)
    
    workflow.add_edge(START, "orchestration")
    
    workflow.add_conditional_edges(
      "orchestration",
      lambda state: state.current_state.lower()
    )
    workflow.add_edge("billing", "validation")
    workflow.add_edge("general", "validation")
    workflow.add_edge("technical", "validation")
    workflow.add_edge("validation", "respond")
    
    workflow.add_conditional_edges(
      "validation",
      lambda state: state.current_state.lower()
    )
    
    workflow.add_edge("respond", END)
    return workflow.compile()
    
  def _orchestration_node(self, state:WorkflowState)->WorkflowState:
    return self.orchestration.process(state)
  
  def _billing_node(self, state:WorkflowState)->WorkflowState:
    return self.billing.process(state)
  
  def _general_node(self, state:WorkflowState)->WorkflowState:
    return self.general.process(state)
  
  def _technical_node(self, state:WorkflowState)->WorkflowState:
    return self.technical.process(state)
  
  def _respond_node(self, state:WorkflowState)->WorkflowState:
    return self.respond.process(state)
  
  def _validation_node(self, state:WorkflowState)->WorkflowState:
    return self.validation.process(state)
  
  def run(self, query: str) -> WorkflowState:
    print("Multi-agent System started processing this query", query)
    
    initial_state = WorkflowState(
      user_message=query,
      messages=[],
      current_state="Start(Orchestration)",
      data={},
      iterationCounter=0
    )
    result = self.workflow.invoke(initial_state)
    return result
       
if __name__ == "__main__":
  user_query = "I have a billing issue with my last invoice"
  
  manager = WorkflowManager()
  response = manager.run(user_query)
  print("Workflow completed successfully!")
  print("Final State:", response["current_state"])
  
  print("Messages:", response["messages"])
  for msg in response["messages"]:
    print(msg)