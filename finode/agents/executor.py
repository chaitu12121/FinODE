"""
ExecutorAgent

Tool-using agent that performs computations and API calls.
Can execute financial calculations, data lookups, and simulations.
"""

import json
import time
import math
from typing import Dict, Any, List, Optional, Callable
from .base_agent import BaseAgent, AgentMessage, AgentState
from ..llm.groq_client import GroqMessage


class ExecutorAgent(BaseAgent):
    """
    Executes tasks using tools and APIs.
    
    Responsibilities:
    - Tool invocation
    - Financial calculations
    - Data processing
    - Result formatting
    """
    
    def __init__(self, groq_client):
        """
        Initialize ExecutorAgent.
        
        Args:
            groq_client: GroqClient instance
        """
        super().__init__("ExecutorAgent", groq_client)
        self.available_tools = self._register_tools()
        
        self.system_prompt = """You are an execution agent for financial queries.
You have access to the following tools:

1. calculate_compound_interest(principal, rate, periods, compounds_per_year)
   - Calculates compound interest
   
2. calculate_portfolio_return(holdings: dict, price_changes: dict)
   - Calculates portfolio return percentage
   
3. calculate_pe_ratio(stock_price, earnings_per_share)
   - Calculates price-to-earnings ratio
   
4. calculate_discount_rate(risk_free_rate, beta, market_premium)
   - Calculates required return using CAPM
   
5. currency_conversion(amount, from_currency, to_currency)
   - Converts between currencies

When asked to perform a calculation, respond with:
{
  "tool_calls": [
    {
      "tool_name": "tool_name",
      "parameters": {param1: value1, param2: value2}
    }
  ],
  "reasoning": "why we use this tool"
}
"""
    
    def _register_tools(self) -> Dict[str, Callable]:
        """Register available tools"""
        return {
            "calculate_compound_interest": self._calc_compound_interest,
            "calculate_portfolio_return": self._calc_portfolio_return,
            "calculate_pe_ratio": self._calc_pe_ratio,
            "calculate_discount_rate": self._calc_discount_rate,
            "currency_conversion": self._currency_conversion,
        }
    
    def _calc_compound_interest(self, principal: float, rate: float, periods: int, compounds_per_year: int = 1) -> float:
        """A = P(1 + r/n)^(nt)"""
        r = rate / 100.0
        return principal * ((1 + r / compounds_per_year) ** (compounds_per_year * periods))
    
    def _calc_portfolio_return(self, holdings: Dict[str, float], price_changes: Dict[str, float]) -> float:
        """Calculate weighted portfolio return"""
        total_value = sum(holdings.values())
        if total_value == 0:
            return 0.0
        
        weighted_return = 0.0
        for symbol, value in holdings.items():
            weight = value / total_value
            change = price_changes.get(symbol, 0.0)
            weighted_return += weight * change
        
        return weighted_return
    
    def _calc_pe_ratio(self, stock_price: float, earnings_per_share: float) -> float:
        """P/E Ratio = Stock Price / EPS"""
        if earnings_per_share == 0:
            return float('inf')
        return stock_price / earnings_per_share
    
    def _calc_discount_rate(self, risk_free_rate: float, beta: float, market_premium: float) -> float:
        """CAPM: Cost of Equity = Rf + β(Rm - Rf)"""
        return risk_free_rate + beta * market_premium
    
    def _currency_conversion(self, amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        Simple currency conversion using hardcoded rates.
        In production, would call live exchange rate API.
        """
        # Simplified exchange rates (as of Feb 2026)
        rates = {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 149.50,
            "CAD": 1.37,
            "INR": 83.12,
        }
        
        if from_currency not in rates or to_currency not in rates:
            raise ValueError(f"Unsupported currency pair: {from_currency}/{to_currency}")
        
        converted = amount * (rates[to_currency] / rates[from_currency])
        return {
            "original_amount": amount,
            "original_currency": from_currency,
            "converted_amount": round(converted, 2),
            "target_currency": to_currency,
            "rate": round(rates[to_currency] / rates[from_currency], 4),
        }
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Execute a task from a plan.
        
        Args:
            message: Message with content["task_description"] and content["parameters"]
        
        Returns:
            Response with execution result
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            task_desc = message.content.get("task_description", "")
            parameters = message.content.get("parameters", {})
            
            if not task_desc:
                raise ValueError("No task description provided")
            
            # Use Groq to determine which tool(s) to call
            groq_messages = [
                GroqMessage(
                    role="user",
                    content=f"Task: {task_desc}\nParameters: {json.dumps(parameters)}\n\nWhich tools should I use?"
                )
            ]
            
            plan_response = self.groq_client.infer_json(
                groq_messages,
                system_prompt=self.system_prompt
            )
            
            # Execute tool calls
            execution_results = []
            if "tool_calls" in plan_response:
                for tool_call in plan_response["tool_calls"]:
                    tool_name = tool_call.get("tool_name")
                    tool_params = tool_call.get("parameters", {})
                    
                    if tool_name not in self.available_tools:
                        raise ValueError(f"Tool not found: {tool_name}")
                    
                    tool_func = self.available_tools[tool_name]
                    result = tool_func(**tool_params)
                    
                    execution_results.append({
                        "tool_name": tool_name,
                        "parameters": tool_params,
                        "result": result,
                    })
            
            response_content = {
                "task_description": task_desc,
                "execution_results": execution_results,
                "reasoning": plan_response.get("reasoning", ""),
                "success": True,
            }
            
            self.state.status = "done"
            self.state.last_result = response_content
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="ExecutorAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response_content,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="ExecutorAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
