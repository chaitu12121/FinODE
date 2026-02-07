"""
TemporalReasoningAgent

Stub implementation of Neural ODE for continuous-time forecasting.
Provides deterministic interface for time-series analysis.
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
from .base_agent import BaseAgent, AgentMessage, AgentState


class SimpleODESolver:
    """
    Simple deterministic ODE solver for time-series.
    Stub implementation - can be replaced with torchdiffeq in production.
    
    Model: dy/dt = -k*y + trend + noise
    """
    
    def __init__(self, initial_state: float = 0.0):
        """Initialize ODE solver"""
        self.initial_state = initial_state
        self.k = 0.5  # Decay constant
        self.trend = 0.01  # Linear trend
    
    def solve(
        self,
        t_span: Tuple[float, float],
        t_eval: List[float],
        parameters: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Solve ODE: dy/dt = -k*y + trend
        Uses simple Euler method (deterministic).
        
        Args:
            t_span: (t_start, t_end)
            t_eval: Time points to evaluate
            parameters: Optional {"k": value, "trend": value}
        
        Returns:
            Solution at t_eval points
        """
        if parameters:
            self.k = parameters.get("k", self.k)
            self.trend = parameters.get("trend", self.trend)
        
        y = np.zeros(len(t_eval))
        y[0] = self.initial_state
        
        # Euler method
        for i in range(1, len(t_eval)):
            dt = t_eval[i] - t_eval[i-1]
            dydt = -self.k * y[i-1] + self.trend
            y[i] = y[i-1] + dydt * dt
        
        return y


class TemporalReasoningAgent(BaseAgent):
    """
    Temporal sequence modeling using ODE framework.
    
    Responsibilities:
    - Time-series forecasting
    - Trend decomposition
    - Continuous-time flow modeling
    - Confidence intervals
    """
    
    def __init__(self, groq_client):
        """
        Initialize TemporalReasoningAgent.
        
        Args:
            groq_client: GroqClient instance
        """
        super().__init__("TemporalReasoningAgent", groq_client)
        self.ode_solver = SimpleODESolver()
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Perform temporal reasoning.
        
        Args:
            message: Message with content:
              - "timeseries": List of (time, value) tuples
              - "forecast_horizon": Future time steps
              - "task": "forecast" | "decompose" | "trend"
        
        Returns:
            Response with temporal analysis
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            timeseries = message.content.get("timeseries", [])
            forecast_horizon = message.content.get("forecast_horizon", 10)
            task = message.content.get("task", "forecast")
            
            if not timeseries:
                raise ValueError("No timeseries provided")
            
            # Parse timeseries
            times, values = zip(*timeseries)
            times = np.array(times, dtype=float)
            values = np.array(values, dtype=float)
            
            if task == "forecast":
                result = self._forecast(times, values, forecast_horizon)
            elif task == "decompose":
                result = self._decompose(times, values)
            elif task == "trend":
                result = self._extract_trend(times, values)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            self.state.status = "done"
            self.state.last_result = result
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="TemporalReasoningAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=result,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="TemporalReasoningAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _forecast(self, times: np.ndarray, values: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Forecast future values using ODE"""
        # Estimate initial state from data
        initial_state = values[-1]
        self.ode_solver.initial_state = initial_state
        
        # Setup ODE parameters
        t_start = float(times[-1])
        t_end = float(times[-1]) + horizon
        
        # Generate future time points
        future_times = np.linspace(t_start, t_end, horizon + 1)[1:]
        
        # Solve ODE
        forecast = self.ode_solver.solve((t_start, t_end), future_times)
        
        # Add confidence intervals (simple ±10%)
        confidence = float(np.std(values) * 0.1)
        
        return {
            "task": "forecast",
            "timeseries_length": len(values),
            "forecast_horizon": horizon,
            "forecast_times": future_times.tolist(),
            "forecast_values": forecast.tolist(),
            "confidence_interval": {
                "upper": (forecast + confidence).tolist(),
                "lower": (forecast - confidence).tolist(),
            },
            "algorithm": "ODE (Euler solver)",
        }
    
    def _decompose(self, times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Decompose series into trend + residual"""
        # Simple trend extraction
        trend = np.polyfit(times, values, 1)
        trend_line = np.polyval(trend, times)
        residual = values - trend_line
        
        return {
            "task": "decompose",
            "trend": {
                "slope": float(trend[0]),
                "intercept": float(trend[1]),
                "values": trend_line.tolist(),
            },
            "residual": residual.tolist(),
            "trend_strength": float(1 - np.var(residual) / np.var(values)),
        }
    
    def _extract_trend(self, times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """Extract trend component"""
        # Simple moving average
        window = max(3, len(values) // 5)
        kernel = np.ones(window) / window
        trend = np.convolve(values, kernel, mode="same")
        
        return {
            "task": "trend_extraction",
            "window_size": window,
            "trend": trend.tolist(),
            "trend_direction": "increasing" if np.polyfit(times, trend, 1)[0] > 0 else "decreasing",
        }
