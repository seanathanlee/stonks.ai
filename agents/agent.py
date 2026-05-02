"""
Stonks.ai Agent
An AI agent that performs stock analysis using Azure OpenAI.
"""

import json
import os
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Azure OpenAI client (lazily initialised so the module can be imported
# without credentials being present at import time)
# ---------------------------------------------------------------------------

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    return _client


def _get_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ---------------------------------------------------------------------------
# Tool definitions (function calling)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Return basic information and a simulated price for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. MSFT, AAPL.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_moving_average",
            "description": "Calculate a simple moving average for a list of closing prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prices": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of historical closing prices.",
                    },
                    "window": {
                        "type": "integer",
                        "description": "The window size for the moving average.",
                        "default": 5,
                    },
                },
                "required": ["prices"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Simulated price data – in production this would call a real market data API.
_MOCK_PRICES: dict[str, dict] = {
    "MSFT": {"name": "Microsoft Corporation", "price": 415.23, "sector": "Technology"},
    "AAPL": {"name": "Apple Inc.", "price": 189.87, "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc.", "price": 175.10, "sector": "Technology"},
    "AMZN": {"name": "Amazon.com Inc.", "price": 185.50, "sector": "Consumer Discretionary"},
    "TSLA": {"name": "Tesla Inc.", "price": 172.30, "sector": "Consumer Discretionary"},
}


def get_stock_info(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker in _MOCK_PRICES:
        return _MOCK_PRICES[ticker]
    return {"error": f"Ticker '{ticker}' not found in data source."}


def calculate_moving_average(prices: list[float], window: int = 5) -> dict:
    if len(prices) < window:
        return {"error": f"Not enough data points ({len(prices)}) for window size {window}."}
    averages = [
        round(sum(prices[i : i + window]) / window, 4)
        for i in range(len(prices) - window + 1)
    ]
    return {"window": window, "moving_averages": averages, "latest": averages[-1]}


def dispatch_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments)
    if name == "get_stock_info":
        result = get_stock_info(**args)
    elif name == "calculate_moving_average":
        result = calculate_moving_average(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Stonks.ai, an expert financial analysis AI assistant. "
    "You help users analyse stocks and make sense of market data. "
    "Use the available tools to fetch stock information and perform calculations. "
    "Always provide clear, concise analysis and explain your reasoning."
)


def run_agent(user_message: str) -> str:
    """Run the agent for a single user message and return the final response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = _get_client().chat.completions.create(
            model=_get_deployment(),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Append the assistant's tool-call message
            messages.append(choice.message)

            # Execute each requested tool and append results
            for tool_call in choice.message.tool_calls:
                result = dispatch_tool(tool_call.function.name, tool_call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            # Continue the loop so the model can incorporate the results
            continue

        # finish_reason == "stop" (or similar) – return the final text
        return choice.message.content or ""


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the current price of MSFT?"
    print(run_agent(query))
