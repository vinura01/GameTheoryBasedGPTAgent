from typing import Optional, Dict, Any
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class Route(BaseModel):
    kind: str  # "normal_form" | "auction" | "bargain" | "tictactoe"
    data: dict

SYSTEM = """You are a precise router for game-theory tasks.
Map the user's request into one of: normal_form, auction, bargain, tictactoe.
Return JSON with fields 'kind' and 'data'. Keep it minimal.
For normal_form, expect fields: rows, cols, A, B.
For auction: type ('vickrey' or 'first_price'), valuations (list of floats) or v (float) & n_bidders (int).
For bargain: points (list of [u1,u2]) and d (disagreement point [d1,d2]).
For tictactoe: board (list of 9 strings), player ('X' or 'O').
"""

def get_llm():
    model = os.getenv("OLLAMA_MODEL", "mistral:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return ChatOllama(model=model, base_url=base_url, temperature=0.0)

def route_text(query: str) -> Route:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("user", "{q}")
    ])
    chain = prompt | llm
    resp = chain.invoke({"q": query}).content
    # Try to locate JSON
    import json, re
    m = re.search(r"\{.*\}", resp, re.S)
    if m:
        obj = json.loads(m.group(0))
        return Route(**obj)
    raise ValueError(f"Router returned unparseable response: {resp}")
