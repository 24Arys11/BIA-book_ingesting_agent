"""LLM client abstraction and OpenAI-compatible implementation (LM Studio friendly)."""
from __future__ import annotations

from typing import Protocol, Optional, List, Dict

from openai import OpenAI


class IResponseGenerator(Protocol):
    def query(self, user_prompt: str, system_prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, *, temperature: Optional[float] = None, n: Optional[int] = None) -> str:
        """Generate a response from the LLM.

        Provide either a user_prompt (optionally with system_prompt) or a full messages list.
        Returns the assistant text content.
        """
        ...

    def query_n(self, user_prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None, n: int = 3) -> List[str]:
        """Generate multiple samples in one call if the backend supports it; else fall back to repeated calls."""
        ...


class OpenaiResponseGenerator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def query(self, user_prompt: str, system_prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, *, temperature: Optional[float] = None, n: Optional[int] = None) -> str:
        if messages is None:
            msgs: List[Dict[str, str]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
        else:
            msgs = messages

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if n is not None and n > 0:
            kwargs["n"] = int(n)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def query_n(self, user_prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None, n: int = 3) -> List[str]:
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                n=n,
                **({"temperature": float(temperature)} if temperature is not None else {}),
            )
            return [(c.message.content or "") for c in resp.choices]
        except Exception:
            # Fallback to repeated single-sample calls
            out: List[str] = []
            for _ in range(max(1, n)):
                out.append(self.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temperature))
            return out
