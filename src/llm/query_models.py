import os
import openai
import anthropic
from typing import Dict, Optional
import time
import random

class LLMWrapper:
    """
    Unified interface for querying different LLMs.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            
        if self.anthropic_api_key:
            self.claude = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def query(self, prompt: str, model: str, system_prompt: str = "") -> str:
        """
        Dispatches query to the appropriate provider based on model name.
        """
        if "gpt" in model:
            return self._query_openai(prompt, model, system_prompt)
        elif "claude" in model:
            return self._query_anthropic(prompt, model, system_prompt)
        elif "llama" in model or "local" in model:
            return self._query_local(prompt, model)
        else:
            raise ValueError(f"Unknown model provider for {model}")

    def _query_openai(self, prompt: str, model: str, system_prompt: str) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return ""

    def _query_anthropic(self, prompt: str, model: str, system_prompt: str) -> str:
        try:
            # Claude API structure might vary by version
            response = self.claude.messages.create(
                model=model,
                max_tokens=self.config['llm']['max_tokens'],
                temperature=self.config['llm']['temperature'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic Error: {e}")
            return ""

    def _query_local(self, prompt: str, model: str) -> str:
        # Placeholder for local model inference (e.g., via llama.cpp python bindings or HF pipeline)
        # Assuming HF pipeline or similar is set up elsewhere or this is a mock
        return "Local model response placeholder."

    def query_with_retry(self, prompt: str, model: str, system_prompt: str = "") -> str:
        retries = self.config['llm']['retry_attempts']
        backoff = self.config['llm']['backoff_factor']
        
        for i in range(retries):
            res = self.query(prompt, model, system_prompt)
            if res:
                return res
            
            sleep_time = backoff ** i
            print(f"Query failed, retrying in {sleep_time}s...")
            time.sleep(sleep_time + random.random())
            
        return ""
