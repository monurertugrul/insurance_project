import os
import json
import re
from openai import OpenAI

SYSTEM_PROMPT_FRONTIER = """
You are an expert medical insurance pricing assistant.

You receive:
- A JSON object with policyholder features.
- Retrieved similar historical cases (if available).

Your job:
1. Use the features and retrieved cases to estimate medical insurance charges in USD.
2. Provide a short, clear explanation referencing patterns in the retrieved cases.
3. Be conservative and realistic; typical charges range from 1000 to 50000 USD.

You MUST respond with ONLY this JSON object:

{
  "predicted_charges_usd": <float>,
  "explanation": "<short explanation>"
}
"""

class FrontierAgent:
    def __init__(self, api_key=None, model="meta-llama/llama-3.2-3b-instruct", rag=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.rag = rag

    def _build_prompt(self, features, similar_cases):
        rag_context = ""
        if similar_cases:
            rag_context = "\nRetrieved similar cases:\n" + "\n".join(
                f"- {case}" for case in similar_cases
            )

        return (
            SYSTEM_PROMPT_FRONTIER
            + rag_context
            + "\n\nUser input JSON:\n"
            + json.dumps(features)
            + "\n\nReturn ONLY the JSON object."
        )

    def price(self, features: dict):
        similar_cases = self.rag.retrieve(features, k=3) if self.rag else []
        prompt = self._build_prompt(features, similar_cases)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        text = resp.choices[0].message.content

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {
                "predicted_charges_usd": None,
                "explanation": "Model did not return valid JSON."
            }

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "predicted_charges_usd": None,
                "explanation": "Model returned invalid JSON."
            }
