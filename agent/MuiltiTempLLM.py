# from __future__ import annotations
from langchain.llms.openai import BaseOpenAI
from typing import Dict,Any


class MultiTempLLM(BaseOpenAI):
    """
    Add set function to modify temperature to adapt to different usage
    """

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

    def set_temperature(self, temperature=0) -> None:
        self.temperature = temperature
        return

if __name__ == "__main__":
    llm = MultiTempLLM(engine="test-deploy",
                         model_name='gpt-3.5-turbo',
                        temperature=0.7)
