from deeptools.samplers.abstract import AbstractSampler
from typing import AsyncGenerator
from litellm import completion

class LiteLLMSampler(AbstractSampler):
    def __init__(self, model_name : str, max_output=32000):
        self.max_output = max_output
        self.model_name = model_name

    async def sample(self, messages: list[dict[str, str]], ) -> AsyncGenerator[str, None]:
        messages = messages
        response = completion(model=self.model_name, messages=messages,max_completion_tokens=self.max_output, stream=True)
        for chunk in response:
            msg = chunk["choices"][0]["delta"]["content"]
            if msg is not None:
                yield msg

