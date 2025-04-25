from typing import AsyncGenerator, Generator
from litellm import completion 
import abc

class AbstractSampler:
    @abc.abstractmethod
    async def sample(self, messages: list[dict[str, str]], ) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Subclasses must implement this method")
        yield ""
