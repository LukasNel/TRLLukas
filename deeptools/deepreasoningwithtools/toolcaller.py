from typing import AsyncGenerator
from deepreasoningwithtools.samplers.abstract import AbstractSampler
from smolagents import Tool, LocalPythonExecutor




class ToolCaller:
    def __init__(self, sampler : AbstractSampler, authorized_imports : list[str] = []):
        self.sampler = sampler
        self.authorized_imports = authorized_imports
    
    def _init_pyexp(self, tools: list[Tool] = [],):
        pyexp = LocalPythonExecutor(additional_authorized_imports=["yfinance", "pandas"])
        tool_dict = {}
        for tool in tools:
            tool_dict[tool.name] = tool
        pyexp.send_tools(tool_dict)
        tool_desc = self._generate_tool_descriptions(tool_dict)
        return pyexp, tool_desc
    def _generate_tool_descriptions(self, tools: dict) -> str:
        """Generate tool descriptions from a dictionary of tools.
        
        Args:
            tools: Dictionary of tools where keys are tool names and values are tool objects
            
        Returns:
            str: Formatted string containing tool descriptions
        """
        descriptions = []
        for tool in tools.values():
            # Generate function signature
            args = []
            for arg_name, arg_info in tool.inputs.items():
                args.append(f"{arg_name}: {arg_info['type']}")
            signature = f"def {tool.name}({', '.join(args)}) -> {tool.output_type}:"
            
            # Generate docstring
            docstring = [f'    """{tool.description}', '', '    Args:']
            for arg_name, arg_info in tool.inputs.items():
                docstring.append(f'        {arg_name}: {arg_info["description"]}')
            docstring.append('    """')
            
            # Combine into full description
            descriptions.append('\n'.join([signature] + docstring))
        
        return '\n\n'.join(descriptions)

    async def _run_code_block(self, code_block: str, pyexp: LocalPythonExecutor) -> str:
        observation = "\n```text\n"
        try:
            output, execution_logs, is_final_answer = pyexp(code_block)
            observation += "Successfully executed. Output from code block:\n"
            # observation += str(output)
            observation += str(execution_logs)
        except Exception as e:
            observation += "Failed. Please try another strategy. \n"
            if hasattr(pyexp, "state") and "_print_outputs" in pyexp.state:
                execution_logs = str(pyexp.state["_print_outputs"])
                if len(execution_logs) > 0:
                    observation += "Execution logs:\n" + execution_logs
            observation += "Exception:\n " + str(e) + "\n"
        observation += "\n```\n"
        return observation

    def find_last_message_index(self, messages: list[dict[str, str]]) -> int:
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                return i
        return -1
    
    async def _sample(self, in_messages: list[dict[str, str]], ) -> AsyncGenerator[str, str]:
        messages = in_messages
        response = self.sampler.sample(messages)
        assistant_message_index = self.find_last_message_index(messages)
        if assistant_message_index == -1:
            messages.append({"role": "assistant", "content": ""})
            assistant_message_index = len(messages) - 1
        async for msg in response:
            if msg is not None:
                messages[assistant_message_index]["content"] += msg
                input = yield msg
                if input is not None:
                    messages[assistant_message_index]["content"] += input
                    async for output in self._sample(messages):
                        yield output
                    break

    async def generate(self, system_prompt: str, user_prompt: str, tools: list[Tool] = []):
        pyexp, tool_desc = self._init_pyexp(tools)
        system_prompt = system_prompt.format(tool_desc=tool_desc)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        total_output : str = ""
        is_in_code_block = False
        code_block = ""
        gen = self._sample(messages)
        async for output in gen:
            total_output += output
            yield str(output)
            if total_output.strip().endswith("```python"):
                is_in_code_block = True
            elif total_output.strip().endswith("```") and is_in_code_block:
                execution_result = await self._run_code_block(code_block, pyexp)
                try:
                    await gen.asend(execution_result)
                    yield execution_result
                except StopAsyncIteration:
                    pass
                code_block = ""
                is_in_code_block = False
            elif is_in_code_block and output != '`' and output != '``' and output != '```':
                code_block += output
  
  