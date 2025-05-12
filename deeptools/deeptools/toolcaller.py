from typing import AsyncGenerator
from deeptools.samplers.abstract import AbstractSampler
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

    async def _sample(self, messages: list[dict[str, str]], ) -> AsyncGenerator[str, str]:
        messages = messages
        response = self.sampler.sample(messages)
        assistant_message = ""
        async for msg in response:
            if msg is not None:
                assistant_message += msg
                input = yield msg
                if input is not None:
                    assistant_message += input
                    find_last_message = [message for message in messages if message["role"] == "assistant"]
                    if len(find_last_message) > 0:
                        find_last_message[0]["content"] += assistant_message
                    else:
                        messages.append({"role": "assistant", "content": assistant_message})
                    async for output in self._sample(messages):
                        yield output
                    break

    async def generate(self, system_prompt: str, user_prompt: str, tools: list[Tool] = []):
        pyexp, tool_desc = self._init_pyexp(tools)
        system_prompt = system_prompt.format(tool_desc=tool_desc)
        print(system_prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        total_output : str = ""
        is_in_code_block = False
        code_block = ""
        gen = self._sample(messages)
        async for output in gen:
            total_output += output
            if total_output.strip().endswith("```python"):
                is_in_code_block = True
                print("in code block")
            elif total_output.strip().endswith("```") and is_in_code_block:
                print("out code block")
                # print(code_block)
                is_in_code_block = False
                try:
                    output, execution_logs, is_final_answer = pyexp(code_block)
                    observation = "\n```text\nSuccessfully executed. Output from code block:\n" + str(execution_logs) + "\n```"
                    # print(observation)
                except Exception as e:
                    observation = "```text\n"
                    if hasattr(pyexp, "state") and "_print_outputs" in pyexp.state:
                        execution_logs = str(pyexp.state["_print_outputs"])
                        if len(execution_logs) > 0:
                            observation += execution_logs
                    observation += "Failed. Please try another strategy. " + str(e) + "\n```"
                try:
                    await gen.asend(observation)
                except StopAsyncIteration:
                    pass
                if observation:
                    yield observation
                code_block = ""
            elif is_in_code_block and output != '`' and output != '``' and output != '```':
                code_block += output
            yield str(output)
  
  