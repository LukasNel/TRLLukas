from deeptools.toolcaller import ToolCaller
from deeptools.tools.yfinance_tools import StockPriceTool, CompanyFinancialsTool
from datetime import datetime
import asyncio
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are an expert assistant. You will be given a task to solve as best you can. 
You have access to a python interpreter and a set of tools that runs anything you write in a code block.
You have access to pandas and yfinance. 
All code blocks written between ```python and ``` will get executed by a python interpreter and the result will be given to you.
On top of performing computations in the Python code snippets that you create, you only have access to these tools:
{tool_desc}
"""

class TestModalToolCaller:
    def __init__(self, vllm_model_id: str | None = None, litellm_model_name: str | None = None):
        self.vllm_model_id = vllm_model_id
        self.litellm_model_name = litellm_model_name
        self.test_queries = [
            "What was Apple's stock price last week?",
            "Compare Tesla and Microsoft's financial performance over the last quarter",
            "What is the current market cap of Amazon?",
            "Analyze the stock price trends of NVIDIA over the past month",
            "Which tech company has the highest P/E ratio: Apple, Microsoft, or Google?"
        ]
        
    async def run_test_query(self, toolcaller: ToolCaller, query: str, tools: List[Any], authorized_imports: List[str] = ["pandas"]) -> Dict[str, Any]:
        """Run a single test query and collect the response."""
        toolcaller.authorized_imports = authorized_imports
        
        response = []
        async for output in toolcaller.generate(
            user_prompt=query,
            system_prompt=SYSTEM_PROMPT,
            tools=tools
        ):
            response.append(output)
        
        return {
            "query": query,
            "response": "".join(response),
            "success": any("error" not in r.lower() for r in response)
        }

    async def test_litellm_toolcaller(self) -> List[Dict[str, Any]]:
        """Run comprehensive tests for LiteLLM toolcaller."""
        from deeptools.samplers.litellm_sampler import LiteLLMSampler
        if not self.litellm_model_name:
            return []
            
        toolcaller = ToolCaller(
            sampler=LiteLLMSampler(model_name=self.litellm_model_name),
            authorized_imports=["pandas"]
        )
        
        tools = [
            StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d")),
            CompanyFinancialsTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))
        ]
        
        results = []
        for query in self.test_queries:
            result = await self.run_test_query(toolcaller, query, tools)
            results.append(result)
            
        return results

    async def test_vllm_toolcaller(self) -> List[Dict[str, Any]]:
        from deeptools.samplers.vllm.sampler import VLLMSampler

        """Run comprehensive tests for vLLM toolcaller."""
        if not self.vllm_model_id:
            return []
            
        toolcaller = ToolCaller(
            sampler=VLLMSampler(model_id=self.vllm_model_id),
            authorized_imports=["pandas"]
        )
        
        tools = [
            StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d")),
            CompanyFinancialsTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))
        ]
        
        results = []
        for query in self.test_queries:
            result = await self.run_test_query(toolcaller, query, tools)
            results.append(result)
            
        return results

    async def test_stock_comparison(self) -> Dict[str, Any]:
        """Run a specific test comparing Apple and Tesla stock performance."""
        from deeptools.samplers.litellm_sampler import LiteLLMSampler
        
        if not self.litellm_model_name:
            return {"query": "Stock comparison test", "response": "No model specified", "success": False}
            
        toolcaller = ToolCaller(
            sampler=LiteLLMSampler(model_name=self.litellm_model_name),
            authorized_imports=["pandas", "yfinance"]
        )
        
        tools = [
            StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d")),
            CompanyFinancialsTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))
        ]
        
        query = """Compare Apple (AAPL) and Tesla (TSLA) stock performance over the last year. 
        Consider factors like price movement, volatility, and key financial metrics. 
        Which stock would you recommend for a long-term investment and why?"""
        
        return await self.run_test_query(toolcaller, query, tools, authorized_imports=["pandas", "yfinance"])

    def print_test_results(self, results: List[Dict[str, Any]], model_type: str):
        """Print formatted test results."""
        print(f"\n=== {model_type} Test Results ===")
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}:")
            print(f"Query: {result['query']}")
            print(f"Success: {'✅' if result['success'] else '❌'}")
            print(f"Response: {result['response'][:200]}...")  # Print first 200 chars
        print("\nSummary:")
        success_count = sum(1 for r in results if r['success'])
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        print(f"Success Rate: {(success_count/len(results))*100:.2f}%") 