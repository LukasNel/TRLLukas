
from smolagents import Tool
from datetime import datetime
import requests


class NewsSearchTool(Tool):
    name = "news_search"
    description = """
    This tool searches for news articles based on a query and date range using the NewsAPI.
    It returns a list of relevant articles matching the search criteria.
    Will throw an exception if asked for data beyond the cutoff date.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant news articles",
        },
        "from_date": {
            "type": "string",
            "description": "The start date in format 'YYYY-MM-DD'",
            "nullable": True
        },
        "to_date": {
            "type": "string",
            "description": "The end date in format 'YYYY-MM-DD'",
            "nullable": True
        },
    }
    output_type = "object"

    def __init__(self, api_key, cutoff_date=None):
        """
        Initialize the tool with an API key and cutoff date.
        
        Args:
            api_key: NewsAPI key for authentication
            cutoff_date: A string in format 'YYYY-MM-DD' or datetime object. 
                        If provided, will prevent fetching data beyond this date.
        """
        self.api_key = api_key
        self.cutoff_date = None
        if cutoff_date:
            if isinstance(cutoff_date, str):
                self.cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
            elif isinstance(cutoff_date, datetime):
                self.cutoff_date = cutoff_date
        super().__init__()
        
    def forward(self, query: str, from_date: str = None, to_date: str = None):
        """
        Search for news articles using the NewsAPI.
        
        Args:
            query (str): The search query to find relevant news articles
            from_date (str, optional): Start date in format 'YYYY-MM-DD'
            to_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            dict: JSON response from NewsAPI containing:
                - status: Response status ("ok" or "error")
                - totalResults: Total number of results found
                - articles: List of article objects with properties:
                    - source: Source information (id, name)
                    - author: Article author
                    - title: Article title
                    - description: Article description
                    - url: URL to article
                    - publishedAt: Publication date
                    - content: Article content snippet
                    
        Raises:
            Exception: If the API request fails or if requesting data beyond cutoff date
        """
        # Check if to_date is beyond cutoff date
        if to_date and self.cutoff_date:
            request_end_date = datetime.strptime(to_date, "%Y-%m-%d")
            if request_end_date > self.cutoff_date:
                raise Exception(f"Access to news data not allowed beyond cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        # Base URL for the 'everything' endpoint
        url = "https://newsapi.org/v2/everything"
        
        # Prepare parameters
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": 5,  # Ensure we don't exceed the max
        }
        
        # Add optional parameters if provided
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        # Make the API request
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            error_info = response.json()
            raise Exception(f"News API error: {error_info.get('message', 'Unknown error')}")
            
        # Return the results
        return response.json()

