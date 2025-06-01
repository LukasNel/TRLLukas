from smolagents import Tool
from datetime import datetime
import yfinance as yf
import pandas as pd

class CompanyFinancialsTool(Tool):
    name = "company_financials"
    description = """
    This tool fetches financial data for a company using yfinance.
    It returns the financial statements as requested.
    The fetch_before_date is the date before which to fetch the financial data.
    Will throw an exception if asked for data beyond the cutoff date.
    """
    inputs = {
        "ticker": {
            "type": "string",
            "description": "The stock ticker symbol of the company (e.g., 'AAPL' for Apple)",
        },
        "statement_type": {
            "type": "string",
            "description": "Type of financial statement to fetch: 'income', 'balance', or 'cash'",
        },
        "period": {
            "type": "string",
            "description": "Period of the financial data: 'annual' or 'quarterly'",
        },
        "fetch_before_date": {
            "type": "string",
            "description": "The date before which to fetch the financial data in format 'YYYY-MM-DD'",
        }
    }
    output_type = "object"

    def __init__(self, cutoff_date=None):
        """
        Initialize the tool with a cutoff date.
        
        Args:
            cutoff_date: A string in format 'YYYY-MM-DD' or datetime object. 
                         If provided, will prevent fetching data beyond this date.
        """
        self.cutoff_date = None
        if cutoff_date:
            if isinstance(cutoff_date, str):
                self.cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
            elif isinstance(cutoff_date, datetime):
                self.cutoff_date = cutoff_date
        super().__init__()
        

    def forward(self, ticker: str, statement_type: str, period: str, fetch_before_date: str,):
        # Convert string dates to datetime objects
        fetch_before = datetime.strptime(fetch_before_date, "%Y-%m-%d")
        
        # Check if we're allowed to access this data based on cutoff date
        if self.cutoff_date and fetch_before > self.cutoff_date:
            raise ValueError(f"Cannot fetch data beyond cutoff date {self.cutoff_date}")

        # Get company info
        stock = yf.Ticker(ticker)
        
        # Map statement type to yfinance method
        statement_methods = {
            'income': stock.income_stmt,
            'balance': stock.balance_sheet,
            'cash': stock.cashflow
        }
        
        if statement_type not in statement_methods:
            raise ValueError(f"Invalid statement type. Must be one of: {', '.join(statement_methods.keys())}")
            
        # Get the financial statement
        statement = statement_methods[statement_type]
        
        # Get data with specified period
        if period == 'annual':
            data = statement.yearly
        elif period == 'quarterly':
            data = statement.quarterly
        else:
            raise ValueError("Period must be either 'annual' or 'quarterly'")
            
        # Convert index to datetime if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Filter data before fetch_before date and get the most recent
        filtered_data = data.loc[data.index <= fetch_before]
        
        if filtered_data.empty:
            raise ValueError(f"No financial data available before {fetch_before_date}")
            
        # Get the most recent statement before the fetch_before date
        latest_statement = filtered_data.iloc[-1]
        
        # Convert to dictionary and handle any non-serializable types
        return latest_statement

class StockPriceTool(Tool):
    name = "stock_price"
    description = """
    This tool fetches the historical stock price data for a company using yfinance.
    Will throw an exception if asked for data beyond the cutoff date.
    """
    inputs = {
        "ticker": {
            "type": "string",
            "description": "The stock ticker symbol of the company (e.g., 'AAPL' for Apple)",
        },
        "start_date": {
            "type": "string",
            "description": "The start date in format 'YYYY-MM-DD'",
        },
        "end_date": {
            "type": "string",
            "description": "The end date in format 'YYYY-MM-DD'",
        },
        "interval": {
            "type": "string", 
            "description": "The data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly)",
            "default": "1d",
            "nullable": True
        }
    }
    output_type = "object"

    def __init__(self, cutoff_date=None):
        """
        Initialize the tool with a cutoff date.
        
        Args:
            cutoff_date: A string in format 'YYYY-MM-DD' or datetime object. 
                         If provided, will prevent fetching data beyond this date.
        """
        self.cutoff_date = None
        if cutoff_date:
            if isinstance(cutoff_date, str):
                self.cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d")
            elif isinstance(cutoff_date, datetime):
                self.cutoff_date = cutoff_date
        super().__init__()

    def forward(self, ticker: str, start_date: str, end_date: str, interval: str = "1d"):
        # Convert string dates to datetime objects
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if we're allowed to access this data based on cutoff date
        if self.cutoff_date and end > self.cutoff_date:
            raise Exception(f"Access to stock price data not allowed beyond cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        # Fetch the stock price data
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        # Convert to dictionary for easier serialization
        return data