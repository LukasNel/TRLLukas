from smolagents import Tool
from datetime import datetime
import yfinance as yf


class CompanyFinancialsTool(Tool):
    name = "company_financials"
    description = """
    This tool fetches financial data for a company using yfinance.
    It returns the financial statements as requested.
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
        

    def forward(self, ticker: str, statement_type: str, period: str):
        # First check if we're allowed to access this data based on cutoff date
        current_date = datetime.now()
        if self.cutoff_date and current_date > self.cutoff_date:
            raise Exception(f"Access to financial data not allowed beyond cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        # Get the stock data
        ticker_obj = yf.Ticker(ticker)
        
        # Fetch the appropriate financial statement
        if statement_type.lower() == 'income':
            if period.lower() == 'annual':
                financials = ticker_obj.income_stmt
            elif period.lower() == 'quarterly':
                financials = ticker_obj.quarterly_income_stmt
            else:
                raise ValueError("Period must be either 'annual' or 'quarterly'")
        elif statement_type.lower() == 'balance':
            if period.lower() == 'annual':
                financials = ticker_obj.balance_sheet
            elif period.lower() == 'quarterly':
                financials = ticker_obj.quarterly_balance_sheet
            else:
                raise ValueError("Period must be either 'annual' or 'quarterly'")
        elif statement_type.lower() == 'cash':
            if period.lower() == 'annual':
                financials = ticker_obj.cashflow
            elif period.lower() == 'quarterly':
                financials = ticker_obj.quarterly_cashflow
            else:
                raise ValueError("Period must be either 'annual' or 'quarterly'")
        else:
            raise ValueError("Statement type must be one of: 'income', 'balance', or 'cash'")
        
        # Convert to dictionary for easier serialization
        return financials

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