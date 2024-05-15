import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY = os.getenv('ALPACA_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
