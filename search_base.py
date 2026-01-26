import os
import pprint
from dotenv import load_dotenv  

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

from langchain_community.utilities import GoogleSerperAPIWrapper

search = GoogleSerperAPIWrapper()

res=search.run("Latest news about rohit sharma")
print(res)