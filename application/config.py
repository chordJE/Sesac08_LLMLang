import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY  = os.environ.get('OPENAI_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
KAKAO_API_KEY   = os.environ.get('KAKAO_API_KEY')