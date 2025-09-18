import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    SERVER_BASE_URL = os.getenv("SERVER_BASE_URL")