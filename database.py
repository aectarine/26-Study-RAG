import os

import psycopg
from dotenv import load_dotenv

# Git에 포함되는 개발 환경 설정을 우선 로드합니다.
load_dotenv(".env.dev")

def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5
    )
