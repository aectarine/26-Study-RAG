import os

import psycopg
from dotenv import load_dotenv

# Uvicorn의 --env-file로 주입된 환경변수를 사용합니다.
# 직접 실행할 때는 기본 .env 파일을 보조적으로 읽습니다.
load_dotenv()

def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5
    )
