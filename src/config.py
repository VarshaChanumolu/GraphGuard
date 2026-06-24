"""Database connection config, loaded from environment variables.

Keep all credentials out of source control — this module only reads
from the environment (populated via a local .env file, see .env.example).
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PGConfig:
    host: str = os.getenv("PG_HOST", "localhost")
    port: str = os.getenv("PG_PORT", "5432")
    db: str = os.getenv("PG_DB", "graphguard")
    user: str = os.getenv("PG_USER", "graphguard")
    password: str = os.getenv("PG_PASSWORD", "")

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


PG = PGConfig()
