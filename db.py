# db.py  – very small wrapper
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

DB_URL = "sqlite:///agriscan_users.db"   # file in project dir
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    hash     = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)
