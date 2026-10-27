from datetime import datetime
from sqlalchemy import create_engine, String, Text, DateTime, JSON, Integer, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase): pass

class ServiceRequest(Base):
    __tablename__ = "service_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    telegram_user_id: Mapped[str] = mapped_column(String(64), index=True)
    compound: Mapped[str | None] = mapped_column(String(160), nullable=True)
    building: Mapped[str | None] = mapped_column(String(80), nullable=True)
    unit_number: Mapped[str | None] = mapped_column(String(80), nullable=True)
    name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(80), nullable=True)
    issue_description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(40))
    urgency: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    predictions: Mapped[dict] = mapped_column(JSON, default=dict)

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

def init_db(): Base.metadata.create_all(engine)
