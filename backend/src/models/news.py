"""
新闻相关模型

包含新闻数据和新闻摘要的数据模型。
"""

from sqlalchemy import Column, String, Text, DECIMAL, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from .base import BaseModel


class NewsData(BaseModel):
    """新闻数据模型"""

    __tablename__ = "news_data"

    # 新闻基本信息
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(1000))
    source = Column(String(100), nullable=False)
    author = Column(String(200))

    # 相关性信息
    relevance_score = Column(DECIMAL(3, 2))  # 0.00-1.00
    related_symbols = Column(ARRAY(String))  # 相关的交易符号
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(DECIMAL(3, 2))

    # 时间信息
    published_at = Column(DateTime(timezone=True), nullable=False)
    collected_at = Column(DateTime(timezone=True), nullable=False)

    # 元数据
    language = Column(String(10), default="en")
    word_count = Column(Integer)
    hash = Column(String(64), unique=True)  # 内容哈希，避免重复

    # 关系
    news_summaries = relationship("NewsSummary", back_populates="news_data")

    def __repr__(self):
        return f"<NewsData(title='{self.title[:50]}...', source='{self.source}')>"


class NewsSummary(BaseModel):
    """新闻摘要模型"""

    __tablename__ = "news_summaries"

    # 摘要信息
    summary_text = Column(Text, nullable=False)
    key_points = Column(JSONB)
    market_impact = Column(String(20))  # high, medium, low

    # LLM生成信息
    llm_provider = Column(String(50), nullable=False)
    llm_model = Column(String(100), nullable=False)
    generation_confidence = Column(DECIMAL(3, 2))

    # 关联信息
    news_count = Column(Integer, nullable=False)  # 摘要的新闻数量
    time_period_hours = Column(Integer, nullable=False)  # 覆盖的时间段

    # 时间信息
    expires_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<NewsSummary(news_count={self.news_count}, impact='{self.market_impact}')>"