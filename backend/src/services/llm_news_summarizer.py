"""
LLM新闻摘要服务

负责使用大语言模型生成新闻摘要和关键信息提取。
"""

import logging
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from datetime import datetime

from ..models.news import NewsData, NewsSummary
from ..core.logging import BusinessLogger
from ..core.llm_integration import LLMIntegrationService
from ..core.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("llm_news_summarizer")


class LLMNewsSummarizer:
    """LLM新闻摘要生成器"""

    def __init__(self, llm_service: Optional[LLMIntegrationService] = None):
        self.logger = logger
        self.business_logger = business_logger
        self.llm_service = llm_service or LLMIntegrationService()

        # 摘要生成配置
        self.max_summary_length = 500
        self.min_content_length = 50  # 最小内容长度才进行摘要
        self.batch_size = 5  # 批处理大小
        self.llm_model = "gpt-3.5-turbo"  # 默认模型
        self.temperature = 0.3  # 较低的温度确保一致性
        self.max_tokens = 800  # 最大token数

        # 摘要提示词模板
        self.summary_prompt_template = """
你是一个专业的加密货币新闻分析师。请对以下新闻进行简洁、准确的摘要，提取关键信息。

新闻标题: {title}
新闻来源: {source}
发布时间: {published_at}
新闻内容: {content}

请提供以下格式的摘要：
1. **核心摘要**: 2-3句话概括新闻要点
2. **关键影响**: 对加密货币市场的影响
3. **情绪倾向**: positive/negative/neutral
4. **相关币种**: 提及的加密货币（如有）
5. **重要程度**: high/medium/low

请以JSON格式回复：
{{
    "core_summary": "核心摘要内容",
    "key_impact": "对市场的影响分析",
    "sentiment": "positive/negative/neutral",
    "mentioned_cryptos": ["BTC", "ETH", ...],
    "importance_level": "high/medium/low",
    "confidence_score": 0.85
}}
"""

        # 批量摘要提示词
        self.batch_summary_prompt = """
你是一个专业的加密货币市场分析师。请对以下多条新闻进行综合分析，提供市场整体观点。

{news_items}

请提供：
1. **市场总体情绪**: positive/negative/neutral/mixed
2. **关键主题**: 提取2-3个主要讨论主题
3. **重要影响**: 对市场的整体影响分析
4. **关注币种**: 最多提及的5个加密货币
5. **风险评估**: 当前市场风险水平 low/medium/high

JSON格式回复：
{{
    "overall_sentiment": "市场总体情绪",
    "key_themes": ["主题1", "主题2", "主题3"],
    "market_impact": "整体影响分析",
    "top_mentioned_cryptos": ["BTC", "ETH", ...],
    "risk_assessment": "low/medium/high",
    "news_count": 处理的新闻数量
}}
"""

    async def generate_single_summary(self, news: NewsData) -> Optional[NewsSummary]:
        """为单条新闻生成摘要"""
        try:
            # 验证输入
            if not self._validate_news_content(news):
                self.logger.warning(f"新闻内容不充分，跳过摘要: {news.id}")
                return None

            # 构建提示词
            prompt = self._build_summary_prompt(news)

            # 调用LLM
            response = await self.llm_service.generate_completion(
                prompt=prompt,
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 解析响应
            summary_data = self._parse_summary_response(response)
            if not summary_data:
                self.logger.error(f"无法解析LLM响应: {news.id}")
                return None

            # 创建摘要对象
            summary = NewsSummary(
                news_id=news.id,
                core_summary=summary_data.get("core_summary", ""),
                key_impact=summary_data.get("key_impact", ""),
                sentiment=summary_data.get("sentiment", "neutral"),
                mentioned_cryptos=summary_data.get("mentioned_cryptos", []),
                importance_level=summary_data.get("importance_level", "medium"),
                confidence_score=summary_data.get("confidence_score", 0.5),
                processing_time=datetime.utcnow(),
                llm_model=self.llm_model
            )

            self.business_logger.log_system_event(
                event_type="single_summary_generated",
                severity="info",
                message=f"生成单条新闻摘要: {news.title[:50]}...",
                details={
                    "news_id": news.id,
                    "sentiment": summary.sentiment,
                    "importance": summary.importance_level,
                    "confidence": summary.confidence_score
                }
            )

            return summary

        except LLMError as e:
            self.logger.error(f"LLM API错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"生成摘要失败: {e}")
            self.business_logger.log_system_event(
                event_type="summary_generation_failed",
                severity="error",
                message=f"摘要生成失败: {str(e)}",
                details={"news_id": news.id, "error": str(e)}
            )
            return None

    async def generate_batch_summaries(self, news_list: List[NewsData]) -> List[NewsSummary]:
        """批量生成新闻摘要"""
        try:
            if not news_list:
                return []

            # 过滤有效内容
            valid_news = [news for news in news_list if self._validate_news_content(news)]
            if not valid_news:
                self.logger.warning("没有有效的新闻内容进行摘要")
                return []

            summaries = []

            # 分批处理
            for i in range(0, len(valid_news), self.batch_size):
                batch = valid_news[i:i + self.batch_size]
                batch_summaries = await self._process_batch(batch)
                summaries.extend(batch_summaries)

                # 添加延迟避免API限制
                if i + self.batch_size < len(valid_news):
                    await asyncio.sleep(0.5)

            self.business_logger.log_system_event(
                event_type="batch_summary_completed",
                severity="info",
                message=f"批量摘要完成，处理 {len(valid_news)} 条新闻",
                details={
                    "processed_count": len(valid_news),
                    "summary_count": len(summaries),
                    "success_rate": len(summaries) / len(valid_news) if valid_news else 0
                }
            )

            return summaries

        except Exception as e:
            self.logger.error(f"批量摘要生成失败: {e}")
            raise

    async def generate_market_overview(self, news_list: List[NewsData]) -> Optional[Dict[str, Any]]:
        """生成市场整体概览"""
        try:
            if len(news_list) < 3:
                self.logger.warning("新闻数量不足，无法生成市场概览")
                return None

            # 构建批量新闻文本
            news_items_text = ""
            for i, news in enumerate(news_list[:10], 1):  # 最多处理10条
                news_items_text += f"""
新闻 {i}:
标题: {news.title}
来源: {news.source}
时间: {news.published_at}
内容: {news.content[:500]}...
"""

            # 替换模板中的占位符
            prompt = self.batch_summary_prompt.format(news_items=news_items_text)

            # 调用LLM
            response = await self.llm_service.generate_completion(
                prompt=prompt,
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 解析响应
            overview_data = self._parse_overview_response(response)
            if not overview_data:
                return None

            # 添加元数据
            overview_data.update({
                "generated_at": datetime.utcnow().isoformat(),
                "news_count": len(news_list),
                "time_range": {
                    "start": min(news.published_at for news in news_list if news.published_at).isoformat(),
                    "end": max(news.published_at for news in news_list if news.published_at).isoformat()
                }
            })

            self.business_logger.log_system_event(
                event_type="market_overview_generated",
                severity="info",
                message=f"生成市场概览，基于 {len(news_list)} 条新闻",
                details={
                    "overall_sentiment": overview_data.get("overall_sentiment"),
                    "risk_level": overview_data.get("risk_assessment"),
                    "key_themes_count": len(overview_data.get("key_themes", []))
                }
            )

            return overview_data

        except Exception as e:
            self.logger.error(f"生成市场概览失败: {e}")
            return None

    async def extract_key_entities(self, news: NewsData) -> Dict[str, List[str]]:
        """提取新闻中的关键实体"""
        try:
            if not self._validate_news_content(news):
                return {}

            prompt = f"""
从以下新闻中提取关键实体信息：

标题: {news.title}
内容: {news.content}

请提取并以JSON格式返回：
{{
    "cryptocurrencies": ["BTC", "ETH", ...],
    "companies": ["公司名称", ...],
    "people": ["人物姓名", ...],
    "technologies": ["技术名称", ...],
    "events": ["事件名称", ...]
}}
"""

            response = await self.llm_service.generate_completion(
                prompt=prompt,
                model=self.llm_model,
                temperature=0.1,  # 更低的温度确保准确性
                max_tokens=300
            )

            entities = self._parse_entities_response(response)
            return entities or {}

        except Exception as e:
            self.logger.error(f"实体提取失败: {e}")
            return {}

    def _validate_news_content(self, news: NewsData) -> bool:
        """验证新闻内容是否适合生成摘要"""
        if not news.content or len(news.content.strip()) < self.min_content_length:
            return False

        # 检查是否为重复或无意义内容
        content = news.content.lower().strip()
        if content in ["", "n/a", "null", "undefined"]:
            return False

        return True

    def _build_summary_prompt(self, news: NewsData) -> str:
        """构建单条新闻的摘要提示词"""
        return self.summary_prompt_template.format(
            title=news.title or "",
            source=news.source or "",
            published_at=news.published_at.isoformat() if news.published_at else "",
            content=news.content or ""
        )

    def _parse_summary_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析摘要响应"""
        try:
            # 尝试解析JSON响应
            if "```json" in response:
                # 提取JSON代码块
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # 直接解析整个响应
                json_str = response.strip()

            data = json.loads(json_str)

            # 验证必需字段
            required_fields = ["core_summary", "sentiment", "importance_level"]
            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"摘要响应缺少必需字段: {field}")
                    data[field] = "" if field == "core_summary" else "neutral" if field == "sentiment" else "medium"

            # 验证情绪值
            valid_sentiments = ["positive", "negative", "neutral"]
            if data.get("sentiment") not in valid_sentiments:
                data["sentiment"] = "neutral"

            # 验证重要程度
            valid_importance = ["high", "medium", "low"]
            if data.get("importance_level") not in valid_importance:
                data["importance_level"] = "medium"

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 响应: {response[:200]}...")
            return None
        except Exception as e:
            self.logger.error(f"摘要响应解析失败: {e}")
            return None

    def _parse_overview_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析市场概览响应"""
        return self._parse_summary_response(response)  # 使用相同的解析逻辑

    def _parse_entities_response(self, response: str) -> Optional[Dict[str, List[str]]]:
        """解析实体提取响应"""
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)

            # 确保所有字段都是列表
            entities = {}
            for key, value in data.items():
                if isinstance(value, list):
                    entities[key] = [str(item) for item in value if item]
                elif isinstance(value, str):
                    entities[key] = [value] if value else []
                else:
                    entities[key] = []

            return entities

        except Exception as e:
            self.logger.error(f"实体响应解析失败: {e}")
            return None

    async def _process_batch(self, batch: List[NewsData]) -> List[NewsSummary]:
        """处理一批新闻摘要"""
        tasks = []
        for news in batch:
            task = self.generate_single_summary(news)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"批处理中第{i+1}条新闻摘要失败: {result}")
            elif result:
                summaries.append(result)

        return summaries


# 便捷函数
async def summarize_news(news_list: List[NewsData], **kwargs) -> List[NewsSummary]:
    """批量摘要新闻的便捷函数"""
    summarizer = LLMNewsSummarizer()
    return await summarizer.generate_batch_summaries(news_list, **kwargs)


async def generate_market_summary(news_list: List[NewsData], **kwargs) -> Optional[Dict[str, Any]]:
    """生成市场摘要的便捷函数"""
    summarizer = LLMNewsSummarizer()
    return await summarizer.generate_market_overview(news_list, **kwargs)