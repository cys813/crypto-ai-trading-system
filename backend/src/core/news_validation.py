"""
新闻数据验证模块

提供新闻数据的验证、清洗和标准化功能。
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib

from .exceptions import NewsValidationError, ValidationError
from .config import settings

logger = logging.getLogger(__name__)


class NewsValidator:
    """新闻数据验证器"""

    def __init__(self):
        self.logger = logger

        # 配置参数
        self.min_title_length = 10
        self.max_title_length = 200
        self.min_content_length = 50
        self.max_content_length = 10000
        self.max_url_length = 500
        self.max_source_length = 100

        # 可信新闻源（正则表达式模式）
        self.trusted_sources_patterns = [
            r'.*coindesk.*',
            r'.*cointelegraph.*',
            r'.*theblock.*',
            r'.*decrypt.*',
            r'.*cryptonews.*',
            r'.*reuters.*',
            r'.*bloomberg.*',
            r'.*coinbase.*blog.*',
            r'.*binance.*research.*'
        ]

        # 垃圾内容关键词
        self.spam_keywords = [
            'click here', 'buy now', 'limited time', 'guaranteed profit',
            '100% return', 'get rich quick', 'pump and dump', 'scam',
            'fake news', 'misleading', 'advertisement', 'sponsored'
        ]

        # 无效URL模式
        self.invalid_url_patterns = [
            r'.*\.bit\..*',
            r'.*\.xyz/.*',
            r'.*short\.link/.*',
            r'.*tinyurl\..*',
            r'.*bit\.ly/.*'
        ]

        # 新闻去重配置
        self.duplicate_similarity_threshold = 0.9  # 相似度阈值

    def validate_news_data(self, news_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证新闻数据

        Args:
            news_data: 新闻数据字典

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误消息列表)
        """
        errors = []

        try:
            # 验证必需字段
            self._validate_required_fields(news_data, errors)

            # 验证标题
            self._validate_title(news_data.get('title', ''), errors)

            # 验证内容
            self._validate_content(news_data.get('content', ''), errors)

            # 验证URL
            self._validate_url(news_data.get('url', ''), errors)

            # 验证新闻源
            self._validate_source(news_data.get('source', ''), errors)

            # 验证时间
            self._validate_publish_time(news_data.get('published_at'), errors)

            # 验证相关性分数
            self._validate_relevance_score(news_data.get('relevance_score'), errors)

            # 检查垃圾内容
            self._check_spam_content(news_data, errors)

            # 验证数据一致性
            self._validate_data_consistency(news_data, errors)

        except Exception as e:
            self.logger.error(f"新闻验证过程中发生错误: {e}")
            errors.append(f"验证过程异常: {str(e)}")

        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f"新闻验证失败: {errors}")

        return is_valid, errors

    def _validate_required_fields(self, news_data: Dict[str, Any], errors: List[str]) -> None:
        """验证必需字段"""
        required_fields = ['title', 'content', 'source']

        for field in required_fields:
            if not news_data.get(field):
                errors.append(f"缺少必需字段: {field}")

    def _validate_title(self, title: str, errors: List[str]) -> None:
        """验证标题"""
        if not title:
            errors.append("标题不能为空")
            return

        title = title.strip()

        if len(title) < self.min_title_length:
            errors.append(f"标题长度不足，至少需要 {self.min_title_length} 个字符")

        if len(title) > self.max_title_length:
            errors.append(f"标题长度过长，最多允许 {self.max_title_length} 个字符")

        # 检查标题是否包含垃圾内容
        if any(keyword.lower() in title.lower() for keyword in self.spam_keywords):
            errors.append("标题包含垃圾内容关键词")

    def _validate_content(self, content: str, errors: List[str]) -> None:
        """验证内容"""
        if not content:
            errors.append("内容不能为空")
            return

        content = content.strip()

        if len(content) < self.min_content_length:
            errors.append(f"内容长度不足，至少需要 {self.min_content_length} 个字符")

        if len(content) > self.max_content_length:
            errors.append(f"内容长度过长，最多允许 {self.max_content_length} 个字符")

        # 检查内容质量
        if self._is_low_quality_content(content):
            errors.append("内容质量过低，可能是重复或无意义内容")

    def _validate_url(self, url: str, errors: List[str]) -> None:
        """验证URL"""
        if not url:
            return  # URL是可选的

        url = url.strip()

        if len(url) > self.max_url_length:
            errors.append(f"URL长度过长，最多允许 {self.max_url_length} 个字符")

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append("URL格式无效")
        except Exception:
            errors.append("URL格式无效")
            return

        # 检查是否为无效URL模式
        for pattern in self.invalid_url_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                errors.append("URL来自不可信的短链接服务")
                break

    def _validate_source(self, source: str, errors: List[str]) -> None:
        """验证新闻源"""
        if not source:
            errors.append("新闻源不能为空")
            return

        source = source.strip()

        if len(source) > self.max_source_length:
            errors.append(f"新闻源名称过长，最多允许 {self.max_source_length} 个字符")

        # 检查新闻源是否包含垃圾内容
        if any(keyword.lower() in source.lower() for keyword in self.spam_keywords):
            errors.append("新闻源名称包含垃圾内容关键词")

    def _validate_publish_time(self, publish_time: Any, errors: List[str]) -> None:
        """验证发布时间"""
        if not publish_time:
            return  # 发布时间是可选的

        try:
            if isinstance(publish_time, str):
                # 尝试解析字符串时间
                parsed_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            elif isinstance(publish_time, datetime):
                parsed_time = publish_time
            else:
                errors.append("发布时间格式无效")
                return

            # 检查时间是否合理（不能太旧或太新）
            now = datetime.utcnow()
            max_past_days = 30  # 最多30天前的新闻
            max_future_hours = 1  # 最多1小时后的新闻

            if parsed_time < now - timedelta(days=max_past_days):
                errors.append(f"新闻发布时间过旧，超过 {max_past_days} 天")

            if parsed_time > now + timedelta(hours=max_future_hours):
                errors.append(f"新闻发布时间过新，超过未来 {max_future_hours} 小时")

        except Exception as e:
            errors.append(f"发布时间解析失败: {str(e)}")

    def _validate_relevance_score(self, score: Any, errors: List[str]) -> None:
        """验证相关性分数"""
        if score is None:
            return  # 相关性分数是可选的

        try:
            score_float = float(score)
            if not 0.0 <= score_float <= 1.0:
                errors.append("相关性分数必须在0.0到1.0之间")
        except (ValueError, TypeError):
            errors.append("相关性分数必须是有效的数字")

    def _check_spam_content(self, news_data: Dict[str, Any], errors: List[str]) -> None:
        """检查垃圾内容"""
        title = news_data.get('title', '').lower()
        content = news_data.get('content', '').lower()
        source = news_data.get('source', '').lower()

        combined_text = f"{title} {content} {source}"

        # 检查垃圾关键词
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in combined_text)
        if spam_count > 2:  # 如果包含超过2个垃圾关键词
            errors.append("内容包含过多垃圾关键词，可能是垃圾信息")

        # 检查重复字符
        if self._has_excessive_repetition(combined_text):
            errors.append("内容包含过多重复字符，质量较低")

        # 检查是否全大写（可能是标题党）
        if title.isupper() and len(title) > 20:
            errors.append("标题全部大写，可能是标题党")

    def _validate_data_consistency(self, news_data: Dict[str, Any], errors: List[str]) -> None:
        """验证数据一致性"""
        title = news_data.get('title', '')
        content = news_data.get('content', '')
        url = news_data.get('url', '')

        # 检查标题和内容是否过于相似
        if title and content:
            similarity = self._calculate_text_similarity(title, content)
            if similarity > 0.9:
                errors.append("标题和内容过于相似，信息重复度高")

        # 检查URL域名和新闻源是否一致
        if url and news_data.get('source'):
            try:
                parsed_url = urlparse(url)
                url_domain = parsed_url.netloc.lower()
                source_lower = news_data['source'].lower()

                # 如果URL域名和新闻源完全不相关
                if not any(domain in url_domain for domain in source_lower.split()) and \
                   not any(word in url_domain for word in source_lower.split()):
                    # 只有在新闻源比较知名的情况下才报错
                    if any(re.match(pattern, source_lower, re.IGNORECASE)
                          for pattern in self.trusted_sources_patterns):
                        pass  # 可信新闻源允许不一致
            except Exception:
                pass  # URL解析失败时跳过检查

    def _is_low_quality_content(self, content: str) -> bool:
        """检查是否为低质量内容"""
        content_lower = content.lower()

        # 检查是否为占位符内容
        placeholders = [
            'lorem ipsum', 'placeholder', 'coming soon', 'to be updated',
            'null', 'n/a', 'undefined', 'example text', 'sample content'
        ]

        if any(placeholder in content_lower for placeholder in placeholders):
            return True

        # 检查字符重复率
        unique_chars = len(set(content))
        total_chars = len(content)
        if total_chars > 50 and unique_chars / total_chars < 0.3:
            return True

        # 检查句子重复
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 3:
            unique_sentences = len(set(sentence.strip() for sentence in sentences if sentence.strip()))
            if unique_sentences / len(sentences) < 0.5:
                return True

        return False

    def _has_excessive_repetition(self, text: str) -> bool:
        """检查是否有过多的重复"""
        # 检查字符重复
        if len(text) > 20:
            # 连续相同字符
            if re.search(r'(.)\1{5,}', text):
                return True

            # 重复单词
            words = text.split()
            if len(words) > 10:
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

                # 如果有单词重复超过总词数的30%
                max_count = max(word_counts.values()) if word_counts else 0
                if max_count > len(words) * 0.3:
                    return True

        return False

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单的词重叠率）"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def generate_content_hash(self, title: str, content: str, source: str) -> str:
        """生成内容哈希用于去重"""
        combined = f"{title.strip()} {content.strip()} {source.strip()}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def is_duplicate_news(self, news_data: Dict[str, Any], existing_hashes: set) -> bool:
        """检查是否为重复新闻"""
        title = news_data.get('title', '')
        content = news_data.get('content', '')
        source = news_data.get('source', '')

        content_hash = self.generate_content_hash(title, content, source)
        return content_hash in existing_hashes

    def sanitize_news_data(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """清洗和标准化新闻数据"""
        sanitized = news_data.copy()

        # 清洗标题
        if 'title' in sanitized:
            sanitized['title'] = self._sanitize_text(sanitized['title'])

        # 清洗内容
        if 'content' in sanitized:
            sanitized['content'] = self._sanitize_text(sanitized['content'])

        # 清洗新闻源
        if 'source' in sanitized:
            sanitized['source'] = self._sanitize_text(sanitized['source'])

        # 标准化URL
        if 'url' in sanitized and sanitized['url']:
            sanitized['url'] = self._normalize_url(sanitized['url'])

        # 生成内容哈希
        if all(key in sanitized for key in ['title', 'content', 'source']):
            sanitized['hash'] = self.generate_content_hash(
                sanitized['title'],
                sanitized['content'],
                sanitized['source']
            )

        return sanitized

    def _sanitize_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())

        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\-\.,!?;:()\[\]{}"\'/]', '', text)

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def _normalize_url(self, url: str) -> str:
        """标准化URL"""
        if not url:
            return ""

        url = url.strip()

        # 移除URL中的片段标识符
        if '#' in url:
            url = url.split('#')[0]

        # 移除UTM参数
        utm_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
        parsed = urlparse(url)

        if parsed.query:
            query_params = []
            for param in parsed.query.split('&'):
                key_value = param.split('=', 1)
                if len(key_value) == 2 and key_value[0] not in utm_params:
                    query_params.append(param)

            if query_params:
                from urllib.parse import urlencode, parse_qs, urlunparse
                new_parsed = parsed._replace(query='&'.join(query_params))
                url = urlunparse(new_parsed)

        return url


# 便捷函数
def validate_news(news_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """验证新闻数据的便捷函数"""
    validator = NewsValidator()
    return validator.validate_news_data(news_data)


def sanitize_news(news_data: Dict[str, Any]) -> Dict[str, Any]:
    """清洗新闻数据的便捷函数"""
    validator = NewsValidator()
    return validator.sanitize_news_data(news_data)