"""ModernBERT-ja センチメント分析システム"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import spacy
from dataclasses import dataclass

from src.utils.logger_utils import create_dual_logger
from .news_collector import NewsItem

logger = create_dual_logger(__name__, console_output=True)


@dataclass
class SentimentResult:
    """センチメント分析結果"""
    text: str
    sentiment_score: float  # -1.0 (negative) to 1.0 (positive)
    confidence: float  # 0.0 to 1.0
    emotion_scores: Dict[str, float]  # joy, fear, anger, sadness, etc.
    keywords: List[str]  # 重要キーワード
    analysis_time: datetime


class ModernBERTSentimentAnalyzer:
    """ModernBERT-ja-130mベースのセンチメント分析システム"""
    
    def __init__(
        self, 
        model_name: str = "sbintuitions/modernbert-ja-130m",
        device: Optional[str] = None,
        enable_spacy: bool = True
    ):
        """初期化
        
        Args:
            model_name: HuggingFaceモデル名
            device: 計算デバイス（None=自動選択）
            enable_spacy: spaCy前処理を有効にするか
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.enable_spacy = enable_spacy
        
        # HuggingFace transformers初期化
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error = None
        
        # spaCy初期化
        self.nlp = None
        if enable_spacy:
            self._load_spacy()
        
        # センチメント辞書（日本語）
        self.sentiment_keywords = self._build_sentiment_keywords()
        
        # ModernBERT読み込み
        self._load_modernbert()
    
    def _get_device(self) -> str:
        """最適なデバイスを選択"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_spacy(self):
        """spaCy日本語モデルを読み込み"""
        try:
            self.nlp = spacy.load("ja_ginza")
            logger.info("spaCy (ja_ginza) 読み込み完了")
        except OSError:
            try:
                self.nlp = spacy.load("ja_core_news_sm")
                logger.info("spaCy (ja_core_news_sm) 読み込み完了")
            except OSError:
                logger.warning("spaCy日本語モデルが見つかりません。基本的な前処理のみ使用")
                self.nlp = None
    
    def _load_modernbert(self):
        """ModernBERT-jaモデルを読み込み"""
        try:
            logger.info(f"ModernBERT読み込み開始: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # センチメント分類層を追加
            hidden_size = self.model.config.hidden_size
            self.sentiment_classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),  # 感情スコア (-1 to 1)
                nn.Tanh()
            ).to(self.device)
            
            self.is_loaded = True
            logger.info("ModernBERT-ja 読み込み完了")
            
        except Exception as e:
            self.load_error = f"ModernBERT読み込みエラー: {e}"
            logger.error(self.load_error)
            self.is_loaded = False
    
    def _build_sentiment_keywords(self) -> Dict[str, Dict[str, float]]:
        """日本語センチメントキーワード辞書を構築"""
        return {
            "positive": {
                # 業績・財務ポジティブ
                "増益": 0.8, "黒字": 0.7, "利益": 0.6, "成長": 0.7, "拡大": 0.6,
                "好調": 0.8, "上昇": 0.6, "向上": 0.6, "改善": 0.7, "増加": 0.5,
                # 事業ポジティブ  
                "新規": 0.5, "革新": 0.7, "成功": 0.8, "達成": 0.7, "獲得": 0.6,
                "提携": 0.6, "協力": 0.5, "展開": 0.5, "拡張": 0.6,
                # 評価ポジティブ
                "期待": 0.6, "評価": 0.5, "注目": 0.5, "推奨": 0.7, "強気": 0.8
            },
            "negative": {
                # 業績・財務ネガティブ
                "減益": -0.8, "赤字": -0.8, "損失": -0.7, "減少": -0.6, "下落": -0.7,
                "不振": -0.8, "低迷": -0.7, "悪化": -0.8, "減収": -0.7,
                # 事業ネガティブ
                "撤退": -0.8, "中止": -0.7, "延期": -0.6, "困難": -0.6, "問題": -0.5,
                "リスク": -0.5, "懸念": -0.6, "不安": -0.7, "危機": -0.8,
                # 評価ネガティブ
                "弱気": -0.8, "売り": -0.6, "下方": -0.7, "警戒": -0.6
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """テキストの前処理
        
        Args:
            text: 処理対象テキスト
            
        Returns:
            前処理済みテキスト
        """
        if not text:
            return ""
        
        # HTML/XML タグ除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL除去
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 特殊文字正規化
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', ' ', text)
        
        # 多重空白を単一空白に
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """重要キーワードを抽出
        
        Args:
            text: 対象テキスト
            top_k: 上位何件を返すか
            
        Returns:
            重要キーワードリスト
        """
        if not text or not self.nlp:
            # フォールバック: 基本的な単語分割
            words = text.split()
            return words[:top_k]
        
        try:
            doc = self.nlp(text)
            
            # 名詞、動詞、形容詞を抽出
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                    len(token.text) > 1 and 
                    not token.is_stop and 
                    not token.is_punct):
                    keywords.append(token.lemma_)
            
            # 頻度順でソート（簡易実装）
            from collections import Counter
            counter = Counter(keywords)
            return [word for word, count in counter.most_common(top_k)]
        
        except Exception as e:
            logger.debug(f"キーワード抽出エラー: {e}")
            words = text.split()
            return words[:top_k]
    
    def analyze_with_keywords(self, text: str) -> Tuple[float, float]:
        """キーワードベースのセンチメント分析
        
        Args:
            text: 分析対象テキスト
            
        Returns:
            (センチメントスコア, 信頼度) のタプル
        """
        if not text:
            return 0.0, 0.0
        
        text_lower = text.lower()
        positive_score = 0.0
        negative_score = 0.0
        match_count = 0
        
        # ポジティブキーワード
        for keyword, score in self.sentiment_keywords["positive"].items():
            if keyword in text_lower:
                positive_score += score
                match_count += 1
        
        # ネガティブキーワード
        for keyword, score in self.sentiment_keywords["negative"].items():
            if keyword in text_lower:
                negative_score += abs(score)  # 絶対値で加算
                match_count += 1
        
        if match_count == 0:
            return 0.0, 0.0
        
        # スコア計算
        net_score = (positive_score - negative_score) / match_count
        confidence = min(match_count / 10.0, 1.0)  # マッチ数に基づく信頼度
        
        return np.clip(net_score, -1.0, 1.0), confidence
    
    def analyze_with_modernbert(self, text: str) -> Tuple[float, float]:
        """ModernBERT-jaを使用したセンチメント分析
        
        Args:
            text: 分析対象テキスト
            
        Returns:
            (センチメントスコア, 信頼度) のタプル
        """
        if not self.is_loaded or not text:
            return 0.0, 0.0
        
        try:
            # テキストのトークン化
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                # ModernBERT エンコーディング
                outputs = self.model(**inputs)
                pooled_output = outputs.last_hidden_state.mean(dim=1)  # 平均プーリング
                
                # センチメント予測
                sentiment_logit = self.sentiment_classifier(pooled_output)
                sentiment_score = sentiment_logit.item()
                
                # 信頼度計算（分布の尖度から推定）
                confidence = min(abs(sentiment_score) * 2, 1.0)
                
                return sentiment_score, confidence
        
        except Exception as e:
            logger.warning(f"ModernBERT分析エラー: {e}")
            return 0.0, 0.0
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """統合センチメント分析
        
        Args:
            text: 分析対象テキスト
            
        Returns:
            センチメント分析結果
        """
        start_time = datetime.now()
        
        # テキスト前処理
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                confidence=0.0,
                emotion_scores={},
                keywords=[],
                analysis_time=start_time
            )
        
        # キーワードベース分析
        keyword_score, keyword_confidence = self.analyze_with_keywords(processed_text)
        
        # ModernBERT分析（利用可能な場合）
        bert_score, bert_confidence = self.analyze_with_modernbert(processed_text)
        
        # 統合スコア計算
        if self.is_loaded and bert_confidence > 0.1:
            # ModernBERTの結果を重視
            final_score = (bert_score * 0.7 + keyword_score * 0.3)
            final_confidence = (bert_confidence + keyword_confidence) / 2
        else:
            # キーワードベースのみ
            final_score = keyword_score
            final_confidence = keyword_confidence
        
        # キーワード抽出
        keywords = self.extract_keywords(processed_text)
        
        # 感情カテゴリスコア（簡易実装）
        emotion_scores = {
            "positive": max(0, final_score),
            "negative": max(0, -final_score),
            "neutral": 1 - abs(final_score)
        }
        
        return SentimentResult(
            text=text,
            sentiment_score=final_score,
            confidence=final_confidence,
            emotion_scores=emotion_scores,
            keywords=keywords,
            analysis_time=start_time
        )
    
    def analyze_news_batch(self, news_items: List[NewsItem]) -> List[SentimentResult]:
        """ニュースバッチのセンチメント分析
        
        Args:
            news_items: ニュースアイテムリスト
            
        Returns:
            センチメント分析結果リスト
        """
        results = []
        
        for news in news_items:
            # タイトルとコンテンツを結合
            full_text = f"{news.title} {news.content}"
            
            sentiment_result = self.analyze_sentiment(full_text)
            results.append(sentiment_result)
        
        logger.info(f"バッチセンチメント分析完了: {len(results)}件")
        return results
    
    def calculate_overall_sentiment(
        self, 
        sentiment_results: List[SentimentResult],
        weight_by_confidence: bool = True
    ) -> Tuple[float, float]:
        """全体的なセンチメントスコアを計算
        
        Args:
            sentiment_results: センチメント分析結果リスト
            weight_by_confidence: 信頼度で重み付けするか
            
        Returns:
            (全体センチメントスコア, 平均信頼度) のタプル
        """
        if not sentiment_results:
            return 0.0, 0.0
        
        if weight_by_confidence:
            # 信頼度重み付き平均
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for result in sentiment_results:
                weight = result.confidence if result.confidence > 0 else 0.1
                weighted_sum += result.sentiment_score * weight
                weight_sum += weight
            
            overall_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            # 単純平均
            overall_score = sum(r.sentiment_score for r in sentiment_results) / len(sentiment_results)
        
        avg_confidence = sum(r.confidence for r in sentiment_results) / len(sentiment_results)
        
        return overall_score, avg_confidence
    
    def get_model_status(self) -> Dict[str, Any]:
        """モデル状態情報を取得
        
        Returns:
            モデル状態情報
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "has_spacy": self.nlp is not None,
            "spacy_model": self.nlp.meta.get("name") if self.nlp else None
        }