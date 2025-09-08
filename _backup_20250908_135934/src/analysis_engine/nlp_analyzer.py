"""Natural Language Processing analyzer for IR documents and social media."""

from typing import Dict, List, Any, Tuple
from decimal import Decimal
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    positive: float
    negative: float
    neutral: float
    overall: float
    confidence: float


@dataclass
class ImportanceScore:
    """IR/News importance scoring result."""
    score: int
    max_score: int
    matched_keywords: List[str]
    category: str


class NlpAnalyzer:
    """Natural Language Processing analyzer using GiNZA and BERT models."""
    
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese") -> None:
        """
        Initialize NLP Analyzer.
        
        Args:
            model_name: Pre-trained BERT model name
        """
        self.model_name = model_name
        self.is_initialized = False
        self.keyword_scores = self._load_keyword_scoring_config()
        logger.info(f"NlpAnalyzer initialized with model: {model_name}")
    
    def _load_keyword_scoring_config(self) -> Dict[str, int]:
        """
        Load keyword scoring configuration.
        
        Returns:
            Dictionary mapping keywords to scores
        """
        # TODO: Load from configuration file
        return {
            "上方修正": 50,
            "業務提携": 40,
            "決算": 30,
            "買収": 45,
            "合併": 45,
            "新商品": 25,
            "増配": 35,
            "株式分割": 20
        }
    
    def initialize_models(self) -> None:
        """Initialize NLP models (GiNZA, BERT)."""
        try:
            import spacy
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            logger.info("Initializing NLP models...")
            
            # Initialize GiNZA for Japanese morphological analysis
            self.nlp = spacy.load("ja_ginza")
            logger.info("GiNZA model loaded successfully")
            
            # Initialize BERT tokenizer and model for sentiment analysis
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                "cl-tohoku/bert-base-japanese-sentiment"
            )
            self.sentiment_model.eval()
            
            # Set device
            # Auto-detect best available device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon GPU
            elif torch.cuda.is_available():
                self.device = torch.device('cuda:0')  # NVIDIA GPU
            else:
                self.device = torch.device('cpu')  # CPU fallback
            logger.info(f"NLP Analyzer using device: {self.device}")
            self.sentiment_model.to(self.device)
            
            logger.info(f"BERT sentiment model loaded on device: {self.device}")
            
            self.is_initialized = True
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            # Fallback to simple implementation
            self.is_initialized = False
            logger.warning("Falling back to simple NLP implementation")
    
    def analyze_ir_importance(self, text: str) -> Dict[str, Any]:
        """
        Analyze IR/press release importance using GiNZA morphological analysis.
        
        Args:
            text: IR text content
            
        Returns:
            Dictionary with score and matched keywords
        """
        if not text.strip():
            return {"score": 0, "keywords": [], "entities": []}
        
        matched_keywords = []
        total_score = 0
        entities = []
        
        if self.is_initialized and hasattr(self, 'nlp'):
            try:
                # Use GiNZA for advanced analysis
                doc = self.nlp(text)
                
                # Extract named entities and important terms
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERCENT']:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
                
                # Advanced keyword matching with context
                for keyword, score in self.keyword_scores.items():
                    if keyword in text:
                        matched_keywords.append(keyword)
                        # Context-aware scoring based on surrounding words
                        keyword_doc = self.nlp(keyword)
                        context_boost = 1.0
                        
                        # Look for amplifying words nearby
                        amplifiers = ['大幅', '急激', '著しい', '大きく']
                        for amplifier in amplifiers:
                            if amplifier in text:
                                context_boost = 1.2
                                break
                        
                        adjusted_score = int(score * context_boost)
                        total_score = max(total_score, adjusted_score)
                
                logger.debug(f"GiNZA analysis found {len(entities)} entities, score: {total_score}")
                
            except Exception as e:
                logger.warning(f"GiNZA analysis failed, falling back to simple method: {e}")
                return self._simple_ir_analysis(text)
        else:
            return self._simple_ir_analysis(text)
        
        return {
            "score": total_score,
            "keywords": matched_keywords,
            "entities": entities
        }

    
    def _simple_ir_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback simple IR analysis using keyword matching.
        
        Args:
            text: IR text content
            
        Returns:
            Dictionary with score and matched keywords
        """
        matched_keywords = []
        total_score = 0
        
        # Simple keyword matching for fallback
        for keyword, score in self.keyword_scores.items():
            if keyword in text:
                matched_keywords.append(keyword)
                total_score = max(total_score, score)  # Use highest score
        
        return {
            "score": total_score,
            "keywords": matched_keywords,
            "entities": []
        }
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze sentiment of text collection using BERT model.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment ratios
        """
        if not texts:
            return {
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
                "change_rate": 0.0
            }
        
        if self.is_initialized and hasattr(self, 'sentiment_model'):
            try:
                return self._bert_sentiment_analysis(texts)
            except Exception as e:
                logger.warning(f"BERT sentiment analysis failed, falling back to simple method: {e}")
                return self._simple_sentiment_analysis(texts)
        else:
            return self._simple_sentiment_analysis(texts)
    
    def _bert_sentiment_analysis(self, texts: List[str]) -> Dict[str, float]:
        """
        Perform sentiment analysis using BERT model.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment ratios
        """
        import torch
        
        positive_count = 0
        negative_count = 0
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            # Convert prediction to sentiment (0: negative, 1: positive)
            if predicted_class == 1 and confidence > 0.6:  # High confidence positive
                positive_count += 1
            elif predicted_class == 0 and confidence > 0.6:  # High confidence negative
                negative_count += 1
        
        total = len(texts)
        neutral_count = total - positive_count - negative_count
        
        positive_ratio = positive_count / total if total > 0 else 0.0
        negative_ratio = negative_count / total if total > 0 else 0.0
        
        # Calculate change rate based on sentiment ratio
        if positive_ratio > 0.6:
            change_rate = 0.5  # Strong positive sentiment
        elif negative_ratio > 0.6:
            change_rate = -0.3  # Strong negative sentiment
        else:
            change_rate = 0.1  # Neutral
        
        logger.debug(f"BERT sentiment analysis: pos={positive_ratio:.2f}, neg={negative_ratio:.2f}")
        
        return {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_count / total if total > 0 else 1.0,
            "change_rate": change_rate
        }
    
    def _simple_sentiment_analysis(self, texts: List[str]) -> Dict[str, float]:
        """
        Fallback simple sentiment analysis using keyword matching.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment ratios
        """
        positive_words = ["素晴らしい", "期待", "好調", "買い時", "上昇", "強い", "良好"]
        negative_words = ["危険", "悪化", "心配", "売り時", "下落", "弱い", "懸念"]
        
        positive_count = 0
        negative_count = 0
        
        for text in texts:
            for word in positive_words:
                if word in text:
                    positive_count += 1
                    break
            for word in negative_words:
                if word in text:
                    negative_count += 1
                    break
        
        total = len(texts)
        neutral_count = total - positive_count - negative_count
        
        positive_ratio = positive_count / total if total > 0 else 0.0
        negative_ratio = negative_count / total if total > 0 else 0.0
        
        # Calculate change rate based on sentiment ratio
        if positive_ratio > 0.6:
            change_rate = 0.5  # Strong positive sentiment
        elif negative_ratio > 0.6:
            change_rate = -0.3  # Strong negative sentiment
        else:
            change_rate = 0.1  # Neutral
        
        return {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_count / total if total > 0 else 1.0,
            "change_rate": change_rate
        }
    
    def calculate_sentiment_momentum(self, current_texts: List[str], 
                                   previous_texts: List[str]) -> Tuple[float, float]:
        """
        Calculate sentiment change rate (momentum).
        
        Args:
            current_texts: Current period texts
            previous_texts: Previous period texts
            
        Returns:
            Tuple of (sentiment_change, momentum_score)
        """
        current_sentiment = self.analyze_sentiment(current_texts)
        previous_sentiment = self.analyze_sentiment(previous_texts)
        
        sentiment_change = current_sentiment.overall - previous_sentiment.overall
        momentum_score = abs(sentiment_change) * current_sentiment.confidence
        
        logger.debug(f"Sentiment momentum: {momentum_score:.3f}")
        return sentiment_change, momentum_score
    
    def extract_stock_mentions(self, text: str) -> List[str]:
        """
        Extract stock symbols/codes mentioned in text using GiNZA.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of stock symbols found
        """
        import re
        
        stock_symbols = []
        
        if self.is_initialized and hasattr(self, 'nlp'):
            try:
                # Use GiNZA for named entity recognition
                doc = self.nlp(text)
                
                # Look for organization entities that might be stock-related
                for ent in doc.ents:
                    if ent.label_ == 'ORG':
                        # Check if it's followed by stock code pattern
                        stock_symbols.append(ent.text)
                
            except Exception as e:
                logger.warning(f"GiNZA analysis failed for stock extraction: {e}")
        
        # Regex patterns for Japanese stock codes (improved for Japanese text)
        patterns = [
            r'(\d{4})(?=\D|$)',  # 4-digit codes followed by non-digit or end of string
            r'(\d{4})\.T',       # Tokyo Stock Exchange format like 7203.T
            r'[（(](\d{4})[）)]', # Codes in parentheses (both Japanese and ASCII)
            r'^(\d{4})',         # Codes at the beginning of text
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Validate stock code range (1000-9999 for Japanese stocks)
                if isinstance(match, str):
                    if match.isdigit():
                        code = int(match)
                        if 1000 <= code <= 9999:
                            stock_symbols.append(match)
                    elif '.' in match and match.replace('.T', '').isdigit():
                        code = int(match.replace('.T', ''))
                        if 1000 <= code <= 9999:
                            stock_symbols.append(match)
        
        # Remove duplicates while preserving order
        unique_symbols = []
        seen = set()
        for symbol in stock_symbols:
            if symbol not in seen:
                unique_symbols.append(symbol)
                seen.add(symbol)
        
        logger.debug(f"Extracted stock mentions: {unique_symbols}")
        return unique_symbols
    
    def calculate_sentiment_score_for_trading(self, texts: List[str]) -> int:
        """
        Calculate sentiment score for trading decision (0-30 points).
        
        Args:
            texts: Texts to analyze
            
        Returns:
            Score from 0 to 30 for trading algorithm
        """
        sentiment = self.analyze_sentiment(texts)
        
        # TODO: Implement trading-specific scoring logic
        # Base score on positive ratio and momentum
        base_score = sentiment.positive * 20  # Max 20 points
        momentum_bonus = min(sentiment.confidence * 10, 10)  # Max 10 points
        
        total_score = int(base_score + momentum_bonus)
        logger.debug(f"Sentiment trading score: {total_score}/30")
        
        return min(total_score, 30)
    
    def calculate_mention_anomaly(self, mention_counts: List[int], current_count: int) -> Dict[str, Any]:
        """
        Calculate mention count anomaly detection.
        
        Args:
            mention_counts: Historical mention counts
            current_count: Current mention count
            
        Returns:
            Dictionary with anomaly detection result
        """
        if not mention_counts:
            raise ValueError("mention_counts cannot be empty")
        
        import statistics
        
        mean = statistics.mean(mention_counts)
        stdev = statistics.stdev(mention_counts) if len(mention_counts) > 1 else 0
        
        if stdev == 0:
            z_score = 0.0
        else:
            z_score = (current_count - mean) / stdev
        
        threshold = 3.0
        is_anomaly = abs(z_score) > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "threshold": threshold
        }
    
    async def analyze_daily_catalyst(self, ir_data: List[Dict[str, Any]], sns_data: Dict[str, Any], board_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive daily catalyst analysis for 16:15 scheduled processing.
        
        Args:
            ir_data: List of IR announcements for the day
            sns_data: SNS mention statistics
            board_data: Yahoo board post data
            
        Returns:
            Comprehensive catalyst analysis result
        """
        logger.info("Starting daily catalyst analysis")
        
        # Analyze IR importance
        ir_score = 0
        ir_keywords = []
        if ir_data:
            for ir in ir_data:
                if ir.get("has_trigger"):
                    analysis = self.analyze_ir_importance(ir.get("title", ""), ir.get("content", ""))
                    ir_score = max(ir_score, analysis.score)
                    ir_keywords.extend(analysis.keywords)
        
        # Analyze SNS sentiment and anomaly
        sns_score = 0
        sns_anomaly = False
        if sns_data:
            # Check for anomaly
            if sns_data.get("is_anomaly", False):
                sns_anomaly = True
                sns_score += 15  # Bonus for anomaly
            
            # Add sentiment score
            sentiment_ratio = sns_data.get("sentiment_ratio", 0)
            if sentiment_ratio > 0.7:  # 70% positive
                sns_score += 15
            elif sentiment_ratio > 0.5:
                sns_score += 10
            else:
                sns_score += 5
        
        # Analyze board sentiment
        board_score = 0
        if board_data and board_data.get("sentiment"):
            board_sentiment = board_data["sentiment"]
            if board_sentiment.get("positive_ratio", 0) > 0.6:
                board_score = 10
            elif board_sentiment.get("positive_ratio", 0) > 0.4:
                board_score = 5
        
        # Calculate total catalyst score (max 50)
        total_catalyst_score = min(ir_score + sns_score + board_score, 50)
        
        return {
            "catalyst_score": total_catalyst_score,
            "ir_score": ir_score,
            "ir_keywords": ir_keywords,
            "sns_score": sns_score,
            "sns_anomaly": sns_anomaly,
            "board_score": board_score,
            "analysis_timestamp": datetime.now()
        }
    
    def analyze_daily_sentiment_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment for a batch of texts (for daily processing).
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Batch sentiment analysis result
        """
        if not texts:
            return {
                "average_sentiment": 0.5,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
        
        sentiments = []
        for text in texts:
            result = self.analyze_sentiment(text)
            sentiments.append(result)
        
        # Aggregate results
        positive_count = sum(1 for s in sentiments if s.sentiment == "positive")
        negative_count = sum(1 for s in sentiments if s.sentiment == "negative")
        neutral_count = len(sentiments) - positive_count - negative_count
        
        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
        
        return {
            "average_sentiment": avg_confidence,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_analyzed": len(sentiments),
            "positive_ratio": positive_count / len(sentiments) if sentiments else 0
        }

    def calculate_sentiment_score(self, sentiment_result: Dict[str, float]) -> int:
        """
        Calculate sentiment score for trading (0-30 points).
        
        Args:
            sentiment_result: Result from analyze_sentiment
            
        Returns:
            Score from 0 to 30
        """
        positive_ratio = sentiment_result.get("positive_ratio", 0.0)
        negative_ratio = sentiment_result.get("negative_ratio", 0.0)
        change_rate = sentiment_result.get("change_rate", 0.0)
        
        # Base score from sentiment ratio (max 20 points)
        sentiment_score = (positive_ratio - negative_ratio) * 25
        
        # Bonus for positive change rate (max 10 points)
        change_bonus = max(change_rate * 20, 0)
        
        total_score = int(max(0, sentiment_score + change_bonus))
        return min(total_score, 30)