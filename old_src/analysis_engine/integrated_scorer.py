"""Integrated scoring system for investment decisions.

Implements requirements-based scoring:
- Catalyst Importance Score: Max 50 points
- Sentiment Score: Max 30 points
- Technical/Market Score: Max 20 points
- Total: Max 100 points
- Buy threshold: >= 80 points
"""

from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ScoringRules:
    """Scoring rules based on requirements."""
    
    # Catalyst keywords and their scores (Max 50 points)
    CATALYST_KEYWORDS = {
        "上方修正": 50,
        "業績予想の上方修正": 50,
        "売上高上方修正": 45,
        "業務提携": 40,
        "資本提携": 40,
        "M&A": 40,
        "買収": 40,
        "新製品": 30,
        "新サービス": 30,
        "特許取得": 25,
        "決算": 20,
        "増配": 25,
        "自社株買い": 30,
        "株式分割": 15,
        "IR": 10,
        "開示": 5
    }
    
    # Sentiment scoring thresholds
    SENTIMENT_THRESHOLDS = {
        "very_positive": (0.8, 30),   # >80% positive = 30 points
        "positive": (0.6, 20),         # 60-80% = 20 points
        "neutral": (0.4, 10),          # 40-60% = 10 points
        "negative": (0.0, 0)           # <40% = 0 points
    }
    
    # Technical scoring components
    TECHNICAL_SCORING = {
        "uptrend": 10,         # Uptrend = 10 points
        "volume_spike": 5,     # Volume >1.5x average = 5 points
        "sector_positive": 5,  # Sector performing well = 5 points
    }
    
    # Filter conditions (prevent buy even if score is high)
    FILTER_CONDITIONS = {
        "rsi_overbought": 75,    # RSI > 75 = skip
        "ma_deviation_max": 0.25  # >25% above MA = skip
    }
    
    # Buy decision threshold
    BUY_THRESHOLD = 80


class IntegratedScorer:
    """Integrated scoring system for investment decisions."""
    
    def __init__(self, rules: Optional[Dict[str, Any]] = None) -> None:
        """Initialize integrated scorer.
        
        Args:
            rules: Custom scoring rules (optional)
        """
        self.rules = ScoringRules()
        
        # Override with custom rules if provided
        if rules:
            for key, value in rules.items():
                if hasattr(self.rules, key.upper()):
                    setattr(self.rules, key.upper(), value)
    
    def calculate_catalyst_score(self, ir_data: Optional[Dict[str, Any]]) -> int:
        """Calculate catalyst importance score (Max 50 points).
        
        Args:
            ir_data: IR/news data containing title, keywords, importance
            
        Returns:
            Score between 0-50
        """
        if not ir_data:
            return 0
        
        score = 0
        
        # Check for keywords in title and content
        title = ir_data.get("title", "").lower()
        keywords = ir_data.get("keywords", [])
        
        # Combine title and keywords for checking
        text_to_check = title + " " + " ".join(keywords).lower()
        
        # Find highest scoring keyword
        for keyword, points in self.rules.CATALYST_KEYWORDS.items():
            if keyword in text_to_check:
                score = max(score, points)
        
        # Cap at 50 points
        score = min(score, 50)
        
        logger.debug(f"Catalyst score: {score}/50 for '{title[:50]}...'")
        return score
    
    def calculate_sentiment_score(self, sentiment_data: Optional[Dict[str, Any]]) -> int:
        """Calculate sentiment score (Max 30 points).
        
        Args:
            sentiment_data: Sentiment data with positive_ratio, change_rate, mention_count
            
        Returns:
            Score between 0-30
        """
        if not sentiment_data:
            return 0
        
        positive_ratio = sentiment_data.get("positive_ratio", 0.5)
        change_rate = sentiment_data.get("change_rate", 1.0)  # 1.0 = no change
        mention_count = sentiment_data.get("mention_count", 0)
        
        # Base score from positive ratio
        base_score = 0
        for threshold_name, (threshold, points) in self.rules.SENTIMENT_THRESHOLDS.items():
            if positive_ratio >= threshold:
                base_score = points
                break
        
        # Apply momentum multiplier (change_rate)
        # If sentiment is improving rapidly, add bonus
        momentum_multiplier = 1.0
        if change_rate > 2.0:  # >100% increase
            momentum_multiplier = 1.3
        elif change_rate > 1.5:  # >50% increase
            momentum_multiplier = 1.2
        elif change_rate > 1.2:  # >20% increase
            momentum_multiplier = 1.1
        elif change_rate < 0.8:  # >20% decrease
            momentum_multiplier = 0.8
        
        # Apply volume consideration
        volume_multiplier = 1.0
        if mention_count < 10:
            volume_multiplier = 0.5  # Too few mentions
        elif mention_count > 100:
            volume_multiplier = 1.1  # High interest
        
        # Calculate final score
        score = int(base_score * momentum_multiplier * volume_multiplier)
        
        # Cap at 30 points
        score = min(score, 30)
        
        logger.debug(f"Sentiment score: {score}/30 (ratio={positive_ratio:.2f}, change={change_rate:.2f})")
        return score
    
    def calculate_technical_score(self, technical_data: Optional[Dict[str, Any]]) -> int:
        """Calculate technical/market environment score (Max 20 points).
        
        Args:
            technical_data: Technical indicators and market data
            
        Returns:
            Score between 0-20
        """
        if not technical_data:
            return 0
        
        score = 0
        
        # Trend analysis (10 points max)
        trend = technical_data.get("trend", "sideways")
        if trend == "uptrend":
            score += self.rules.TECHNICAL_SCORING["uptrend"]
        elif trend == "sideways":
            score += 5  # Half points for sideways
        
        # Volume analysis (5 points max)
        volume_ratio = technical_data.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            score += self.rules.TECHNICAL_SCORING["volume_spike"]
        elif volume_ratio > 1.2:
            score += 3  # Partial points
        
        # Sector performance (5 points max)
        sector_performance = technical_data.get("sector_performance", 0.0)
        if sector_performance > 0.01:  # >1% sector gain
            score += self.rules.TECHNICAL_SCORING["sector_positive"]
        elif sector_performance > 0:
            score += 2  # Partial points
        
        # Cap at 20 points
        score = min(score, 20)
        
        logger.debug(f"Technical score: {score}/20")
        return score
    
    def calculate_total_score(self, analysis_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate total score from all components.
        
        Args:
            analysis_data: Combined analysis data with catalyst, sentiment, technical
            
        Returns:
            Dictionary with individual scores and total
        """
        catalyst_score = self.calculate_catalyst_score(analysis_data.get("catalyst"))
        sentiment_score = self.calculate_sentiment_score(analysis_data.get("sentiment"))
        technical_score = self.calculate_technical_score(analysis_data.get("technical"))
        
        total_score = catalyst_score + sentiment_score + technical_score
        
        return {
            "catalyst_score": catalyst_score,
            "sentiment_score": sentiment_score,
            "technical_score": technical_score,
            "total_score": total_score,
            "max_possible": 100
        }
    
    def check_filter_conditions(self, technical_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if any filter conditions prevent trading.
        
        Args:
            technical_data: Technical indicators
            
        Returns:
            Tuple of (should_trade, reason_if_blocked)
        """
        rsi = technical_data.get("rsi", 50)
        ma_deviation = technical_data.get("ma_deviation", 0)
        
        # Check RSI overbought
        if rsi > self.rules.FILTER_CONDITIONS["rsi_overbought"]:
            return False, f"RSI overbought (>{self.rules.FILTER_CONDITIONS['rsi_overbought']})"
        
        # Check MA deviation
        if ma_deviation > self.rules.FILTER_CONDITIONS["ma_deviation_max"]:
            max_pct = int(self.rules.FILTER_CONDITIONS["ma_deviation_max"] * 100)
            return False, f"MA deviation too high (>{max_pct}%)"
        
        return True, None
    
    def make_investment_decision(self, total_score: int, 
                                rsi: float = 50, 
                                ma_deviation: float = 0) -> Dict[str, Any]:
        """Make investment decision based on score and filters.
        
        Args:
            total_score: Total score from analysis
            rsi: RSI indicator value
            ma_deviation: Moving average deviation ratio
            
        Returns:
            Investment decision dictionary
        """
        # Check filter conditions first
        technical_data = {"rsi": rsi, "ma_deviation": ma_deviation}
        should_trade, filter_reason = self.check_filter_conditions(technical_data)
        
        if not should_trade:
            return {
                "action": "hold",
                "execute": False,
                "reason": filter_reason,
                "total_score": total_score
            }
        
        # Check score threshold
        if total_score >= self.rules.BUY_THRESHOLD:
            return {
                "action": "buy",
                "execute": True,
                "reason": f"Score {total_score} exceeds threshold {self.rules.BUY_THRESHOLD}",
                "total_score": total_score
            }
        else:
            return {
                "action": "hold",
                "execute": False,
                "reason": f"Score {total_score} below threshold {self.rules.BUY_THRESHOLD}",
                "total_score": total_score
            }
    
    def analyze_with_breakdown(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with detailed score breakdown.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Analysis result with detailed breakdown
        """
        # Calculate scores
        scores = self.calculate_total_score(analysis_data)
        
        # Get technical data for filters
        technical_data = analysis_data.get("technical", {})
        
        # Make decision
        decision = self.make_investment_decision(
            scores["total_score"],
            technical_data.get("rsi", 50),
            technical_data.get("ma_deviation", 0)
        )
        
        # Create detailed breakdown
        breakdown = {
            "catalyst_details": {
                "score": scores["catalyst_score"],
                "max_score": 50,
                "keywords_found": self._find_keywords(analysis_data.get("catalyst"))
            },
            "sentiment_details": {
                "score": scores["sentiment_score"],
                "max_score": 30,
                "positive_ratio": analysis_data.get("sentiment", {}).get("positive_ratio", 0),
                "momentum": analysis_data.get("sentiment", {}).get("change_rate", 1.0)
            },
            "technical_details": {
                "score": scores["technical_score"],
                "max_score": 20,
                "trend": analysis_data.get("technical", {}).get("trend", "unknown"),
                "volume_ratio": analysis_data.get("technical", {}).get("volume_ratio", 1.0)
            }
        }
        
        return {
            "scores": scores,
            "decision": decision["action"],
            "execute": decision["execute"],
            "scoring_breakdown": breakdown,
            "decision_rationale": decision["reason"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _find_keywords(self, ir_data: Optional[Dict[str, Any]]) -> List[str]:
        """Find matching keywords in IR data.
        
        Args:
            ir_data: IR/news data
            
        Returns:
            List of found keywords
        """
        if not ir_data:
            return []
        
        found = []
        text = (ir_data.get("title", "") + " " + 
                " ".join(ir_data.get("keywords", []))).lower()
        
        for keyword in self.rules.CATALYST_KEYWORDS.keys():
            if keyword in text:
                found.append(keyword)
        
        return found
    
    def batch_analyze(self, companies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple companies and rank by score.
        
        Args:
            companies_data: List of company analysis data
            
        Returns:
            Sorted list of analysis results (highest score first)
        """
        results = []
        
        for company_data in companies_data:
            result = self.analyze_with_breakdown(company_data)
            result["symbol"] = company_data.get("symbol", "UNKNOWN")
            result["company_name"] = company_data.get("company_name", "Unknown")
            results.append(result)
        
        # Sort by total score (descending)
        results.sort(key=lambda x: x["scores"]["total_score"], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results, 1):
            result["rank"] = i
        
        return results