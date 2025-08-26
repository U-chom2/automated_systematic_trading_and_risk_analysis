"""
システム設定管理モジュール

取引システムの各種設定を管理する。
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from decimal import Decimal


logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """取引システムの設定"""
    
    # 資金管理
    capital: Decimal = Decimal("1000000")  # 初期資金（円）
    risk_per_trade_ratio: float = 0.01     # 1取引あたりのリスク率
    max_positions: int = 5                 # 最大ポジション数
    
    # 取引時間
    market_start_time: str = "09:00"       # 市場開始時間
    market_end_time: str = "15:00"         # 市場終了時間
    
    # スコアリング設定
    buy_decision_threshold: int = 80       # 買い判断のスコア閾値
    rsi_overheating_threshold: float = 75.0        # RSI過熱閾値
    ma_deviation_threshold: float = 25.0           # MA乖離率閾値
    
    # リスク管理
    stop_loss_percentage: float = 0.08     # デフォルト損切り率
    take_profit_percentage: float = 0.15   # デフォルト利確率
    re_entry_prohibition_hours: int = 3    # 再エントリー禁止時間
    
    # データ収集設定
    tdnet_poll_interval: int = 1           # TDnetポーリング間隔（秒）
    sns_mention_check_interval: int = 60   # SNSメンション確認間隔（秒）
    price_update_interval: int = 5         # 価格更新間隔（秒）
    
    # ログ設定
    log_level: str = "INFO"                # ログレベル
    log_file_path: str = "_docs/system.log"  # ログファイルパス
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で出力"""
        data = asdict(self)
        # Decimalを文字列に変換
        data['capital'] = str(self.capital)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingConfig':
        """辞書から設定を復元"""
        if 'capital' in data:
            data['capital'] = Decimal(str(data['capital']))
        return cls(**data)


class ConfigManager:
    """
    設定管理クラス
    
    システムの各種設定を管理し、ファイルへの保存・読み込みを行う。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        ConfigManagerを初期化
        
        Args:
            config_path: 設定ファイルパス（デフォルト: config/trading_config.json）
        """
        self.config_path = Path(config_path or "config/trading_config.json")
        self.config = TradingConfig()
        
        # 設定ディレクトリを作成
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存設定を読み込み
        self.load_config()
    
    def load_config(self) -> None:
        """設定ファイルから設定を読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.config = TradingConfig.from_dict(data)
                logger.info(f"設定を読み込みました: {self.config_path}")
            else:
                logger.info("設定ファイルが存在しません。デフォルト設定を使用します。")
                self.save_config()  # デフォルト設定を保存
        except Exception as e:
            logger.error(f"設定の読み込みに失敗しました: {e}")
            logger.info("デフォルト設定を使用します。")
    
    def save_config(self) -> None:
        """設定をファイルに保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"設定を保存しました: {self.config_path}")
        except Exception as e:
            logger.error(f"設定の保存に失敗しました: {e}")
            raise
    
    def get_config(self) -> TradingConfig:
        """現在の設定を取得"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        設定を更新
        
        Args:
            **kwargs: 更新する設定項目
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # Decimalの特別処理
                if key == 'capital':
                    value = Decimal(str(value))
                setattr(self.config, key, value)
                logger.info(f"設定を更新しました: {key} = {value}")
            else:
                logger.warning(f"不明な設定項目です: {key}")
        
        self.save_config()
    
    def reset_to_default(self) -> None:
        """設定をデフォルトにリセット"""
        self.config = TradingConfig()
        self.save_config()
        logger.info("設定をデフォルトにリセットしました")
    
    def load_from_environment(self) -> None:
        """
        環境変数から設定を読み込み
        
        環境変数名の形式: TRADING_{設定名の大文字}
        例: TRADING_CAPITAL, TRADING_RISK_PER_TRADE_RATIO
        """
        env_mapping = {
            'TRADING_CAPITAL': ('capital', Decimal),
            'TRADING_RISK_PER_TRADE_RATIO': ('risk_per_trade_ratio', float),
            'TRADING_MAX_POSITIONS': ('max_positions', int),
            'TRADING_MARKET_START_TIME': ('market_start_time', str),
            'TRADING_MARKET_END_TIME': ('market_end_time', str),
            'TRADING_BUY_DECISION_THRESHOLD': ('buy_decision_threshold', int),
            'TRADING_RSI_OVERHEATING_THRESHOLD': ('rsi_overheating_threshold', float),
            'TRADING_MA_DEVIATION_THRESHOLD': ('ma_deviation_threshold', float),
            'TRADING_STOP_LOSS_PERCENTAGE': ('stop_loss_percentage', float),
            'TRADING_TAKE_PROFIT_PERCENTAGE': ('take_profit_percentage', float),
            'TRADING_LOG_LEVEL': ('log_level', str),
        }
        
        updated = False
        for env_var, (attr_name, type_func) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    typed_value = type_func(value)
                    setattr(self.config, attr_name, typed_value)
                    logger.info(f"環境変数から設定を更新: {attr_name} = {typed_value}")
                    updated = True
                except (ValueError, TypeError) as e:
                    logger.warning(f"環境変数の変換に失敗しました {env_var}: {e}")
        
        if updated:
            self.save_config()
    
    def export_config(self, export_path: Optional[str] = None) -> str:
        """
        設定をファイルにエクスポート
        
        Args:
            export_path: エクスポート先パス
            
        Returns:
            エクスポートしたファイルパス
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"config/trading_config_backup_{timestamp}.json"
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"設定をエクスポートしました: {export_path}")
        return str(export_path)
    
    def import_config(self, import_path: str) -> None:
        """
        設定をファイルからインポート
        
        Args:
            import_path: インポート元パス
        """
        import_path = Path(import_path)
        if not import_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {import_path}")
        
        with open(import_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.config = TradingConfig.from_dict(data)
        self.save_config()
        logger.info(f"設定をインポートしました: {import_path}")
    
    def validate_config(self) -> Dict[str, Any]:
        """
        設定の妥当性をチェック
        
        Returns:
            バリデーション結果
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 資金管理チェック
        if self.config.capital <= 0:
            validation_result["errors"].append("初期資金は0より大きい値を設定してください")
            validation_result["is_valid"] = False
        
        if not (0.001 <= self.config.risk_per_trade_ratio <= 0.1):
            validation_result["warnings"].append("1取引あたりのリスク率は0.1%〜10%の範囲を推奨します")
        
        if self.config.max_positions <= 0:
            validation_result["errors"].append("最大ポジション数は1以上を設定してください")
            validation_result["is_valid"] = False
        
        # スコアリング設定チェック
        if not (50 <= self.config.buy_decision_threshold <= 100):
            validation_result["warnings"].append("買い判断閾値は50〜100の範囲を推奨します")
        
        if not (70 <= self.config.rsi_overheating_threshold <= 90):
            validation_result["warnings"].append("RSI過熱閾値は70〜90の範囲を推奨します")
        
        # リスク管理チェック
        if not (0.01 <= self.config.stop_loss_percentage <= 0.3):
            validation_result["warnings"].append("損切り率は1%〜30%の範囲を推奨します")
        
        if self.config.take_profit_percentage <= self.config.stop_loss_percentage:
            validation_result["warnings"].append("利確率は損切り率より大きく設定することを推奨します")
        
        return validation_result
    
    def get_api_urls(self) -> Dict[str, str]:
        """API URLの設定を取得"""
        return {
            "tdnet_url": "https://www.release.tdnet.info/inbs/I_main_00.html",
            "yahoo_finance_url": "https://finance.yahoo.co.jp/",
            "x_api_url": "https://api.twitter.com/2/",
        }
    
    def get_risk_model_config(self) -> Dict[str, Any]:
        """リスクモデルの設定を取得"""
        return {
            "model_type": "neural_network",
            "input_features": ["atr", "historical_volatility", "volume_ratio", "price_momentum"],
            "hidden_layers": [64, 32, 16],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
    
    def get_nlp_config(self) -> Dict[str, Any]:
        """NLP分析の設定を取得"""
        return {
            "ginza_model": "ja_ginza",
            "bert_model": "cl-tohoku/bert-base-japanese-sentiment",
            "max_sequence_length": 512,
            "sentiment_confidence_threshold": 0.6,
            "batch_size": 16
        }


# 日付のインポートを追加
from datetime import datetime