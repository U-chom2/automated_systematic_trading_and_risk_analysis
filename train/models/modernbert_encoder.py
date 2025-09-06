"""
ModernBERT-jaを使用したIRニュースエンコーダー
方針.mdに基づいた実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModernBERTNewsEncoder(nn.Module):
    """ModernBERT-ja-130mを使用したニュースエンコーダー"""
    
    def __init__(
        self,
        model_name: str = "sbintuitions/modernbert-ja-130m",
        output_size: int = 64,
        max_length: int = 512,
        device: str = "cpu"
    ):
        """
        ModernBERT-jaベースのニュースエンコーダー初期化
        
        Args:
            model_name: HuggingFaceのモデル名
            output_size: 出力埋め込みサイズ
            max_length: 最大シーケンス長
            device: 実行デバイス
        """
        super().__init__()
        self.output_size = output_size
        self.max_length = max_length
        self.device = device
        
        # Transformersライブラリがインストールされている場合のみ使用
        self.use_bert = False
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            
            # ModernBERT-jaの読み込みを試行
            try:
                self.config = AutoConfig.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name)
                self.use_bert = True
                logger.info(f"ModernBERT-ja loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load ModernBERT-ja: {e}")
                raise e
        except ImportError:
            self.use_bert = False
            logger.warning("Transformers not installed, using simple encoder")
        
        if self.use_bert:
            # BERTの隠れ層サイズを取得
            bert_hidden_size = self.config.hidden_size if hasattr(self, 'config') else 768
            
            # センチメント分析ヘッド
            self.sentiment_head = nn.Sequential(
                nn.Linear(bert_hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 3)  # Positive, Neutral, Negative
            )
            
            # インパクトスコア予測ヘッド
            self.impact_head = nn.Sequential(
                nn.Linear(bert_hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            # 最終出力層
            self.output_projection = nn.Sequential(
                nn.Linear(bert_hidden_size + 4, output_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_size, output_size)
            )
        else:
            # シンプルエンコーダー（フォールバック）
            self.simple_encoder = SimpleNewsEncoder(output_size)
        
        self.to(device)
    
    def forward(self, news_texts: List[str]) -> torch.Tensor:
        """
        ニューステキストをエンコード
        
        Args:
            news_texts: IRニュースのテキストリスト
            
        Returns:
            エンコードされた特徴ベクトル
        """
        if not news_texts:
            return torch.zeros(1, self.output_size).to(self.device)
        
        if self.use_bert:
            # BERTでエンコード
            return self._encode_with_bert(news_texts)
        else:
            # シンプルエンコーダーでエンコード
            return self.simple_encoder(news_texts)
    
    def _encode_with_bert(self, texts: List[str]) -> torch.Tensor:
        """BERTを使用したエンコーディング"""
        # テキストを結合
        combined_text = " ".join(texts[:5])  # 最大5つのニュースを使用
        
        # トークン化
        inputs = self.tokenizer(
            combined_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # BERTでエンコード
        with torch.no_grad():
            outputs = self.bert(**inputs)
            cls_output = outputs.last_hidden_state[:, 0, :]  # CLSトークン
        
        # センチメントとインパクトを計算
        sentiment_logits = self.sentiment_head(cls_output)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        impact_score = self.impact_head(cls_output)
        
        # 特徴量を結合
        combined_features = torch.cat([
            cls_output,
            sentiment_probs,
            impact_score
        ], dim=-1)
        
        # 最終出力
        return self.output_projection(combined_features)


class SimpleNewsEncoder(nn.Module):
    """シンプルなニュースエンコーダー（フォールバック用）"""
    
    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        
        # 埋め込み層
        self.embedding = nn.Embedding(1000, 32)
        
        # LSTM
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        
        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(128, output_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_size, output_size)
        )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """キーワードベースの簡易エンコーディング"""
        # キーワード特徴抽出
        features = self._extract_features(texts)
        
        # LSTMでエンコード（features: [seq_len, feature_dim]）
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, seq_len, feature_dim]
        elif features.dim() == 3 and features.size(0) > 1:
            features = features[0].unsqueeze(0)  # 最初のバッチのみ使用
            
        lstm_out, _ = self.lstm(features)
        
        # 平均プーリング
        pooled = torch.mean(lstm_out, dim=1)
        
        # 出力
        return self.output_layer(pooled)
    
    def _extract_features(self, texts: List[str]) -> torch.Tensor:
        """キーワードベースの特徴抽出"""
        positive_keywords = ['増収', '増益', '上方修正', '好調', '拡大', '成長', '黒字']
        negative_keywords = ['減収', '減益', '下方修正', '低迷', '縮小', '赤字', '損失']
        
        # 簡易的なトークン化と特徴抽出
        feature_matrix = []
        for text in texts[:10]:  # 最大10ニュース
            features = torch.zeros(20)
            for i, word in enumerate(positive_keywords):
                if word in text:
                    features[i] = 1.0
            for i, word in enumerate(negative_keywords):
                if word in text:
                    features[10 + i] = -1.0
            feature_matrix.append(features)
        
        if not feature_matrix:
            feature_matrix = [torch.zeros(20)]
        
        # 埋め込みに変換
        features_tensor = torch.stack(feature_matrix)
        token_ids = torch.clamp(features_tensor.long() + 500, 0, 999)
        return self.embedding(token_ids)