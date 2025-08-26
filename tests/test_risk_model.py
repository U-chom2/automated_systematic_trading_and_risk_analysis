"""RiskModelのテストケース"""

import pytest
import numpy as np
import torch
from typing import Dict, List, Any
from src.analysis_engine.risk_model import RiskModel


class TestRiskModel:
    """RiskModelのテストクラス"""
    
    @pytest.fixture
    def model(self) -> RiskModel:
        """テスト用のRiskModelインスタンスを作成"""
        return RiskModel()
    
    @pytest.fixture
    def sample_indicators(self) -> Dict[str, float]:
        """テスト用のリスク指標データを作成"""
        return {
            "atr": 50.0,
            "historical_volatility": 0.25,
            "volume_ratio": 1.5,
            "price_momentum": 2.3,
            "rsi": 65.0,
            "ma_deviation": 5.2
        }
    
    @pytest.fixture
    def sample_training_data(self) -> List[Dict[str, float]]:
        """テスト用の訓練データを作成"""
        np.random.seed(42)
        training_data = []
        
        for _ in range(100):
            # ランダムな特徴量を生成
            indicators = {
                "atr": np.random.uniform(10, 100),
                "historical_volatility": np.random.uniform(0.1, 0.5),
                "volume_ratio": np.random.uniform(0.5, 3.0),
                "price_momentum": np.random.uniform(-5, 5),
                "rsi": np.random.uniform(20, 80),
                "ma_deviation": np.random.uniform(-10, 10)
            }
            
            # ターゲット（損切り幅）を生成
            # ボラティリティに基づいて設定
            target_stop_loss = min(0.02 + indicators["historical_volatility"] * 0.2, 0.15)
            indicators["target_stop_loss"] = target_stop_loss
            
            training_data.append(indicators)
        
        return training_data
    
    def test_model_initialization(self, model: RiskModel) -> None:
        """モデル初期化のテスト"""
        assert model.input_size == 6  # 入力特徴量数
        assert model.hidden_size == 32
        assert model.output_size == 1  # 損切り幅の出力
        assert model.is_trained is False
        assert model.scaler is not None
    
    def test_prepare_features(self, model: RiskModel, sample_indicators: Dict[str, float]) -> None:
        """特徴量準備のテスト"""
        features = model.prepare_features(sample_indicators)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (6,)  # 6つの特徴量
        assert not np.isnan(features).any()  # NaNが含まれていない
        
        # 期待される順序で特徴量が並んでいることを確認
        expected_features = [
            sample_indicators["atr"],
            sample_indicators["historical_volatility"],
            sample_indicators["volume_ratio"],
            sample_indicators["price_momentum"],
            sample_indicators["rsi"],
            sample_indicators["ma_deviation"]
        ]
        np.testing.assert_array_equal(features, expected_features)
    
    def test_predict_before_training(self, model: RiskModel, sample_indicators: Dict[str, float]) -> None:
        """訓練前の予測のテスト（デフォルト値を返すことを確認）"""
        stop_loss_pct = model.predict(sample_indicators)
        
        assert isinstance(stop_loss_pct, float)
        assert 0.01 <= stop_loss_pct <= 0.15  # 1%〜15%の範囲
        # 訓練前はデフォルト値（8%）が返される
        assert stop_loss_pct == 0.08
    
    def test_train_model(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """モデル訓練のテスト"""
        initial_loss = model.train(sample_training_data, epochs=5, batch_size=16)
        
        assert isinstance(initial_loss, float)
        assert initial_loss > 0  # 損失は正の値
        assert model.is_trained is True
        
        # 訓練後に予測できることを確認
        test_indicators = sample_training_data[0].copy()
        del test_indicators["target_stop_loss"]  # ターゲットを削除
        
        prediction = model.predict(test_indicators)
        assert isinstance(prediction, float)
        assert 0.01 <= prediction <= 0.15
    
    def test_predict_after_training(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """訓練後の予測のテスト"""
        # モデルを訓練
        model.train(sample_training_data, epochs=10, batch_size=16)
        
        # 予測を実行
        test_indicators = sample_training_data[0].copy()
        del test_indicators["target_stop_loss"]
        
        prediction = model.predict(test_indicators)
        
        assert isinstance(prediction, float)
        assert 0.01 <= prediction <= 0.15
        assert prediction != 0.08  # デフォルト値でない（訓練済み）
    
    def test_calculate_default_stop_loss(self, model: RiskModel) -> None:
        """デフォルト損切り幅計算のテスト"""
        # 高ボラティリティのケース
        high_vol_indicators = {
            "historical_volatility": 0.4,
            "atr": 80.0,
            "volume_ratio": 2.0,
            "price_momentum": 3.0,
            "rsi": 70.0,
            "ma_deviation": 8.0
        }
        high_vol_stop_loss = model.calculate_default_stop_loss(high_vol_indicators)
        assert high_vol_stop_loss > 0.08  # 高ボラティリティは高い損切り幅
        
        # 低ボラティリティのケース
        low_vol_indicators = {
            "historical_volatility": 0.15,
            "atr": 30.0,
            "volume_ratio": 1.0,
            "price_momentum": 0.5,
            "rsi": 50.0,
            "ma_deviation": 2.0
        }
        low_vol_stop_loss = model.calculate_default_stop_loss(low_vol_indicators)
        assert low_vol_stop_loss < high_vol_stop_loss  # 低ボラティリティは低い損切り幅
    
    def test_batch_predict(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """バッチ予測のテスト"""
        # モデルを訓練
        model.train(sample_training_data, epochs=5, batch_size=16)
        
        # バッチ予測用データ準備
        batch_indicators = []
        for data in sample_training_data[:10]:
            indicators = data.copy()
            del indicators["target_stop_loss"]
            batch_indicators.append(indicators)
        
        predictions = model.batch_predict(batch_indicators)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 10
        for pred in predictions:
            assert isinstance(pred, float)
            assert 0.01 <= pred <= 0.15
    
    def test_model_validation(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """モデル検証のテスト"""
        # 訓練データとテストデータに分割
        train_data = sample_training_data[:80]
        test_data = sample_training_data[80:]
        
        # 訓練
        model.train(train_data, epochs=10)
        
        # 検証
        validation_score = model.validate(test_data)
        
        assert isinstance(validation_score, dict)
        assert "mae" in validation_score  # Mean Absolute Error
        assert "rmse" in validation_score  # Root Mean Square Error
        assert "mape" in validation_score  # Mean Absolute Percentage Error
        
        assert validation_score["mae"] >= 0
        assert validation_score["rmse"] >= 0
        assert validation_score["mape"] >= 0
    
    def test_save_and_load_model(self, model: RiskModel, sample_training_data: List[Dict[str, float]], tmp_path) -> None:
        """モデル保存・読み込みのテスト"""
        # 訓練
        model.train(sample_training_data, epochs=5)
        
        # 予測値を保存
        test_indicators = sample_training_data[0].copy()
        del test_indicators["target_stop_loss"]
        original_prediction = model.predict(test_indicators)
        
        # モデル保存
        model_path = tmp_path / "risk_model.pth"
        model.save_model(str(model_path))
        
        # 新しいモデルインスタンス作成
        new_model = RiskModel()
        
        # モデル読み込み
        new_model.load_model(str(model_path))
        
        # 同じ予測値が得られることを確認
        loaded_prediction = new_model.predict(test_indicators)
        assert abs(original_prediction - loaded_prediction) < 1e-6
        assert new_model.is_trained is True
    
    def test_feature_importance(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """特徴量重要度のテスト"""
        # 訓練
        model.train(sample_training_data, epochs=10)
        
        # 特徴量重要度を計算（簡易版）
        importance = model.get_feature_importance(sample_training_data[:20])
        
        assert isinstance(importance, dict)
        expected_features = ["atr", "historical_volatility", "volume_ratio", 
                           "price_momentum", "rsi", "ma_deviation"]
        
        for feature in expected_features:
            assert feature in importance
            assert isinstance(importance[feature], float)
    
    def test_error_handling(self, model: RiskModel) -> None:
        """エラーハンドリングのテスト"""
        # 空の指標辞書
        with pytest.raises(ValueError):
            model.predict({})
        
        # 必要な特徴量が不足
        incomplete_indicators = {"atr": 50.0}
        with pytest.raises(ValueError):
            model.predict(incomplete_indicators)
        
        # 空の訓練データ
        with pytest.raises(ValueError):
            model.train([])
    
    def test_model_performance_bounds(self, model: RiskModel, sample_training_data: List[Dict[str, float]]) -> None:
        """モデルパフォーマンスの境界テスト"""
        model.train(sample_training_data, epochs=20)
        
        # 極端な値でのテスト
        extreme_indicators = {
            "atr": 200.0,  # 非常に高いATR
            "historical_volatility": 0.8,  # 非常に高いボラティリティ
            "volume_ratio": 10.0,  # 非常に高い出来高比
            "price_momentum": 20.0,  # 非常に高いモメンタム
            "rsi": 95.0,  # 過度に高いRSI
            "ma_deviation": 50.0  # 非常に高い移動平均乖離
        }
        
        extreme_prediction = model.predict(extreme_indicators)
        assert 0.01 <= extreme_prediction <= 0.15  # 境界内に収まることを確認