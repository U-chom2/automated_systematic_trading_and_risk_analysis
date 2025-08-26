"""WorkflowManagerのテストケース"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from enum import Enum

from src.system_core.workflow_manager import (
    WorkflowManager, TradingPhase, WorkflowState, 
    TriggerEvent, AnalysisResult, ExecutionResult
)
from src.analysis_engine.nlp_analyzer import NlpAnalyzer
from src.analysis_engine.technical_analyzer import TechnicalAnalyzer
from src.analysis_engine.risk_model import RiskModel


class TestWorkflowManager:
    """WorkflowManagerのテストクラス"""
    
    @pytest.fixture
    def workflow_manager(self) -> WorkflowManager:
        """テスト用WorkflowManagerインスタンス"""
        config = {
            "scoring_threshold": 80,
            "rsi_filter_threshold": 75,
            "ma_deviation_filter_threshold": 25,
            "processing_timeout": 30
        }
        return WorkflowManager(config=config)
    
    @pytest.fixture
    def sample_ir_trigger(self) -> TriggerEvent:
        """テスト用IRトリガー"""
        return TriggerEvent(
            trigger_type="IR",
            symbol="7203",
            title="業績の上方修正に関するお知らせ",
            content="当社は、業績予想の上方修正を発表いたします。売上高は前回予想を10%上回る見込みです。",
            timestamp=datetime.now(),
            source="TDnet",
            importance_score=50
        )
    
    @pytest.fixture
    def sample_sns_trigger(self) -> TriggerEvent:
        """テスト用SNSトリガー"""
        return TriggerEvent(
            trigger_type="SNS",
            symbol="7203",
            title="言及数異常検知",
            content="過去24時間の言及数が3σを超過",
            timestamp=datetime.now(),
            source="X_Streaming",
            mention_count=150,
            baseline_count=25
        )
    
    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """テスト用市場データ"""
        return {
            "symbol": "7203",
            "current_price": Decimal("2500"),
            "volume": 1000000,
            "rsi": 65.0,
            "ma_deviation": 15.0,
            "atr": 80.0,
            "historical_volatility": 0.25,
            "timestamp": datetime.now()
        }
    
    def test_workflow_manager_initialization(self, workflow_manager: WorkflowManager) -> None:
        """WorkflowManager初期化のテスト"""
        assert workflow_manager is not None
        assert workflow_manager.current_phase == TradingPhase.IDLE
        assert workflow_manager.state == WorkflowState.WAITING
        assert workflow_manager.config["scoring_threshold"] == 80
        assert workflow_manager.nlp_analyzer is not None
        assert workflow_manager.technical_analyzer is not None
        assert workflow_manager.risk_model is not None
    
    @pytest.mark.asyncio
    async def test_phase_2_trigger_processing(self, workflow_manager: WorkflowManager, 
                                              sample_ir_trigger: TriggerEvent) -> None:
        """フェーズ2：トリガー処理のテスト"""
        # IRトリガーの処理
        result = await workflow_manager.process_trigger(sample_ir_trigger)
        
        assert result is not None
        assert result.trigger_activated is True
        assert result.symbol == "7203"
        assert result.trigger_type == "IR"
        assert result.timestamp is not None
        assert workflow_manager.current_phase == TradingPhase.TRIGGER_DETECTED
    
    @pytest.mark.asyncio
    async def test_phase_2_sns_trigger_processing(self, workflow_manager: WorkflowManager,
                                                  sample_sns_trigger: TriggerEvent) -> None:
        """フェーズ2：SNSトリガー処理のテスト"""
        # SNSトリガーの処理
        result = await workflow_manager.process_trigger(sample_sns_trigger)
        
        assert result is not None
        assert result.trigger_activated is True
        assert result.symbol == "7203"
        assert result.trigger_type == "SNS"
        assert result.anomaly_detected is True
        assert result.z_score > 3.0  # 3σ超過
    
    @pytest.mark.asyncio
    async def test_phase_3_analysis_and_decision(self, workflow_manager: WorkflowManager,
                                                 sample_ir_trigger: TriggerEvent,
                                                 sample_market_data: Dict[str, Any]) -> None:
        """フェーズ3：分析と判断のテスト"""
        # トリガー処理
        trigger_result = await workflow_manager.process_trigger(sample_ir_trigger)
        assert trigger_result.trigger_activated is True
        
        # 分析と判断
        analysis_result = await workflow_manager.analyze_and_decide(sample_ir_trigger, sample_market_data)
        
        assert analysis_result is not None
        assert "catalyst_importance_score" in analysis_result
        assert "sentiment_score" in analysis_result
        assert "market_environment_score" in analysis_result
        assert "total_score" in analysis_result
        assert "buy_decision" in analysis_result
        assert "filter_passed" in analysis_result
        
        # スコア範囲の確認
        assert 0 <= analysis_result["catalyst_importance_score"] <= 50
        assert 0 <= analysis_result["sentiment_score"] <= 30
        assert 0 <= analysis_result["market_environment_score"] <= 20
        assert 0 <= analysis_result["total_score"] <= 100
        
        assert workflow_manager.current_phase == TradingPhase.ANALYSIS_COMPLETE
    
    @pytest.mark.asyncio
    async def test_scoring_system(self, workflow_manager: WorkflowManager,
                                  sample_ir_trigger: TriggerEvent,
                                  sample_market_data: Dict[str, Any]) -> None:
        """スコアリングシステムのテスト"""
        analysis_result = await workflow_manager.analyze_and_decide(sample_ir_trigger, sample_market_data)
        
        # S級キーワード「上方修正」により高スコア期待
        assert analysis_result["catalyst_importance_score"] >= 40
        
        # 総合スコア80点以上で買い判断
        if analysis_result["total_score"] >= 80:
            assert analysis_result["buy_decision"] is True
        else:
            assert analysis_result["buy_decision"] is False
    
    @pytest.mark.asyncio
    async def test_risk_filter_application(self, workflow_manager: WorkflowManager) -> None:
        """リスクフィルター適用のテスト"""
        # 過熱状態の市場データ
        overheated_data = {
            "symbol": "7203",
            "current_price": Decimal("2500"),
            "rsi": 85.0,           # RSI > 75（フィルター条件）
            "ma_deviation": 30.0,  # 乖離率 > 25%（フィルター条件）
            "volume": 1000000
        }
        
        filter_result = workflow_manager.check_risk_filters(overheated_data)
        
        assert filter_result["passed"] is False
        assert "rsi_overheated" in filter_result["reasons"]
        assert "ma_deviation_overheated" in filter_result["reasons"]
        
        # 正常状態の市場データ
        normal_data = {
            "symbol": "7203",
            "current_price": Decimal("2500"),
            "rsi": 55.0,          # RSI < 75
            "ma_deviation": 10.0, # 乖離率 < 25%
            "volume": 1000000
        }
        
        filter_result = workflow_manager.check_risk_filters(normal_data)
        
        assert filter_result["passed"] is True
        assert len(filter_result["reasons"]) == 0
    
    @pytest.mark.asyncio
    async def test_phase_4_execution_preparation(self, workflow_manager: WorkflowManager,
                                                 sample_market_data: Dict[str, Any]) -> None:
        """フェーズ4：実行準備のテスト"""
        # 高スコア分析結果をモック
        analysis_result = {
            "total_score": 85,
            "buy_decision": True,
            "filter_passed": True,
            "risk_indicators": {
                "atr": 80.0,
                "historical_volatility": 0.25,
                "volume_ratio": 1.5,
                "price_momentum": 2.3,
                "rsi": 65.0,
                "ma_deviation": 15.0
            }
        }
        
        execution_prep = await workflow_manager.prepare_execution(sample_market_data, analysis_result)
        
        assert execution_prep is not None
        assert "position_size" in execution_prep
        assert "stop_loss_percentage" in execution_prep
        assert "take_profit_percentage" in execution_prep
        assert "entry_price" in execution_prep
        
        assert execution_prep["position_size"] > 0
        assert 0.01 <= execution_prep["stop_loss_percentage"] <= 0.15
        assert execution_prep["take_profit_percentage"] > 0
        assert execution_prep["entry_price"] == sample_market_data["current_price"]
    
    def test_workflow_state_transitions(self, workflow_manager: WorkflowManager) -> None:
        """ワークフロー状態遷移のテスト"""
        # 初期状態
        assert workflow_manager.state == WorkflowState.WAITING
        assert workflow_manager.current_phase == TradingPhase.IDLE
        
        # 状態遷移テスト
        workflow_manager._transition_to_phase(TradingPhase.TRIGGER_DETECTED)
        assert workflow_manager.current_phase == TradingPhase.TRIGGER_DETECTED
        assert workflow_manager.state == WorkflowState.PROCESSING
        
        workflow_manager._transition_to_phase(TradingPhase.ANALYSIS_COMPLETE)
        assert workflow_manager.current_phase == TradingPhase.ANALYSIS_COMPLETE
        
        workflow_manager._transition_to_phase(TradingPhase.EXECUTION_COMPLETE)
        assert workflow_manager.current_phase == TradingPhase.EXECUTION_COMPLETE
        
        # 完了後はIDLEに戻る
        workflow_manager._reset_workflow()
        assert workflow_manager.current_phase == TradingPhase.IDLE
        assert workflow_manager.state == WorkflowState.WAITING
    
    @pytest.mark.asyncio
    async def test_concurrent_trigger_processing(self, workflow_manager: WorkflowManager) -> None:
        """並行トリガー処理のテスト"""
        # 複数のトリガーを同時処理
        triggers = [
            TriggerEvent(
                trigger_type="IR", symbol="7203", title="IR1", 
                content="テスト", timestamp=datetime.now(), source="TDnet", importance_score=40
            ),
            TriggerEvent(
                trigger_type="IR", symbol="6758", title="IR2",
                content="テスト", timestamp=datetime.now(), source="TDnet", importance_score=45
            ),
            TriggerEvent(
                trigger_type="SNS", symbol="9984", title="SNS1",
                content="テスト", timestamp=datetime.now(), source="X_Streaming", mention_count=100
            )
        ]
        
        results = await workflow_manager.process_multiple_triggers(triggers)
        
        assert len(results) == 3
        for result in results:
            assert result.trigger_activated is True
            assert result.symbol in ["7203", "6758", "9984"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, workflow_manager: WorkflowManager) -> None:
        """タイムアウトハンドリングのテスト"""
        # 長時間かかる処理をモック
        slow_trigger = TriggerEvent(
            trigger_type="IR", symbol="7203", title="Slow Process",
            content="テスト", timestamp=datetime.now(), source="TDnet", importance_score=40
        )
        
        with patch.object(workflow_manager.nlp_analyzer, 'analyze_ir_importance') as mock_analyze:
            # 30秒以上かかる処理をシミュレート
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(35)
                return {"score": 40, "keywords": ["上方修正"]}
            
            mock_analyze.side_effect = slow_process
            
            # タイムアウト設定（5秒）
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    workflow_manager.process_trigger(slow_trigger),
                    timeout=5.0
                )
    
    def test_workflow_error_recovery(self, workflow_manager: WorkflowManager) -> None:
        """ワークフローエラー復旧のテスト"""
        # エラー状態にする
        workflow_manager._set_error_state("Test error")
        assert workflow_manager.state == WorkflowState.ERROR
        
        # エラー復旧
        recovery_result = workflow_manager.recover_from_error()
        assert recovery_result is True
        assert workflow_manager.state == WorkflowState.WAITING
        assert workflow_manager.current_phase == TradingPhase.IDLE
    
    def test_workflow_metrics_collection(self, workflow_manager: WorkflowManager) -> None:
        """ワークフローメトリクス収集のテスト"""
        metrics = workflow_manager.get_workflow_metrics()
        
        assert "total_triggers_processed" in metrics
        assert "successful_analyses" in metrics
        assert "failed_analyses" in metrics
        assert "average_processing_time" in metrics
        assert "current_phase" in metrics
        assert "current_state" in metrics
        
        # 初期値確認
        assert metrics["total_triggers_processed"] == 0
        assert metrics["successful_analyses"] == 0
        assert metrics["current_phase"] == TradingPhase.IDLE.value
        assert metrics["current_state"] == WorkflowState.WAITING.value
    
    @pytest.mark.asyncio
    async def test_workflow_persistence(self, workflow_manager: WorkflowManager,
                                       sample_ir_trigger: TriggerEvent) -> None:
        """ワークフロー永続化のテスト"""
        # ワークフロー実行
        await workflow_manager.process_trigger(sample_ir_trigger)
        
        # 状態保存
        save_result = workflow_manager.save_workflow_state()
        assert save_result is True
        
        # 新しいインスタンスで状態復元
        new_workflow = WorkflowManager(config=workflow_manager.config)
        restore_result = new_workflow.restore_workflow_state()
        
        assert restore_result is True
        # 復元確認（実装依存）
    
    def test_workflow_configuration_update(self, workflow_manager: WorkflowManager) -> None:
        """ワークフロー設定更新のテスト"""
        original_threshold = workflow_manager.config["scoring_threshold"]
        
        # 設定更新
        new_config = {
            "scoring_threshold": 90,
            "rsi_filter_threshold": 70
        }
        
        update_result = workflow_manager.update_configuration(new_config)
        
        assert update_result is True
        assert workflow_manager.config["scoring_threshold"] == 90
        assert workflow_manager.config["rsi_filter_threshold"] == 70
        assert workflow_manager.config["scoring_threshold"] != original_threshold
    
    def test_workflow_validation(self, workflow_manager: WorkflowManager) -> None:
        """ワークフローバリデーションのテスト"""
        # 有効なトリガーイベント
        valid_trigger = TriggerEvent(
            trigger_type="IR", symbol="7203", title="Valid IR",
            content="テスト", timestamp=datetime.now(), source="TDnet", importance_score=40
        )
        
        validation_result = workflow_manager.validate_trigger(valid_trigger)
        assert validation_result is True
        
        # 無効なトリガーイベント
        invalid_trigger = TriggerEvent(
            trigger_type="INVALID", symbol="", title="",
            content="", timestamp=None, source="", importance_score=-1
        )
        
        validation_result = workflow_manager.validate_trigger(invalid_trigger)
        assert validation_result is False