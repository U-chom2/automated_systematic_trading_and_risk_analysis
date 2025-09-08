"""ユースケース

アプリケーション層のビジネスロジックを実装。
ドメイン層を利用して具体的な業務処理を行う。
"""
from .portfolio_management import (
    CreatePortfolioUseCase,
    UpdatePortfolioUseCase,
    GetPortfolioUseCase,
    ListPortfoliosUseCase,
    DeletePortfolioUseCase,
)
from .trading import (
    ExecuteTradeUseCase,
    CancelTradeUseCase,
    GetTradeHistoryUseCase,
    GetPendingTradesUseCase,
)
from .signal_processing import (
    GenerateSignalsUseCase,
    ProcessSignalUseCase,
    GetActiveSignalsUseCase,
)
from .risk_management import (
    CalculateRiskMetricsUseCase,
    CheckRiskLimitsUseCase,
    GetPositionRisksUseCase,
)
from .backtesting import (
    RunBacktestUseCase,
    AnalyzeBacktestResultsUseCase,
)
from .ai_training import (
    TrainAIModelUseCase,
    EvaluateModelUseCase,
    PredictWithModelUseCase,
)

__all__ = [
    # Portfolio Management
    "CreatePortfolioUseCase",
    "UpdatePortfolioUseCase",
    "GetPortfolioUseCase",
    "ListPortfoliosUseCase",
    "DeletePortfolioUseCase",
    # Trading
    "ExecuteTradeUseCase",
    "CancelTradeUseCase",
    "GetTradeHistoryUseCase",
    "GetPendingTradesUseCase",
    # Signal Processing
    "GenerateSignalsUseCase",
    "ProcessSignalUseCase",
    "GetActiveSignalsUseCase",
    # Risk Management
    "CalculateRiskMetricsUseCase",
    "CheckRiskLimitsUseCase",
    "GetPositionRisksUseCase",
    # Backtesting
    "RunBacktestUseCase",
    "AnalyzeBacktestResultsUseCase",
    # AI Training
    "TrainAIModelUseCase",
    "EvaluateModelUseCase",
    "PredictWithModelUseCase",
]