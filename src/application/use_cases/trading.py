"""取引ユースケース"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from ...domain.entities.trade import Trade, OrderStatus
from ...domain.entities.portfolio import Portfolio
from ...domain.repositories.trade_repository import TradeRepository
from ...domain.repositories.portfolio_repository import PortfolioRepository
from ...domain.repositories.market_data_repository import MarketDataRepository
from ..dto.trade_dto import TradeDTO, CreateTradeDTO, TradeExecutionDTO, TradeSummaryDTO


@dataclass
class ExecuteTradeUseCase:
    """取引実行ユースケース"""
    
    trade_repository: TradeRepository
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(self, dto: CreateTradeDTO) -> TradeDTO:
        """取引を実行
        
        Args:
            dto: 取引作成DTO
        
        Returns:
            実行された取引DTO
        """
        # バリデーション
        dto.validate()
        
        # ポートフォリオの存在確認
        portfolio = await self.portfolio_repository.find_by_id(dto.portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {dto.portfolio_id}")
        
        # 利用可能資金の確認（買い注文の場合）
        if dto.trade_type in ["BUY", "COVER"]:
            # 最新価格を取得
            latest_price = await self.market_data_repository.get_latest_price(dto.ticker)
            if not latest_price:
                raise ValueError(f"Cannot get price for {dto.ticker}")
            
            order_price = dto.order_price or latest_price.value
            required_amount = order_price * Decimal(dto.quantity)
            
            if portfolio.available_cash < required_amount:
                raise ValueError(
                    f"Insufficient funds. Required: {required_amount}, "
                    f"Available: {portfolio.available_cash}"
                )
        
        # 売り注文の場合はポジションの確認
        if dto.trade_type in ["SELL", "SHORT"]:
            position = await self.portfolio_repository.find_position(
                dto.portfolio_id, dto.ticker
            )
            if not position or position.is_closed:
                raise ValueError(f"No open position for {dto.ticker}")
            
            if position.quantity < dto.quantity:
                raise ValueError(
                    f"Insufficient shares. Available: {position.quantity}, "
                    f"Requested: {dto.quantity}"
                )
        
        # 取引エンティティを作成
        trade = dto.to_entity()
        
        # リポジトリで保存
        saved_trade = await self.trade_repository.save(trade)
        
        # 自動的に送信状態にする
        saved_trade.submit()
        await self.trade_repository.update(saved_trade)
        
        # DTOに変換して返却
        return TradeDTO.from_entity(saved_trade)


@dataclass
class CancelTradeUseCase:
    """取引キャンセルユースケース"""
    
    trade_repository: TradeRepository
    
    async def execute(self, trade_id: UUID) -> bool:
        """取引をキャンセル
        
        Args:
            trade_id: 取引ID
        
        Returns:
            キャンセル成功の可否
        """
        # 取引を取得
        trade = await self.trade_repository.find_by_id(trade_id)
        if not trade:
            raise ValueError(f"Trade not found: {trade_id}")
        
        # キャンセル可能か確認
        if not trade.is_pending:
            raise ValueError(
                f"Cannot cancel trade with status {trade.status.value}"
            )
        
        # キャンセル実行
        return await self.trade_repository.cancel(trade_id)


@dataclass
class GetTradeHistoryUseCase:
    """取引履歴取得ユースケース"""
    
    trade_repository: TradeRepository
    
    async def execute(
        self,
        portfolio_id: Optional[UUID] = None,
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TradeDTO]:
        """取引履歴を取得
        
        Args:
            portfolio_id: ポートフォリオID（オプション）
            ticker: ティッカーシンボル（オプション）
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
            limit: 取得件数上限
            offset: オフセット
        
        Returns:
            取引DTOのリスト
        """
        trades = []
        
        if start_date and end_date:
            # 日付範囲で検索
            trades = await self.trade_repository.find_by_date_range(
                start_date, end_date, portfolio_id
            )
        elif portfolio_id and ticker:
            # ポートフォリオとティッカーで検索
            trades = await self.trade_repository.find_by_ticker(
                ticker, portfolio_id, limit, offset
            )
        elif portfolio_id:
            # ポートフォリオで検索
            trades = await self.trade_repository.find_by_portfolio(
                portfolio_id, limit, offset
            )
        elif ticker:
            # ティッカーで検索
            trades = await self.trade_repository.find_by_ticker(
                ticker, None, limit, offset
            )
        
        # DTOに変換して返却
        return [TradeDTO.from_entity(trade) for trade in trades]


@dataclass
class GetPendingTradesUseCase:
    """待機中取引取得ユースケース"""
    
    trade_repository: TradeRepository
    
    async def execute(
        self,
        portfolio_id: Optional[UUID] = None,
    ) -> List[TradeDTO]:
        """待機中の取引を取得
        
        Args:
            portfolio_id: ポートフォリオID（オプション）
        
        Returns:
            待機中の取引DTOのリスト
        """
        trades = await self.trade_repository.find_pending_trades(portfolio_id)
        
        # DTOに変換して返却
        return [TradeDTO.from_entity(trade) for trade in trades]


@dataclass
class ProcessTradeExecutionUseCase:
    """取引約定処理ユースケース"""
    
    trade_repository: TradeRepository
    portfolio_repository: PortfolioRepository
    
    async def execute(self, dto: TradeExecutionDTO) -> TradeDTO:
        """取引の約定を処理
        
        Args:
            dto: 取引約定DTO
        
        Returns:
            約定済み取引DTO
        """
        # バリデーション
        dto.validate()
        
        # 取引を取得
        trade = await self.trade_repository.find_by_id(dto.trade_id)
        if not trade:
            raise ValueError(f"Trade not found: {dto.trade_id}")
        
        # 約定可能か確認
        if not trade.is_pending:
            raise ValueError(
                f"Cannot execute trade with status {trade.status.value}"
            )
        
        # 取引を約定
        trade.execute(dto.executed_price, dto.commission)
        trade.execution_id = dto.execution_id
        trade.executed_at = dto.executed_at
        
        # リポジトリで更新
        updated_trade = await self.trade_repository.update(trade)
        
        # ポートフォリオのポジションを更新
        portfolio = await self.portfolio_repository.find_by_id(trade.portfolio_id)
        if portfolio:
            if trade.is_buy:
                # 買い注文の場合はポジションを追加/更新
                portfolio.add_position(
                    stock_id=trade.stock_id,
                    ticker=trade.ticker,
                    quantity=trade.quantity,
                    price=dto.executed_price,
                )
            else:
                # 売り注文の場合はポジションを減少
                portfolio.reduce_position(
                    ticker=trade.ticker,
                    quantity=trade.quantity,
                    price=dto.executed_price,
                )
            
            # ポートフォリオを更新
            await self.portfolio_repository.update(portfolio)
        
        # DTOに変換して返却
        return TradeDTO.from_entity(updated_trade)


@dataclass
class GetTradeStatisticsUseCase:
    """取引統計取得ユースケース"""
    
    trade_repository: TradeRepository
    
    async def execute(
        self,
        portfolio_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> TradeSummaryDTO:
        """取引統計を取得
        
        Args:
            portfolio_id: ポートフォリオID
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            取引サマリーDTO
        """
        # 取引を取得
        if start_date and end_date:
            trades = await self.trade_repository.find_by_date_range(
                start_date, end_date, portfolio_id
            )
        else:
            trades = await self.trade_repository.find_by_portfolio(
                portfolio_id, limit=10000
            )
        
        # サマリーDTOを作成
        return TradeSummaryDTO.from_trades(trades)


@dataclass
class BatchExecuteTradesUseCase:
    """一括取引実行ユースケース"""
    
    trade_repository: TradeRepository
    portfolio_repository: PortfolioRepository
    market_data_repository: MarketDataRepository
    
    async def execute(self, trades: List[CreateTradeDTO]) -> List[TradeDTO]:
        """複数の取引を一括実行
        
        Args:
            trades: 取引作成DTOのリスト
        
        Returns:
            実行された取引DTOのリスト
        """
        executed_trades = []
        
        # 各取引を検証
        for dto in trades:
            dto.validate()
        
        # ポートフォリオごとにグループ化
        trades_by_portfolio = {}
        for dto in trades:
            if dto.portfolio_id not in trades_by_portfolio:
                trades_by_portfolio[dto.portfolio_id] = []
            trades_by_portfolio[dto.portfolio_id].append(dto)
        
        # ポートフォリオごとに処理
        for portfolio_id, portfolio_trades in trades_by_portfolio.items():
            # ポートフォリオを取得
            portfolio = await self.portfolio_repository.find_by_id(portfolio_id)
            if not portfolio:
                continue
            
            # 利用可能資金を追跡
            available_cash = portfolio.available_cash
            
            for dto in portfolio_trades:
                try:
                    # 買い注文の資金確認
                    if dto.trade_type in ["BUY", "COVER"]:
                        latest_price = await self.market_data_repository.get_latest_price(dto.ticker)
                        if latest_price:
                            order_price = dto.order_price or latest_price.value
                            required_amount = order_price * Decimal(dto.quantity)
                            
                            if available_cash < required_amount:
                                continue  # スキップ
                            
                            available_cash -= required_amount
                    
                    # 取引を作成
                    trade = dto.to_entity()
                    saved_trade = await self.trade_repository.save(trade)
                    saved_trade.submit()
                    await self.trade_repository.update(saved_trade)
                    
                    executed_trades.append(TradeDTO.from_entity(saved_trade))
                    
                except Exception:
                    # エラーが発生した取引はスキップ
                    continue
        
        return executed_trades