"""アプリケーション層ユースケースのユニットテスト"""
import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

from src.application.use_cases.portfolio_management import (
    CreatePortfolioUseCase,
    GetPortfolioUseCase,
)
from src.application.dto.portfolio_dto import CreatePortfolioDTO, PortfolioDTO


class TestCreatePortfolioUseCase:
    """ポートフォリオ作成ユースケースのテスト"""
    
    @pytest.mark.asyncio
    async def test_create_portfolio_success(self):
        """正常なポートフォリオ作成"""
        # モックリポジトリを作成
        mock_repo = Mock()
        mock_repo.save = AsyncMock()
        
        # ユースケースを作成
        use_case = CreatePortfolioUseCase(mock_repo)
        
        # DTOを作成
        dto = CreatePortfolioDTO(
            name="Test Portfolio",
            initial_capital=Decimal("10000000"),
            description="Test portfolio for testing"
        )
        
        # 実行
        result = await use_case.execute(dto)
        
        # 検証
        assert result is not None
        assert isinstance(result, PortfolioDTO)
        mock_repo.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_portfolio_invalid_capital(self):
        """無効な初期資金でのポートフォリオ作成"""
        mock_repo = Mock()
        use_case = CreatePortfolioUseCase(mock_repo)
        
        # 無効なDTOを作成（負の初期資金）
        dto = CreatePortfolioDTO(
            name="Test Portfolio",
            initial_capital=Decimal("-1000"),
            description="Invalid portfolio"
        )
        
        # バリデーションエラーが発生することを確認
        with pytest.raises(ValueError):
            dto.validate()


class TestGetPortfolioUseCase:
    """ポートフォリオ取得ユースケースのテスト"""
    
    @pytest.mark.asyncio
    async def test_get_portfolio_success(self):
        """正常なポートフォリオ取得"""
        # モックエンティティを作成
        portfolio_id = uuid4()
        mock_portfolio = Mock()
        mock_portfolio.id = portfolio_id
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = Decimal("10000000")
        mock_portfolio.current_capital = Decimal("10500000")
        mock_portfolio.is_active = True
        mock_portfolio.positions = []
        mock_portfolio.created_at = datetime.utcnow()
        mock_portfolio.updated_at = datetime.utcnow()
        
        # モックリポジトリを作成
        mock_portfolio_repo = Mock()
        mock_portfolio_repo.find_by_id = AsyncMock(return_value=mock_portfolio)
        
        mock_market_repo = Mock()
        
        # ユースケースを作成
        use_case = GetPortfolioUseCase(mock_portfolio_repo, mock_market_repo)
        
        # 実行
        result = await use_case.execute(
            portfolio_id,
            include_positions=True,
            update_prices=False
        )
        
        # 検証
        assert result is not None
        assert isinstance(result, PortfolioDTO)
        assert result.id == portfolio_id
        mock_portfolio_repo.find_by_id.assert_called_once_with(portfolio_id)
    
    @pytest.mark.asyncio
    async def test_get_portfolio_not_found(self):
        """存在しないポートフォリオの取得"""
        portfolio_id = uuid4()
        
        # モックリポジトリを作成（Noneを返す）
        mock_portfolio_repo = Mock()
        mock_portfolio_repo.find_by_id = AsyncMock(return_value=None)
        
        mock_market_repo = Mock()
        
        # ユースケースを作成
        use_case = GetPortfolioUseCase(mock_portfolio_repo, mock_market_repo)
        
        # 実行
        result = await use_case.execute(
            portfolio_id,
            include_positions=True,
            update_prices=False
        )
        
        # 検証
        assert result is None
        mock_portfolio_repo.find_by_id.assert_called_once_with(portfolio_id)