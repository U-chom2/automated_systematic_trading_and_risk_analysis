"""
スクリーニングモジュールのデータモデル
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class GrowthCompany:
    """東証グロース企業"""
    company_code: str
    company_name: str
    ticker: str
    market_cap: int  # 時価総額（円）
    latest_ir_date: Optional[datetime] = None
    ir_url: Optional[str] = None
    sector: Optional[str] = None
    is_target_candidate: bool = False
    
    @property
    def market_cap_billion(self) -> float:
        """時価総額を億円単位で返す"""
        return self.market_cap / 100_000_000
    
    def is_within_market_cap_limit(self, limit_billion: float = 100) -> bool:
        """時価総額が指定額以下かチェック"""
        return self.market_cap_billion <= limit_billion
    
    def has_recent_ir(self, days: int = 30) -> bool:
        """指定日数以内にIRがあるかチェック"""
        if not self.latest_ir_date:
            return False
        days_diff = (datetime.now() - self.latest_ir_date).days
        return days_diff <= days


@dataclass
class ScreeningResult:
    """スクリーニング結果"""
    screening_date: datetime
    total_companies: int
    filtered_companies: int
    passed_companies: list[GrowthCompany]
    
    @property
    def pass_rate(self) -> float:
        """通過率を計算"""
        if self.total_companies == 0:
            return 0.0
        return self.filtered_companies / self.total_companies
    
    def to_csv_data(self) -> list[dict]:
        """CSV出力用のデータに変換"""
        return [
            {
                "company_code": company.company_code,
                "company_name": company.company_name,
                "ticker": company.ticker,
                "market_cap": company.market_cap,
                "market_cap_billion": company.market_cap_billion,
                "latest_ir_date": company.latest_ir_date.isoformat() if company.latest_ir_date else None,
                "ir_url": company.ir_url,
                "sector": company.sector,
                "screening_date": self.screening_date.isoformat()
            }
            for company in self.passed_companies
        ]