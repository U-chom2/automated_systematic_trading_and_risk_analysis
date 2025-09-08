"""カスタム例外クラス"""
from typing import Optional, Dict, Any
from uuid import UUID


class BaseException(Exception):
    """基底例外クラス"""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# ドメイン層の例外
class DomainException(BaseException):
    """ドメイン層の例外基底クラス"""
    pass


class EntityNotFoundException(DomainException):
    """エンティティが見つからない"""
    
    def __init__(self, entity_type: str, entity_id: UUID):
        super().__init__(
            f"{entity_type} not found: {entity_id}",
            code="ENTITY_NOT_FOUND",
            details={"entity_type": entity_type, "entity_id": str(entity_id)},
        )


class InsufficientFundsException(DomainException):
    """資金不足"""
    
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient funds: required={required}, available={available}",
            code="INSUFFICIENT_FUNDS",
            details={"required": required, "available": available},
        )


class InvalidTradeException(DomainException):
    """無効な取引"""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Invalid trade: {reason}",
            code="INVALID_TRADE",
            details={"reason": reason},
        )


class PositionNotFoundException(DomainException):
    """ポジションが見つからない"""
    
    def __init__(self, ticker: str):
        super().__init__(
            f"Position not found for ticker: {ticker}",
            code="POSITION_NOT_FOUND",
            details={"ticker": ticker},
        )


class RiskLimitExceededException(DomainException):
    """リスク限度超過"""
    
    def __init__(self, risk_type: str, limit: float, actual: float):
        super().__init__(
            f"Risk limit exceeded: {risk_type} (limit={limit}, actual={actual})",
            code="RISK_LIMIT_EXCEEDED",
            details={"risk_type": risk_type, "limit": limit, "actual": actual},
        )


# アプリケーション層の例外
class ApplicationException(BaseException):
    """アプリケーション層の例外基底クラス"""
    pass


class UseCaseException(ApplicationException):
    """ユースケース実行エラー"""
    
    def __init__(self, use_case: str, reason: str):
        super().__init__(
            f"Use case failed: {use_case} - {reason}",
            code="USE_CASE_FAILED",
            details={"use_case": use_case, "reason": reason},
        )


class ValidationException(ApplicationException):
    """バリデーションエラー"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for {field}: {reason}",
            code="VALIDATION_ERROR",
            details={"field": field, "value": value, "reason": reason},
        )


class AuthenticationException(ApplicationException):
    """認証エラー"""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Authentication failed: {reason}",
            code="AUTHENTICATION_FAILED",
            details={"reason": reason},
        )


class AuthorizationException(ApplicationException):
    """認可エラー"""
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            f"Not authorized to {action} on {resource}",
            code="AUTHORIZATION_FAILED",
            details={"resource": resource, "action": action},
        )


# インフラストラクチャ層の例外
class InfrastructureException(BaseException):
    """インフラストラクチャ層の例外基底クラス"""
    pass


class DatabaseException(InfrastructureException):
    """データベースエラー"""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Database operation failed: {operation} - {reason}",
            code="DATABASE_ERROR",
            details={"operation": operation, "reason": reason},
        )


class ExternalAPIException(InfrastructureException):
    """外部APIエラー"""
    
    def __init__(self, api: str, status_code: Optional[int] = None, reason: str = ""):
        super().__init__(
            f"External API call failed: {api} - {reason}",
            code="EXTERNAL_API_ERROR",
            details={"api": api, "status_code": status_code, "reason": reason},
        )


class CacheException(InfrastructureException):
    """キャッシュエラー"""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Cache operation failed: {operation} - {reason}",
            code="CACHE_ERROR",
            details={"operation": operation, "reason": reason},
        )


class MessageQueueException(InfrastructureException):
    """メッセージキューエラー"""
    
    def __init__(self, queue: str, operation: str, reason: str):
        super().__init__(
            f"Message queue operation failed: {queue}/{operation} - {reason}",
            code="MESSAGE_QUEUE_ERROR",
            details={"queue": queue, "operation": operation, "reason": reason},
        )


# プレゼンテーション層の例外
class PresentationException(BaseException):
    """プレゼンテーション層の例外基底クラス"""
    pass


class BadRequestException(PresentationException):
    """不正なリクエスト"""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Bad request: {reason}",
            code="BAD_REQUEST",
            details={"reason": reason},
        )


class NotFoundException(PresentationException):
    """リソースが見つからない"""
    
    def __init__(self, resource: str):
        super().__init__(
            f"Resource not found: {resource}",
            code="NOT_FOUND",
            details={"resource": resource},
        )


class ConflictException(PresentationException):
    """競合エラー"""
    
    def __init__(self, resource: str, reason: str):
        super().__init__(
            f"Conflict on {resource}: {reason}",
            code="CONFLICT",
            details={"resource": resource, "reason": reason},
        )


class RateLimitException(PresentationException):
    """レート制限"""
    
    def __init__(self, limit: int, window: str):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}",
            code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window},
        )