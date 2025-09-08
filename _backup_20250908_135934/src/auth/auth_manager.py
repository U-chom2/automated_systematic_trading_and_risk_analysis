"""
API認証管理モジュール

各種APIの認証情報を安全に管理し、アクセストークンの取得・更新を行う。
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
import base64


logger = logging.getLogger(__name__)


@dataclass
class APICredentials:
    """API認証情報を格納するデータクラス"""
    
    api_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    additional_params: Optional[Dict[str, Any]] = None
    
    def is_token_expired(self) -> bool:
        """トークンの有効期限をチェック"""
        if not self.token_expires_at:
            return False
        return datetime.now() >= self.token_expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（シリアライゼーション用）"""
        data = asdict(self)
        # datetimeをISO文字列に変換
        if self.token_expires_at:
            data['token_expires_at'] = self.token_expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APICredentials':
        """辞書から復元（デシリアライゼーション用）"""
        if 'token_expires_at' in data and data['token_expires_at']:
            data['token_expires_at'] = datetime.fromisoformat(data['token_expires_at'])
        return cls(**data)


class AuthManager:
    """
    API認証管理クラス
    
    各種APIの認証情報を暗号化して保存し、アクセス管理を行う。
    """
    
    def __init__(self, config_path: Optional[str] = None, encryption_key: Optional[str] = None):
        """
        AuthManagerを初期化
        
        Args:
            config_path: 設定ファイルパス（デフォルト: ~/.trading_system/auth.enc）
            encryption_key: 暗号化キー（環境変数TRADING_SYSTEM_KEYから取得）
        """
        self.config_path = Path(config_path or Path.home() / ".trading_system" / "auth.enc")
        self.credentials: Dict[str, APICredentials] = {}
        
        # 暗号化キーの設定
        if encryption_key:
            self.encryption_key = encryption_key.encode()
        else:
            env_key = os.getenv('TRADING_SYSTEM_KEY')
            if env_key:
                self.encryption_key = env_key.encode()
            else:
                # キーが無い場合は新規生成
                self.encryption_key = Fernet.generate_key()
                logger.warning("新しい暗号化キーを生成しました。TRADING_SYSTEM_KEY環境変数に設定してください。")
                logger.warning(f"暗号化キー: {self.encryption_key.decode()}")
        
        self.fernet = Fernet(self.encryption_key)
        
        # 設定ディレクトリを作成
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存の認証情報を読み込み
        self.load_credentials()
    
    def add_credentials(self, credentials: APICredentials) -> None:
        """
        認証情報を追加
        
        Args:
            credentials: API認証情報
        """
        self.credentials[credentials.api_name] = credentials
        self.save_credentials()
        logger.info(f"API認証情報を追加しました: {credentials.api_name}")
    
    def get_credentials(self, api_name: str) -> Optional[APICredentials]:
        """
        認証情報を取得
        
        Args:
            api_name: API名
            
        Returns:
            API認証情報（存在しない場合はNone）
        """
        return self.credentials.get(api_name)
    
    def update_token(self, api_name: str, access_token: str, 
                    refresh_token: Optional[str] = None,
                    expires_in_seconds: Optional[int] = None) -> bool:
        """
        トークン情報を更新
        
        Args:
            api_name: API名
            access_token: アクセストークン
            refresh_token: リフレッシュトークン
            expires_in_seconds: トークン有効期限（秒）
            
        Returns:
            更新成功の場合True
        """
        if api_name not in self.credentials:
            logger.error(f"API認証情報が見つかりません: {api_name}")
            return False
        
        cred = self.credentials[api_name]
        cred.access_token = access_token
        
        if refresh_token:
            cred.refresh_token = refresh_token
        
        if expires_in_seconds:
            cred.token_expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        self.save_credentials()
        logger.info(f"トークン情報を更新しました: {api_name}")
        return True
    
    def is_token_valid(self, api_name: str) -> bool:
        """
        トークンが有効かチェック
        
        Args:
            api_name: API名
            
        Returns:
            トークンが有効な場合True
        """
        cred = self.credentials.get(api_name)
        if not cred or not cred.access_token:
            return False
        
        return not cred.is_token_expired()
    
    def remove_credentials(self, api_name: str) -> bool:
        """
        認証情報を削除
        
        Args:
            api_name: API名
            
        Returns:
            削除成功の場合True
        """
        if api_name in self.credentials:
            del self.credentials[api_name]
            self.save_credentials()
            logger.info(f"API認証情報を削除しました: {api_name}")
            return True
        return False
    
    def list_api_names(self) -> List[str]:
        """
        登録されているAPI名のリストを取得
        
        Returns:
            API名のリスト
        """
        return list(self.credentials.keys())
    
    def save_credentials(self) -> None:
        """認証情報をファイルに暗号化して保存"""
        try:
            # 認証情報を辞書形式に変換
            data = {
                name: cred.to_dict() 
                for name, cred in self.credentials.items()
            }
            
            # JSON文字列に変換
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            
            # 暗号化
            encrypted_data = self.fernet.encrypt(json_str.encode('utf-8'))
            
            # ファイルに保存
            self.config_path.write_bytes(encrypted_data)
            logger.debug(f"認証情報を保存しました: {self.config_path}")
            
        except Exception as e:
            logger.error(f"認証情報の保存に失敗しました: {e}")
            raise
    
    def load_credentials(self) -> None:
        """ファイルから認証情報を復号化して読み込み"""
        try:
            if not self.config_path.exists():
                logger.info("認証設定ファイルが存在しません。新規作成します。")
                return
            
            # ファイルから暗号化データを読み込み
            encrypted_data = self.config_path.read_bytes()
            
            # 復号化
            decrypted_data = self.fernet.decrypt(encrypted_data)
            json_str = decrypted_data.decode('utf-8')
            
            # JSON解析
            data = json.loads(json_str)
            
            # APICredentialsオブジェクトに復元
            self.credentials = {
                name: APICredentials.from_dict(cred_data)
                for name, cred_data in data.items()
            }
            
            logger.info(f"認証情報を読み込みました: {len(self.credentials)} 件")
            
        except FileNotFoundError:
            logger.info("認証設定ファイルが見つかりません")
        except Exception as e:
            logger.error(f"認証情報の読み込みに失敗しました: {e}")
            # 読み込みエラーの場合は空の認証情報で継続
            self.credentials = {}
    
    def setup_x_api(self, api_key: str, api_secret: str, 
                   access_token: str, access_token_secret: str) -> None:
        """
        X (Twitter) API認証を設定
        
        Args:
            api_key: APIキー
            api_secret: APIシークレット
            access_token: アクセストークン
            access_token_secret: アクセストークンシークレット
        """
        credentials = APICredentials(
            api_name="x_api",
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            additional_params={"access_token_secret": access_token_secret}
        )
        self.add_credentials(credentials)
    
    def setup_broker_api(self, api_name: str, api_key: str, api_secret: str,
                        additional_params: Optional[Dict[str, Any]] = None) -> None:
        """
        証券会社API認証を設定
        
        Args:
            api_name: 証券会社API名（例: "sbi_api", "rakuten_api"）
            api_key: APIキー
            api_secret: APIシークレット
            additional_params: 追加パラメータ（口座番号など）
        """
        credentials = APICredentials(
            api_name=api_name,
            api_key=api_key,
            api_secret=api_secret,
            additional_params=additional_params or {}
        )
        self.add_credentials(credentials)
    
    def get_x_auth(self) -> Optional[Dict[str, str]]:
        """
        X API認証情報を取得
        
        Returns:
            X API用の認証辞書
        """
        cred = self.get_credentials("x_api")
        if not cred:
            return None
        
        return {
            "api_key": cred.api_key,
            "api_secret": cred.api_secret,
            "access_token": cred.access_token,
            "access_token_secret": cred.additional_params.get("access_token_secret")
        }
    
    def get_broker_auth(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """
        証券会社API認証情報を取得
        
        Args:
            broker_name: 証券会社名
            
        Returns:
            証券会社API用の認証辞書
        """
        cred = self.get_credentials(broker_name)
        if not cred:
            return None
        
        auth_data = {
            "api_key": cred.api_key,
            "api_secret": cred.api_secret,
            "access_token": cred.access_token,
        }
        
        # 追加パラメータをマージ
        if cred.additional_params:
            auth_data.update(cred.additional_params)
        
        return auth_data
    
    def validate_all_credentials(self) -> Dict[str, bool]:
        """
        すべての認証情報の有効性をチェック
        
        Returns:
            API名をキーとした有効性の辞書
        """
        results = {}
        for api_name in self.credentials:
            results[api_name] = self.is_token_valid(api_name)
        return results
    
    def get_auth_status(self) -> Dict[str, Any]:
        """
        認証状況のサマリーを取得
        
        Returns:
            認証状況の辞書
        """
        status = {
            "total_apis": len(self.credentials),
            "valid_tokens": 0,
            "expired_tokens": 0,
            "missing_tokens": 0,
            "api_details": {}
        }
        
        for api_name, cred in self.credentials.items():
            if not cred.access_token:
                status["missing_tokens"] += 1
                token_status = "missing"
            elif cred.is_token_expired():
                status["expired_tokens"] += 1
                token_status = "expired"
            else:
                status["valid_tokens"] += 1
                token_status = "valid"
            
            status["api_details"][api_name] = {
                "token_status": token_status,
                "expires_at": cred.token_expires_at.isoformat() if cred.token_expires_at else None
            }
        
        return status