"""システム状態の永続化管理"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_csv_manager import BaseCSVManager


class SystemStateManager(BaseCSVManager):
    """システム状態の保存・管理を行うクラス
    
    システム状態とエラーログをCSVファイルで永続化し、監視機能を提供します。
    """
    
    STATE_CSV_FILENAME = 'system_state.csv'
    ERROR_CSV_FILENAME = 'system_errors.csv'
    
    def get_csv_headers(self) -> List[str]:
        """システム状態CSVヘッダーの定義
        
        Returns:
            システム状態CSVのヘッダー列名リスト
        """
        return [
            'state_id', 'system_status', 'last_update', 'active_plans',
            'portfolio_value', 'cash_balance', 'positions', 'risk_metrics',
            'error_count', 'last_error', 'performance_metrics',
            'maintenance_mode', 'pause_reason', 'uptime_hours',
            'memory_usage_mb', 'cpu_usage_percent', 'active_connections',
            'last_heartbeat'
        ]
    
    def get_error_csv_headers(self) -> List[str]:
        """エラーログCSVヘッダーの定義
        
        Returns:
            エラーログCSVのヘッダー列名リスト
        """
        return [
            'error_id', 'timestamp', 'error_type', 'error_message',
            'stack_trace', 'severity', 'component', 'resolved'
        ]
    
    def _convert_json_strings_to_dict_with_types(self, data: Dict[str, str]) -> Dict[str, Any]:
        """JSON文字列を元のデータ型に復元（数値型も含む）
        
        Args:
            data: JSON文字列を含む辞書
            
        Returns:
            復元されたデータ辞書
        """
        import json
        
        result = {}
        for key, value in data.items():
            if value == '':
                result[key] = None
            elif value.startswith(('{', '[')):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                # 数値の場合は適切な型に変換
                if key in ['portfolio_value', 'cash_balance', 'uptime_hours', 'memory_usage_mb', 'cpu_usage_percent']:
                    try:
                        result[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        result[key] = value
                elif key in ['error_count', 'active_connections']:
                    try:
                        result[key] = int(value)
                    except ValueError:
                        result[key] = value
                elif key == 'resolved':
                    result[key] = value.lower() == 'true'
                else:
                    result[key] = value
        return result
    
    def save_state(self, state_data: Dict[str, Any]) -> bool:
        """システム状態を保存
        
        Args:
            state_data: 保存するシステム状態データ
            
        Returns:
            保存成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.STATE_CSV_FILENAME)
        headers = self.get_csv_headers()
        
        # ヘッダーが存在しない場合は作成
        self._write_csv_header(csv_path, headers)
        
        # JSON化してCSVに追加
        json_data = self._convert_dict_to_json_strings(state_data)
        row_data = [json_data.get(header, '') for header in headers]
        
        return self._append_csv_row(csv_path, row_data)
    
    def load_latest_state(self) -> Optional[Dict[str, Any]]:
        """最新のシステム状態を読み込み
        
        Returns:
            最新のシステム状態データ、見つからない場合はNone
        """
        csv_path = self._get_csv_path(self.STATE_CSV_FILENAME)
        states = self._read_csv_rows(csv_path)
        
        if not states:
            return None
        
        # 最新の状態を取得（last_updateで判定）
        latest_state = None
        latest_update = None
        
        for state in states:
            converted_state = self._convert_json_strings_to_dict_with_types(state)
            update_time_str = converted_state.get('last_update', '')
            
            if update_time_str:
                try:
                    update_time = datetime.fromisoformat(
                        update_time_str.replace('Z', '+00:00')
                    )
                    if latest_update is None or update_time > latest_update:
                        latest_update = update_time
                        latest_state = converted_state
                except ValueError:
                    continue
        
        return latest_state
    
    def load_state_by_id(self, state_id: str) -> Optional[Dict[str, Any]]:
        """指定IDのシステム状態を読み込み
        
        Args:
            state_id: システム状態のID
            
        Returns:
            システム状態データ、見つからない場合はNone
        """
        csv_path = self._get_csv_path(self.STATE_CSV_FILENAME)
        states = self._read_csv_rows(csv_path)
        
        for state in states:
            if state.get('state_id') == state_id:
                return self._convert_json_strings_to_dict_with_types(state)
        
        return None
    
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """システム状態履歴を取得
        
        Args:
            limit: 取得件数の上限
            
        Returns:
            システム状態データのリスト（新しい順）
        """
        csv_path = self._get_csv_path(self.STATE_CSV_FILENAME)
        states = self._read_csv_rows(csv_path)
        
        # 時刻でソート（新しい順）
        sorted_states = []
        for state in states:
            converted_state = self._convert_json_strings_to_dict_with_types(state)
            update_time_str = converted_state.get('last_update', '')
            
            if update_time_str:
                try:
                    update_time = datetime.fromisoformat(
                        update_time_str.replace('Z', '+00:00')
                    )
                    sorted_states.append((update_time, converted_state))
                except ValueError:
                    continue
        
        # 新しい順でソートし、指定件数まで返す
        sorted_states.sort(key=lambda x: x[0], reverse=True)
        return [state for _, state in sorted_states[:limit]]
    
    def update_metrics(self, state_id: str, updates: Dict[str, Any]) -> bool:
        """システムメトリクスを更新
        
        Args:
            state_id: 更新対象のシステム状態ID
            updates: 更新データ
            
        Returns:
            更新成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.STATE_CSV_FILENAME)
        headers = self.get_csv_headers()
        existing_states = self._read_csv_rows(csv_path)
        
        for i, existing_state in enumerate(existing_states):
            if existing_state.get('state_id') == state_id:
                # 既存状態を更新
                updated_state = self._convert_json_strings_to_dict_with_types(existing_state)
                updated_state.update(updates)
                existing_states[i] = updated_state
                
                # ファイル全体を書き換え
                return self._rewrite_csv_file(csv_path, headers, existing_states)
        
        return False
    
    def record_error(self, error_data: Dict[str, Any]) -> bool:
        """システムエラーを記録
        
        Args:
            error_data: 記録するエラーデータ
            
        Returns:
            記録成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.ERROR_CSV_FILENAME)
        headers = self.get_error_csv_headers()
        
        # ヘッダーが存在しない場合は作成
        self._write_csv_header(csv_path, headers)
        
        # JSON化してCSVに追加
        json_data = self._convert_dict_to_json_strings(error_data)
        row_data = [json_data.get(header, '') for header in headers]
        
        return self._append_csv_row(csv_path, row_data)
    
    def get_error_history(self, resolved_only: Optional[bool] = None) -> List[Dict[str, Any]]:
        """エラー履歴を取得
        
        Args:
            resolved_only: Trueなら解決済みのみ、Falseなら未解決のみ、Noneなら全て
            
        Returns:
            エラーデータのリスト
        """
        csv_path = self._get_csv_path(self.ERROR_CSV_FILENAME)
        errors = self._read_csv_rows(csv_path)
        
        result = []
        for error in errors:
            converted_error = self._convert_json_strings_to_dict_with_types(error)
            
            if resolved_only is None:
                result.append(converted_error)
            elif resolved_only and converted_error.get('resolved') == 'True':
                result.append(converted_error)
            elif not resolved_only and converted_error.get('resolved') != 'True':
                result.append(converted_error)
        
        return result
    
    def check_system_health(self) -> Optional[Dict[str, Any]]:
        """システムヘルスチェックを実行
        
        Returns:
            ヘルスチェック結果、状態が見つからない場合はNone
        """
        latest_state = self.load_latest_state()
        
        if not latest_state:
            return None
        
        # 最後の更新からの経過時間を計算
        last_update_str = latest_state.get('last_update', '')
        last_update_age_minutes = 0
        
        if last_update_str:
            try:
                last_update = datetime.fromisoformat(
                    last_update_str.replace('Z', '+00:00')
                )
                age = datetime.now() - last_update.replace(tzinfo=None)
                last_update_age_minutes = age.total_seconds() / 60
            except ValueError:
                pass
        
        # ヘルスステータスを判定
        overall_status = 'healthy'
        if last_update_age_minutes > 60:  # 1時間以上更新がない
            overall_status = 'stale'
        elif latest_state.get('system_status') != 'active':
            overall_status = 'inactive'
        
        # 未解決エラーをチェック
        unresolved_errors = self.get_error_history(resolved_only=False)
        high_severity_errors = [
            e for e in unresolved_errors 
            if e.get('severity') == 'high'
        ]
        
        if high_severity_errors:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'uptime_hours': latest_state.get('uptime_hours', 0),
            'resource_usage': {
                'memory_mb': latest_state.get('memory_usage_mb', 0),
                'cpu_percent': latest_state.get('cpu_usage_percent', 0)
            },
            'last_update_age_minutes': last_update_age_minutes,
            'unresolved_errors': len(unresolved_errors),
            'high_severity_errors': len(high_severity_errors)
        }