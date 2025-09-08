"""実行計画の永続化管理"""

from typing import Dict, Any, List, Optional
from .base_csv_manager import BaseCSVManager


class ExecutionPlanManager(BaseCSVManager):
    """実行計画の保存・復元を管理するクラス
    
    実行計画をCSVファイルで永続化し、CRUD操作を提供します。
    """
    
    CSV_FILENAME = 'execution_plans.csv'
    
    def get_csv_headers(self) -> List[str]:
        """CSV ヘッダーの定義
        
        Returns:
            実行計画CSVのヘッダー列名リスト
        """
        return [
            'id', 'name', 'strategy', 'parameters', 
            'created_at', 'updated_at', 'status'
        ]
    
    def save_plan(self, plan_data: Dict[str, Any]) -> bool:
        """実行計画を保存
        
        Args:
            plan_data: 保存する実行計画データ
            
        Returns:
            保存成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        headers = self.get_csv_headers()
        
        # ヘッダーが存在しない場合は作成
        self._write_csv_header(csv_path, headers)
        
        # 既存計画があれば更新、なければ新規追加
        existing_plans = self._read_csv_rows(csv_path)
        plan_updated = False
        
        for i, existing_plan in enumerate(existing_plans):
            if existing_plan.get('id') == plan_data.get('id'):
                # 既存計画を更新
                updated_plan = self._convert_json_strings_to_dict(existing_plan)
                updated_plan.update(plan_data)
                existing_plans[i] = updated_plan
                plan_updated = True
                break
        
        if plan_updated:
            # ファイル全体を書き換え
            return self._rewrite_csv_file(csv_path, headers, existing_plans)
        else:
            # 新しい計画として追加
            json_data = self._convert_dict_to_json_strings(plan_data)
            row_data = [json_data.get(header, '') for header in headers]
            return self._append_csv_row(csv_path, row_data)
    
    def load_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """指定IDの実行計画を読み込み
        
        Args:
            plan_id: 実行計画のID
            
        Returns:
            実行計画データ、見つからない場合はNone
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        plans = self._read_csv_rows(csv_path)
        
        for plan in plans:
            if plan.get('id') == plan_id:
                return self._convert_json_strings_to_dict(plan)
        
        return None
    
    def list_plans(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """実行計画一覧を取得
        
        Args:
            status: フィルターするステータス（省略時は全て）
            
        Returns:
            実行計画データのリスト
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        plans = self._read_csv_rows(csv_path)
        
        result = []
        for plan in plans:
            converted_plan = self._convert_json_strings_to_dict(plan)
            if status is None or converted_plan.get('status') == status:
                result.append(converted_plan)
        
        return result
    
    def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> bool:
        """実行計画を更新
        
        Args:
            plan_id: 更新対象の実行計画ID
            updates: 更新データ
            
        Returns:
            更新成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        headers = self.get_csv_headers()
        existing_plans = self._read_csv_rows(csv_path)
        
        for i, existing_plan in enumerate(existing_plans):
            if existing_plan.get('id') == plan_id:
                # 既存計画を更新
                updated_plan = self._convert_json_strings_to_dict(existing_plan)
                updated_plan.update(updates)
                existing_plans[i] = updated_plan
                
                # ファイル全体を書き換え
                return self._rewrite_csv_file(csv_path, headers, existing_plans)
        
        return False
    
    def delete_plan(self, plan_id: str) -> bool:
        """実行計画を削除
        
        Args:
            plan_id: 削除対象の実行計画ID
            
        Returns:
            削除成功時True、失敗時False
        """
        csv_path = self._get_csv_path(self.CSV_FILENAME)
        headers = self.get_csv_headers()
        existing_plans = self._read_csv_rows(csv_path)
        
        # 削除対象以外の計画を抽出
        filtered_plans = []
        found_target = False
        
        for existing_plan in existing_plans:
            if existing_plan.get('id') != plan_id:
                filtered_plans.append(self._convert_json_strings_to_dict(existing_plan))
            else:
                found_target = True
        
        if found_target:
            # ファイル全体を書き換え
            return self._rewrite_csv_file(csv_path, headers, filtered_plans)
        
        return False