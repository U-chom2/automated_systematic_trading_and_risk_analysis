"""CSVベースの永続化マネージャーの基底クラス"""

import csv
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class BaseCSVManager(ABC):
    """CSV永続化の基底クラス
    
    共通のCSVファイル操作機能を提供します。
    """
    
    def __init__(self, data_dir: str) -> None:
        """初期化
        
        Args:
            data_dir: データファイルを保存するディレクトリパス
        """
        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_csv_path(self, filename: str) -> Path:
        """CSVファイルのパスを取得
        
        Args:
            filename: CSVファイル名
            
        Returns:
            CSVファイルの完全パス
        """
        return self.data_dir / filename
        
    def _write_csv_header(self, csv_path: Path, headers: List[str]) -> None:
        """CSVヘッダーを書き込み
        
        Args:
            csv_path: CSVファイルパス
            headers: ヘッダー行の列名リスト
        """
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
    def _append_csv_row(self, csv_path: Path, row_data: List[Any]) -> bool:
        """CSVファイルに行を追加
        
        Args:
            csv_path: CSVファイルパス
            row_data: 追加するデータの行
            
        Returns:
            書き込み成功時True、失敗時False
        """
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            return True
        except (IOError, OSError):
            return False
            
    def _read_csv_rows(self, csv_path: Path) -> List[Dict[str, str]]:
        """CSVファイルから全行を読み込み
        
        Args:
            csv_path: CSVファイルパス
            
        Returns:
            辞書形式の行データのリスト
        """
        if not csv_path.exists():
            return []
            
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except (IOError, OSError):
            return []
            
    def _convert_dict_to_json_strings(self, data: Dict[str, Any]) -> Dict[str, str]:
        """辞書内の複雑なデータをJSON文字列に変換
        
        Args:
            data: 変換対象のデータ辞書
            
        Returns:
            JSON文字列化された辞書
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = str(value) if value is not None else ''
        return result
        
    def _convert_json_strings_to_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """JSON文字列を元のデータ型に復元
        
        Args:
            data: JSON文字列を含む辞書
            
        Returns:
            復元されたデータ辞書
        """
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
                result[key] = value
        return result
        
    def _rewrite_csv_file(self, csv_path: Path, headers: List[str], 
                         rows: List[Dict[str, Any]]) -> bool:
        """CSVファイルを完全に書き換え
        
        Args:
            csv_path: CSVファイルパス
            headers: ヘッダー行
            rows: データ行のリスト
            
        Returns:
            書き込み成功時True、失敗時False
        """
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for row in rows:
                    json_row = self._convert_dict_to_json_strings(row)
                    writer.writerow([json_row.get(header, '') for header in headers])
            return True
        except (IOError, OSError):
            return False
            
    @abstractmethod
    def get_csv_headers(self) -> List[str]:
        """各マネージャーで実装するCSVヘッダーの定義
        
        Returns:
            CSVヘッダーの列名リスト
        """
        pass