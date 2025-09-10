"""AI学習統合インターフェース"""
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
from datetime import datetime

# trainディレクトリのパスを追加
train_dir = Path(__file__).parent.parent.parent / "train"
sys.path.append(str(train_dir))

try:
    from main_ppo import PPONikkeiTrainer
    from train import main as train_main
    HAS_TRAIN_MODULES = True
except ImportError:
    HAS_TRAIN_MODULES = False


class AITrainer:
    """AI学習システムの統合インターフェース"""
    
    def __init__(self, model_dir: str = "train/models"):
        """初期化
        
        Args:
            model_dir: モデル保存ディレクトリ
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def train_ppo_model(
        self,
        target_symbols: list[str],
        episodes: int = 1000,
        save_interval: int = 100,
    ) -> Optional[str]:
        """PPOモデルを学習
        
        Args:
            target_symbols: 対象銘柄リスト
            episodes: エピソード数
            save_interval: 保存間隔
        
        Returns:
            保存されたモデルパス（失敗時はNone）
        """
        if not HAS_TRAIN_MODULES:
            # モジュールが利用できない場合はサブプロセスで実行
            return self._train_with_subprocess(target_symbols, episodes)
        
        try:
            trainer = PPONikkeiTrainer()
            model_path = trainer.train(
                target_symbols=target_symbols,
                episodes=episodes,
                save_interval=save_interval
            )
            return str(model_path) if model_path else None
            
        except Exception as e:
            print(f"PPO学習エラー: {e}")
            return None
    
    def _train_with_subprocess(
        self,
        target_symbols: list[str],
        episodes: int,
    ) -> Optional[str]:
        """サブプロセスでPPO学習を実行"""
        try:
            # 現在のワーキングディレクトリを取得
            current_dir = Path.cwd()
            train_script = current_dir / "train" / "main_ppo.py"
            
            if not train_script.exists():
                print(f"学習スクリプトが見つかりません: {train_script}")
                return None
            
            # サブプロセスでPython実行
            cmd = [
                "uv", "run", "python", str(train_script),
                "--episodes", str(episodes),
                "--symbols", ",".join(target_symbols)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1時間タイムアウト
            )
            
            if result.returncode == 0:
                print("PPO学習が完了しました")
                # 最新のモデルファイルを探す
                return self._find_latest_model()
            else:
                print(f"学習エラー: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("学習がタイムアウトしました")
            return None
        except Exception as e:
            print(f"サブプロセス実行エラー: {e}")
            return None
    
    def _find_latest_model(self) -> Optional[str]:
        """最新のモデルファイルを検索"""
        model_patterns = [
            "train/models/rl/*.zip",
            "train/models/*.pth",
            "models/*.pth",
            "*.pth"
        ]
        
        latest_model = None
        latest_time = 0
        
        for pattern in model_patterns:
            for model_path in Path(".").glob(pattern):
                if model_path.is_file():
                    mtime = model_path.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_model = str(model_path)
        
        return latest_model
    
    def evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """モデルを評価
        
        Args:
            model_path: モデルファイルパス
        
        Returns:
            評価結果
        """
        # 簡易評価（実装は後で拡張）
        return {
            "model_path": model_path,
            "evaluation_date": datetime.now().isoformat(),
            "status": "evaluation_pending",
            "metrics": {}
        }
    
    def get_available_models(self) -> list[str]:
        """利用可能なモデル一覧を取得
        
        Returns:
            モデルファイルパスのリスト
        """
        models = []
        
        # 複数の場所からモデルを探す
        search_dirs = [
            Path("train/models/rl/"),
            Path("train/models/"),
            Path("models/"),
            Path(".")
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                models.extend(search_dir.glob("*.zip"))
                models.extend(search_dir.glob("*.pth"))
        
        return [str(model) for model in models if model.is_file()]
    
    def schedule_training(
        self,
        target_symbols: list[str],
        schedule_time: str = "daily",
    ) -> bool:
        """定期学習をスケジュール
        
        Args:
            target_symbols: 対象銘柄リスト  
            schedule_time: スケジュール（daily, weekly, monthly）
        
        Returns:
            スケジュール登録成功フラグ
        """
        # 実装は後で拡張（現在は仮実装）
        print(f"定期学習をスケジュール: {schedule_time}, 銘柄数: {len(target_symbols)}")
        return True