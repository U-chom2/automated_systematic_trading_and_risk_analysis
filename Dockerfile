FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uvをインストール
RUN pip install uv

# 依存関係ファイルをコピー
COPY pyproject.toml .
COPY uv.lock* .

# 依存関係をインストール
RUN uv sync --frozen

# アプリケーションコードをコピー
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

# 非rootユーザーを作成
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# ポートを公開
EXPOSE 8000

# アプリケーションを起動
CMD ["uv", "run", "uvicorn", "src.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]