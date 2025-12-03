# ベースイメージ: Python 3.10 (Slim版で軽量化)
FROM python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /app

# システム依存パッケージのインストール
# gcc, g++: PyTorch Geometricなどのビルドに必要
# git: ライブラリの取得に必要
# curl: ヘルスチェック等に便利
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 依存関係ファイルのコピー
COPY pyproject.toml README.md ./

# ライブラリのインストール
# 1. pipのアップグレード
RUN pip install --no-cache-dir --upgrade pip

# 2. PyTorch (CPU版) のインストール
# Dockerイメージサイズ削減のため、まずはCPU版を指定して入れる
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. GNN関連ライブラリのバイナリインストール
# ビルド時間を短縮するためにホイールを直接指定
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# 4. プロジェクトの依存関係を一括インストール
# pyproject.toml に書かれたライブラリが入ります
RUN pip install --no-cache-dir -e .

# ソースコードのコピー
COPY src ./src
# テスト用スクリプトなどもコピー（デモ用）
COPY run_*.py ./

# ポートの公開 (Dashアプリ用)
EXPOSE 8050

# コンテナ起動時のデフォルトコマンド
CMD ["python", "src/resilientflow/viz/app.py"]