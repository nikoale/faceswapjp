#!/bin/bash
# Start faceswapjp without the app bundle (same as double-clicking ~/Applications/faceswapjp.app).
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ ! -x .venv/bin/faceswapjp ]; then echo "先に「1_セットアップ.command」を実行してください。"; read -r -p "Enter で閉じます"; exit 1; fi
mkdir -p "$HOME/Movies/faceswapjp"
echo "faceswapjp を起動しています… ブラウザが開きます。終了するときは画面右上の「終了」を押してください。"
exec .venv/bin/faceswapjp ui --projects "$HOME/Movies/faceswapjp"
