#!/bin/bash
# faceswapjp setup for macOS (Apple Silicon). Double-click this file in Finder.
# Installs ffmpeg, a private Python environment and the AI models, then creates
# ~/Applications/faceswapjp.app so you can start it from Launchpad / Dock.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

say_step() { printf "\n\033[1;36m▶ %s\033[0m\n" "$1"; }
fail() { printf "\n\033[1;31m✖ %s\033[0m\n" "$1"; echo "このウィンドウの内容をコピーして送ってもらえれば原因を調べます。"; read -r -p "Enter で閉じます"; exit 1; }
trap 'fail "途中でエラーが起きました（$BASH_COMMAND）"' ERR

echo "faceswapjp のセットアップを始めます。10〜20 分ほどかかります（AI モデル約 1.2 GB のダウンロードを含みます）。"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

say_step "1/4 動画処理ソフト（ffmpeg）"
if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then
  echo "インストール済みです。"
else
  if ! command -v brew >/dev/null; then
    echo "Homebrew（Mac 用のソフト管理ツール）をインストールします。"
    echo "Mac のログインパスワードを聞かれたら入力してください（入力中は何も表示されません）。"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
  fi
  brew install ffmpeg
fi

say_step "2/4 Python 環境"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv -q -p 3.11 --allow-existing .venv
uv pip install -q --python .venv/bin/python -e ".[mac,ui]"

say_step "3/4 AI モデル（約 1.2 GB、初回のみ）"
.venv/bin/faceswapjp models download

say_step "4/4 アプリの作成"
APP="$HOME/Applications/faceswapjp.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/MacOS/faceswapjp" <<LAUNCH
#!/bin/bash
# Starts the local studio (127.0.0.1 only) and opens it in the browser.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
PORT=7860
if curl -s -o /dev/null "http://127.0.0.1:\$PORT/"; then open "http://127.0.0.1:\$PORT/"; exit 0; fi
mkdir -p "\$HOME/Movies/faceswapjp" "\$HOME/Library/Logs"
cd "$REPO"
nohup .venv/bin/faceswapjp ui --projects "\$HOME/Movies/faceswapjp" --port \$PORT >> "\$HOME/Library/Logs/faceswapjp.log" 2>&1 &
LAUNCH
chmod +x "$APP/Contents/MacOS/faceswapjp"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>faceswapjp</string>
  <key>CFBundleDisplayName</key><string>faceswapjp</string>
  <key>CFBundleIdentifier</key><string>local.faceswapjp.studio</string>
  <key>CFBundleExecutable</key><string>faceswapjp</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>LSUIElement</key><true/>
</dict></plist>
PLIST
ICONSET="$(mktemp -d)/icon.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s macos/icon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) macos/icon.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns" || true
touch "$APP"

trap - ERR
printf "\n\033[1;32m✔ セットアップが終わりました\033[0m\n"
echo "「アプリケーション」フォルダ（ホーム内）の faceswapjp をダブルクリックすると起動します。"
echo "Dock に入れておくと便利です。作品と書き出したファイルは「ムービー/faceswapjp」に保存されます。"
open "$HOME/Applications"
open "$APP"
read -r -p "Enter でこのウィンドウを閉じられます"
