#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$ROOT/.tmp-render"
ASSETS_DIR="$ROOT/assets"
CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

mkdir -p "$TMP_DIR"

cat > "$TMP_DIR/puppeteer-config.json" <<EOF
{
  "executablePath": "$CHROME_BIN",
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
EOF

render() {
  local name="$1"
  npx -y @mermaid-js/mermaid-cli \
    -p "$TMP_DIR/puppeteer-config.json" \
    -i "$ASSETS_DIR/${name}.mmd" \
    -o "$ASSETS_DIR/${name}.png" \
    -w 1800 \
    -H 1400
}

render "learning-roadmap"
render "paper-reading-map"
render "local-demo-workflow"

echo "Rendered Mermaid diagrams into: $ASSETS_DIR"
