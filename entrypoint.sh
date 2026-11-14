#!/usr/bin/env bash
set -euo pipefail

: "${MCP_AUTH_TOKEN:?missing MCP_AUTH_TOKEN}"
: "${OAUTH2_PROXY_CLIENT_ID:?missing OAUTH2_PROXY_CLIENT_ID}"
: "${OAUTH2_PROXY_CLIENT_SECRET:?missing OAUTH2_PROXY_CLIENT_SECRET}"
: "${OAUTH2_PROXY_COOKIE_SECRET:?missing OAUTH2_PROXY_COOKIE_SECRET}"
: "${PUBLIC_BASE_URL:?missing PUBLIC_BASE_URL}"

CFG=/tmp/oauth2-proxy.cfg
cat > "$CFG" <<CFG
http_address = "0.0.0.0:3004"
provider = "google"
email_domains = ["*"]
authenticated_emails_file = "/app/allowed_emails.txt"
redirect_url = "${PUBLIC_BASE_URL}/oauth2/callback"
upstreams = ["http://127.0.0.1:3002"]
proxy_prefix = "/oauth2"
cookie_secure = true
cookie_httponly = true
cookie_samesite = "lax"
pass_authorization_header = true
set_authorization_header = true
pass_access_token = true
skip_provider_button = true
silence_ping_logging = true
CFG

echo "Starting supergateway..."
supergateway \
  --apiKey "$MCP_AUTH_TOKEN" \
  --stdio "npx -y mcp-server-filesystem /workspace" \
  --outputTransport streamableHttp \
  --port 3002 --host 0.0.0.0 &

sleep 2

echo "Starting oauth2-proxy..."
exec oauth2-proxy --config "$CFG"
