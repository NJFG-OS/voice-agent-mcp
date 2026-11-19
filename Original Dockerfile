FROM node:20-bookworm-slim

ARG OAUTH2_PROXY_VERSION=7.6.0
ARG OAUTH2_PROXY_ARCH=linux-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tar bash tini \
 && rm -rf /var/lib/apt/lists/*

# oauth2-proxy
RUN set -eux; \
    TAR="oauth2-proxy-v${OAUTH2_PROXY_VERSION}.${OAUTH2_PROXY_ARCH}.tar.gz"; \
    URL="https://github.com/oauth2-proxy/oauth2-proxy/releases/download/v${OAUTH2_PROXY_VERSION}/${TAR}"; \
    curl -fsSL -o /tmp/${TAR} "${URL}"; \
    tar -xzf /tmp/${TAR} -C /tmp; \
    mv "/tmp/oauth2-proxy-v${OAUTH2_PROXY_VERSION}.${OAUTH2_PROXY_ARCH}/oauth2-proxy" /usr/local/bin/oauth2-proxy; \
    chmod +x /usr/local/bin/oauth2-proxy; \
    oauth2-proxy --version

# MCP bridge + filesystem server
RUN npm install -g supergateway @modelcontextprotocol/server-filesystem

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
COPY allowed_emails.txt /app/allowed_emails.txt
RUN chmod +x /app/entrypoint.sh

EXPOSE 3002 3004

ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["/app/entrypoint.sh"]
