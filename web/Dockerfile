# ============================================================
# 前端镜像：Node 构建 → Nginx 托管（Linux 下 esbuild 可正常工作）
# 构建上下文为项目根目录：
#   docker build -f web/Dockerfile -t finance-qa-web .
# ============================================================
FROM node:20-slim AS build

WORKDIR /web

COPY web/package*.json ./
# 国内构建走 npmmirror；如需官方源可传 --build-arg NPM_REGISTRY=https://registry.npmjs.org
ARG NPM_REGISTRY=https://registry.npmmirror.com
RUN npm config set registry ${NPM_REGISTRY} \
    && npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html

EXPOSE 80
