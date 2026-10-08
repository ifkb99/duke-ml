FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
COPY packages/engine/package.json packages/engine/
COPY packages/ai/package.json packages/ai/
COPY packages/web/package.json packages/web/
COPY packages/harness/package.json packages/harness/
RUN npm ci
COPY . .
RUN npm run build
# Precompressed copies for nginx's gzip_static (the WebAssembly runtime is 28 MB;
# the network's packed weights shrink ~15%).
RUN find packages/web/dist -type f \( -name '*.js' -o -name '*.mjs' -o -name '*.css' -o -name '*.html' \
      -o -name '*.wasm' -o -name '*.svg' -o -name '*.json' -o -name '*.onnx' -o -name '*.bin' \) -exec gzip -9 -k {} +

FROM nginx:alpine
COPY packages/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/packages/web/dist /usr/share/nginx/html
EXPOSE 80
