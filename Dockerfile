FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
COPY packages/engine/package.json packages/engine/
COPY packages/ai/package.json packages/ai/
COPY packages/web/package.json packages/web/
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/packages/web/dist /usr/share/nginx/html
EXPOSE 80
