# ---- Frontend Dockerfile ----
# Multi-stage build: build with Node, serve with nginx

# Stage 1: Build
FROM node:22-alpine AS build

WORKDIR /app

# Copy dependency files first for layer caching
COPY package.json package-lock.json ./

RUN npm ci

COPY . .

# Build-time arg for backend URL (baked into the JS bundle)
ARG VITE_BACKEND_URL=http://localhost:8000
ARG VITE_WS_URL=ws://localhost:8000

ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
ENV VITE_WS_URL=${VITE_WS_URL}

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Custom nginx config for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
