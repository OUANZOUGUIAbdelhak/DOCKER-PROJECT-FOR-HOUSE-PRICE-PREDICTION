# React Frontend Dockerfile
FROM node:18-alpine as build

WORKDIR /app

# Install dependencies for faster builds
RUN apk add --no-cache python3 make g++

# Copy package files first (for better caching)
COPY frontend/package.json ./

# Install dependencies with verbose output and retry
RUN npm install --verbose --no-audit --no-fund || \
    (echo "First install failed, retrying..." && npm install --verbose --no-audit --no-fund)

# Copy source code
COPY frontend/ .

# Build app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app to nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
