# 🚀 Axion Deployment Guide

This guide covers multiple ways to deploy Axion on your own infrastructure.

---

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Option 1: Docker (Recommended)](#option-1-docker-recommended)
3. [Option 2: Railway (Easiest)](#option-2-railway-easiest)
4. [Option 3: Render](#option-3-render)
5. [Option 4: VPS (DigitalOcean/AWS/Linode)](#option-4-vps)
6. [Option 5: Vercel + MongoDB Atlas](#option-5-vercel--mongodb-atlas)
7. [Environment Variables](#environment-variables)
8. [Troubleshooting](#troubleshooting)

---

## Quick Overview

| Platform | Difficulty | Cost | Best For |
|----------|------------|------|----------|
| **Docker** | Medium | Free (self-hosted) | Full control |
| **Railway** | Easy | Free tier available | Quick demos |
| **Render** | Easy | Free tier available | Production |
| **VPS** | Hard | $5-20/month | Full control |
| **Vercel** | Medium | Free tier | Frontend only |

---

## Option 1: Docker (Recommended)

### Prerequisites
- Docker & Docker Compose installed
- Git installed

### Step 1: Create Docker Files

Create `Dockerfile` in the root directory:

```dockerfile
# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

Create `frontend/Dockerfile`:

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:6.0
    container_name: axion-mongo
    restart: always
    volumes:
      - mongodb_data:/data/db
    ports:
      - "27017:27017"

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: axion-backend
    restart: always
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - DB_NAME=axion
    depends_on:
      - mongodb

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: axion-frontend
    restart: always
    ports:
      - "80:80"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8001
    depends_on:
      - backend

volumes:
  mongodb_data:
```

### Step 2: Build and Run

```bash
# Clone your repo
git clone https://github.com/Soumadeep21/axion.git
cd axion

# Build and start all services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 3: Access

- Frontend: `http://localhost`
- Backend: `http://localhost:8001`

---

## Option 2: Railway (Easiest)

[Railway](https://railway.app) is the easiest way to deploy full-stack apps.

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

### Step 2: Deploy from GitHub

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `axion` repository

### Step 3: Add Services

You need 3 services:

#### MongoDB
1. Click **"New"** → **"Database"** → **"MongoDB"**
2. Railway will create a MongoDB instance
3. Copy the `MONGO_URL` from the **Variables** tab

#### Backend
1. Click **"New"** → **"GitHub Repo"**
2. Select your repo
3. Set **Root Directory**: `backend`
4. Add environment variables:
   ```
   MONGO_URL=<paste from MongoDB service>
   DB_NAME=axion
   PORT=8001
   ```
5. Railway auto-detects Python and deploys

#### Frontend
1. Click **"New"** → **"GitHub Repo"**
2. Select your repo
3. Set **Root Directory**: `frontend`
4. Add environment variable:
   ```
   REACT_APP_BACKEND_URL=https://<your-backend-url>.railway.app
   ```

### Step 4: Get Your URLs

Railway provides URLs like:
- Backend: `https://axion-backend-xxxx.railway.app`
- Frontend: `https://axion-frontend-xxxx.railway.app`

---

## Option 3: Render

[Render](https://render.com) offers generous free tier.

### Step 1: Create Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create MongoDB (or use MongoDB Atlas)

For MongoDB Atlas (free):
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create free cluster
3. Get connection string

### Step 3: Deploy Backend

1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `axion-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   ```
   MONGO_URL=mongodb+srv://...
   DB_NAME=axion
   ```
5. Click **"Create Web Service"**

### Step 4: Deploy Frontend

1. Click **"New"** → **"Static Site"**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `axion-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `build`
4. Add Environment Variable:
   ```
   REACT_APP_BACKEND_URL=https://axion-backend.onrender.com
   ```
5. Click **"Create Static Site"**

---

## Option 4: VPS (DigitalOcean/AWS/Linode)

For full control, deploy on a VPS.

### Step 1: Create VPS

**DigitalOcean:**
1. Create account at [digitalocean.com](https://digitalocean.com)
2. Create Droplet: Ubuntu 22.04, $6/month (1GB RAM)
3. Add SSH key

**AWS EC2:**
1. Create t2.micro instance (free tier)
2. Ubuntu 22.04 AMI
3. Configure security group (ports 22, 80, 443, 8001)

### Step 2: Connect to Server

```bash
ssh root@your-server-ip
```

### Step 3: Install Dependencies

```bash
# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3 python3-pip python3-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs

# Install MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list
apt update
apt install -y mongodb-org
systemctl start mongod
systemctl enable mongod

# Install Nginx
apt install -y nginx

# Install PM2 (process manager)
npm install -g pm2
```

### Step 4: Clone and Setup

```bash
# Clone repo
cd /var/www
git clone https://github.com/Soumadeep21/axion.git
cd axion

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
cat > .env << EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=axion
EOF

# Start backend with PM2
pm2 start "uvicorn server:app --host 0.0.0.0 --port 8001" --name axion-backend

# Setup frontend
cd ../frontend
npm install

# Create .env
echo "REACT_APP_BACKEND_URL=http://your-server-ip:8001" > .env

# Build frontend
npm run build
```

### Step 5: Configure Nginx

```bash
cat > /etc/nginx/sites-available/axion << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # or your-server-ip

    # Frontend
    location / {
        root /var/www/axion/frontend/build;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/axion /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test and restart
nginx -t
systemctl restart nginx
```

### Step 6: Setup SSL (Optional but Recommended)

```bash
# Install Certbot
apt install -y certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d your-domain.com

# Auto-renewal
certbot renew --dry-run
```

### Step 7: Save PM2 Configuration

```bash
pm2 save
pm2 startup
```

---

## Option 5: Vercel + MongoDB Atlas

Good for frontend-heavy deployments.

### Step 1: MongoDB Atlas Setup

1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create free M0 cluster
3. Create database user
4. Whitelist IP: `0.0.0.0/0` (allow all)
5. Get connection string

### Step 2: Deploy Backend to Render/Railway

(Use instructions from Option 2 or 3 for backend)

### Step 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Configure:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
4. Add Environment Variable:
   ```
   REACT_APP_BACKEND_URL=https://your-backend-url.com
   ```
5. Deploy!

---

## Environment Variables

### Backend (.env)

```env
# Required
MONGO_URL=mongodb://localhost:27017
DB_NAME=axion

# Optional - for AI Advisor
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional - for exchange API
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
```

### Frontend (.env)

```env
# Required
REACT_APP_BACKEND_URL=https://your-backend-url.com
```

---

## Troubleshooting

### Backend won't start

```bash
# Check logs
pm2 logs axion-backend

# Check if port is in use
lsof -i :8001

# Check MongoDB
systemctl status mongod
```

### Frontend build fails

```bash
# Clear cache and rebuild
cd frontend
rm -rf node_modules build
npm install
npm run build
```

### MongoDB connection issues

```bash
# Test connection
mongosh "mongodb://localhost:27017"

# Check if running
systemctl status mongod

# Check logs
tail -f /var/log/mongodb/mongod.log
```

### CORS errors

Add to backend `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-url.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Quick Deployment Checklist

- [ ] Clone repository
- [ ] Set up MongoDB (local or Atlas)
- [ ] Configure backend environment variables
- [ ] Deploy backend
- [ ] Note backend URL
- [ ] Configure frontend environment variable with backend URL
- [ ] Deploy frontend
- [ ] Test all features
- [ ] Set up SSL (for production)

---

## Need Help?

- **Issues:** [GitHub Issues](https://github.com/Soumadeep21/axion/issues)
- **Author:** [Soumadeep21](https://github.com/Soumadeep21)

---

**Happy Deploying! 🚀**
