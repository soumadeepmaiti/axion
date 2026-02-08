# Vercel Deployment Guide for Axion

## ⚠️ Important: Vercel = Frontend Only

Vercel only supports **static/frontend** deployments. You need to deploy:
- **Frontend** → Vercel ✅
- **Backend** → Render/Railway (separately)

---

## Step 1: Deploy Backend First (on Render)

### 1.1 Create MongoDB Atlas (Free)
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create free M0 cluster
3. Create database user
4. Whitelist `0.0.0.0/0`
5. Copy connection string

### 1.2 Deploy Backend on Render
1. Go to [render.com](https://render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name**: `axion-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   ```
   MONGO_URL=mongodb+srv://your-connection-string
   DB_NAME=axion
   ```
6. Click **Create Web Service**
7. **Note your backend URL**: `https://axion-backend.onrender.com`

---

## Step 2: Deploy Frontend on Vercel

### 2.1 Create `vercel-deploy` Branch

```bash
# Clone your repo
git clone https://github.com/Soumadeep21/axion.git
cd axion

# Create new branch
git checkout -b vercel-deploy

# Update frontend .env for production
echo "REACT_APP_BACKEND_URL=https://axion-backend.onrender.com" > frontend/.env.production

# Commit
git add .
git commit -m "Setup for Vercel deployment"

# Push branch
git push origin vercel-deploy
```

### 2.2 Deploy on Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **Add New** → **Project**
3. Import your GitHub repo
4. Configure:
   - **Branch**: `vercel-deploy`
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
5. Add Environment Variable:
   ```
   REACT_APP_BACKEND_URL = https://axion-backend.onrender.com
   ```
6. Click **Deploy**

---

## Step 3: Update vercel.json (if needed)

Edit `/app/vercel.json` and replace `your-backend-url.onrender.com` with your actual backend URL:

```json
{
  "version": 2,
  "name": "axion",
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "https://axion-backend.onrender.com/api/$1"
    }
  ],
  "env": {
    "REACT_APP_BACKEND_URL": "https://axion-backend.onrender.com"
  }
}
```

---

## Alternative: Simpler Vercel Setup

If you just want to deploy frontend without API rewrites:

### In Vercel Dashboard:
1. Import repo
2. Set **Root Directory**: `frontend`
3. Add env var: `REACT_APP_BACKEND_URL=https://your-backend.onrender.com`
4. Deploy!

---

## File Structure for Vercel Branch

```
axion/
├── frontend/           ← Vercel deploys this
│   ├── package.json
│   ├── src/
│   ├── public/
│   └── .env.production ← Add this file
├── backend/            ← Deploy separately on Render
├── vercel.json         ← Optional config
└── README.md
```

---

## Environment Variables Summary

### Vercel (Frontend)
```
REACT_APP_BACKEND_URL=https://axion-backend.onrender.com
```

### Render (Backend)
```
MONGO_URL=mongodb+srv://...
DB_NAME=axion
EMERGENT_LLM_KEY=your-key (optional)
```

---

## Quick Commands

```bash
# Create branch and push
git checkout -b vercel-deploy
echo "REACT_APP_BACKEND_URL=https://axion-backend.onrender.com" > frontend/.env.production
git add .
git commit -m "Vercel deployment setup"
git push origin vercel-deploy
```

---

## URLs After Deployment

- **Frontend (Vercel)**: `https://axion.vercel.app`
- **Backend (Render)**: `https://axion-backend.onrender.com`

---

## Troubleshooting

### CORS Errors
Make sure backend allows your Vercel domain:
```python
# In server.py
allow_origins=["https://axion.vercel.app", "https://*.vercel.app"]
```

### API Not Working
Check that `REACT_APP_BACKEND_URL` is set correctly in Vercel dashboard.

### Build Fails
- Check Node version (should be 18+)
- Clear cache: Vercel Dashboard → Settings → Clear Build Cache

---

**Need Help?** Open an issue on [GitHub](https://github.com/Soumadeep21/axion/issues)
