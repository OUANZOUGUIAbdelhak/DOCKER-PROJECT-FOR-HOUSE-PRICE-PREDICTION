# 🚀 Quick Start: React UI & API

Get your beautiful ML prediction interface running in minutes!

## Prerequisites

- Docker and Docker Compose installed
- Trained model in `models/` directory (run training first)

## 🎯 Quick Start (3 Steps)

### Step 1: Ensure Model is Trained

```bash
# Check if model exists
ls models/model.pkl

# If not, train the model first
docker-compose run --rm training
```

### Step 2: Start API and Frontend

```bash
# Start both services
docker-compose up api frontend

# Or run in background
docker-compose up -d api frontend
```

### Step 3: Open Your Browser

- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

That's it! 🎉

## 🎨 Using the Interface

1. **Fill the Form**: 
   - Click "📋 Fill Sample Data" for quick testing
   - Or manually enter house details

2. **Get Prediction**:
   - Click "💰 Predict House Price"
   - Wait a few seconds for the ML model to process

3. **View Results**:
   - See predicted price with beautiful formatting
   - View price breakdown and estimated range
   - Get insights about the prediction

## 🛠️ Development Mode

### Run Frontend Locally (Hot Reload)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000 (or Vite's default port)

### Run API Locally

```bash
pip install -r api/requirements.txt
cd api
uvicorn main:app --reload
```

API runs on http://localhost:8000

## 📱 Features

✨ **Beautiful Modern Design**
- Glass-morphism effects
- Gradient backgrounds  
- Smooth animations
- Fully responsive

🎯 **Smart Features**
- Form validation
- Quick fill sample data
- Real-time predictions
- Price breakdown
- Error handling

## 🔧 Troubleshooting

### Port Already in Use

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "3001:80"   # Frontend
  - "8001:8000" # API
```

### Frontend Can't Connect to API

1. Check API is running: `docker ps`
2. Check API logs: `docker-compose logs api`
3. Verify CORS settings in `api/main.py`

### Model Not Found

1. Train model first: `docker-compose run --rm training`
2. Verify model exists: `ls models/model.pkl`
3. Check volume mount in `docker-compose.yml`

## 📚 Learn More

- See `FRONTEND_README.md` for detailed documentation
- Check `api/main.py` for API endpoints
- Explore `frontend/src/` for React components

---

**Enjoy your beautiful ML prediction interface! 🏠✨**
