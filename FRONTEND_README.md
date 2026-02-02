# 🎨 Frontend & API Setup Guide

This guide explains how to set up and run the beautiful React frontend and FastAPI backend.

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
│                                                         │
│  ┌──────────────────────────────────────────────┐     │
│  │  React Frontend (Port 3000)                  │     │
│  │  • Beautiful UI                              │     │
│  │  • Form inputs                               │     │
│  │  • Real-time predictions                     │     │
│  └──────────────────────────────────────────────┘     │
│           │                                            │
│           │ HTTP Requests                              │
│           ▼                                            │
│  ┌──────────────────────────────────────────────┐     │
│  │  FastAPI Backend (Port 8000)                 │     │
│  │  • REST API                                   │     │
│  │  • Model predictions                         │     │
│  │  • Auto-generated docs                       │     │
│  └──────────────────────────────────────────────┘     │
│           │                                            │
│           │ Loads model                                │
│           ▼                                            │
│  ┌──────────────────────────────────────────────┐     │
│  │  Models Volume                                │     │
│  │  • model.pkl                                 │     │
│  │  • preprocessor.pkl                          │     │
│  └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Start all services (API + Frontend)
docker-compose up api frontend

# Or start in background
docker-compose up -d api frontend
```

Then open:
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

### Option 2: Run Services Separately

#### Start API Backend

```bash
# Build API image
docker build -f api/Dockerfile -t house-price-api .

# Run API
docker run -p 8000:8000 -v ${PWD}/models:/app/models house-price-api
```

#### Start Frontend (Development)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend will be available at http://localhost:3000

## 🎨 Frontend Features

### Beautiful Modern Design
- ✨ Glass-morphism effects
- 🎨 Gradient backgrounds
- 📱 Fully responsive
- ⚡ Smooth animations
- 🎯 Intuitive UX

### Features
- 📋 **Quick Fill**: Fill form with sample data
- 💰 **Real-time Predictions**: Get instant price estimates
- 📊 **Price Breakdown**: See estimated ranges
- 🎯 **Form Validation**: Ensures correct inputs
- 📱 **Mobile Friendly**: Works on all devices

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Predict Price
```bash
POST /predict
Content-Type: application/json

{
  "LotArea": 8450,
  "YearBuilt": 2003,
  "OverallQual": 7,
  ...
}
```

#### 3. Get Feature Options
```bash
GET /features/options
```

#### 4. API Documentation
```
http://localhost:8000/docs
```

## 🛠️ Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (with hot reload)
npm run dev

# Build for production
npm run build
```

### API Development

```bash
# Install dependencies
pip install -r api/requirements.txt

# Run API
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # App header
│   │   ├── PredictionForm.jsx   # Input form
│   │   └── PredictionResult.jsx # Results display
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Tailwind styles
├── package.json
├── vite.config.js
├── tailwind.config.js
└── Dockerfile

api/
├── main.py                     # FastAPI app
├── requirements.txt
└── Dockerfile
```

## 🎯 Usage Example

1. **Open Frontend**: http://localhost:3000
2. **Click "Fill Sample Data"** to populate form
3. **Click "Predict House Price"**
4. **View Results**: See predicted price with breakdown

## 🔧 Configuration

### Environment Variables

**Frontend** (`.env` file):
```env
VITE_API_URL=http://localhost:8000
```

**API**: No configuration needed (reads models from volume)

## 🐛 Troubleshooting

### Frontend can't connect to API

**Problem**: CORS errors or connection refused

**Solution**:
1. Ensure API is running: `docker ps`
2. Check API URL in frontend `.env`
3. Verify CORS settings in `api/main.py`

### Model not found

**Problem**: API returns 503 error

**Solution**:
1. Ensure model is trained: `ls models/model.pkl`
2. Check volume mount: `docker-compose.yml`
3. Verify model path in API

### Port already in use

**Problem**: Port 3000 or 8000 already in use

**Solution**:
```bash
# Change ports in docker-compose.yml
ports:
  - "3001:80"  # Frontend
  - "8001:8000"  # API
```

## 📚 Tech Stack

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Axios**: HTTP client

### Backend
- **FastAPI**: Modern Python API framework
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## 🎓 Learning Points

1. **API Design**: RESTful endpoints with validation
2. **Frontend-Backend Communication**: HTTP requests
3. **Docker Networking**: Services communicate via network
4. **Modern UI/UX**: Responsive, beautiful interfaces
5. **Production Deployment**: Nginx for static files

## 🚀 Next Steps

1. Add authentication
2. Save prediction history
3. Add more visualizations
4. Deploy to cloud (AWS, GCP, Azure)
5. Add CI/CD pipeline

---

**Enjoy your beautiful ML prediction interface! 🎉**
