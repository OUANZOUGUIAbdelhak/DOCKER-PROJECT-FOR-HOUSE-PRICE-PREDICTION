import { useState, useEffect } from 'react'
import axios from 'axios'
import PredictionForm from './components/PredictionForm'
import PredictionResult from './components/PredictionResult'
import Header from './components/Header'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [featureOptions, setFeatureOptions] = useState(null)

  useEffect(() => {
    // Load feature options on mount
    axios.get(`${API_URL}/features/options`)
      .then(response => {
        console.log('Feature options loaded:', response.data)
        setFeatureOptions(response.data)
      })
      .catch(err => {
        console.error('Failed to load options:', err)
        // Set fallback options if API fails
        setFeatureOptions({
          Neighborhood: ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste", "Gilbert"],
          HouseStyle: ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Fin", "2.5Unf", "Split", "SplitFoyer"],
          BldgType: ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
          ExterQual: ["Ex", "Gd", "TA", "Fa", "Po"],
          KitchenQual: ["Ex", "Gd", "TA", "Fa", "Po"]
        })
      })
  }, [])

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, formData)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please check your inputs.')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Form */}
          <div className="glass-effect rounded-2xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">
              Enter House Details
            </h2>
            <PredictionForm 
              onSubmit={handlePredict} 
              loading={loading}
              featureOptions={featureOptions}
            />
          </div>

          {/* Right Column - Results */}
          <div className="glass-effect rounded-2xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">
              Prediction Result
            </h2>
            <PredictionResult 
              prediction={prediction}
              loading={loading}
              error={error}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-8 text-gray-600">
        <p>Powered by Machine Learning • Built with React & FastAPI</p>
      </footer>
    </div>
  )
}

export default App
