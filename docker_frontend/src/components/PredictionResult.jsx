export default function PredictionResult({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-gray-600 text-lg">Analyzing house features...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border-2 border-red-200 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-2">
          <span className="text-2xl">⚠️</span>
          <h3 className="text-lg font-semibold text-red-800">Error</h3>
        </div>
        <p className="text-red-700">{error}</p>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="text-6xl mb-4">🏡</div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">
          Ready to Predict
        </h3>
        <p className="text-gray-500">
          Fill out the form and click "Predict House Price" to get an estimate
        </p>
      </div>
    )
  }

  const price = prediction.predicted_price
  const formattedPrice = prediction.predicted_price_formatted

  // Determine price category for styling
  const getPriceCategory = (price) => {
    if (price < 100000) return { label: 'Budget', color: 'green' }
    if (price < 200000) return { label: 'Affordable', color: 'blue' }
    if (price < 300000) return { label: 'Mid-Range', color: 'purple' }
    if (price < 500000) return { label: 'Premium', color: 'orange' }
    return { label: 'Luxury', color: 'red' }
  }

  const category = getPriceCategory(price)

  return (
    <div className="space-y-6">
      {/* Main Price Display */}
      <div className="bg-gradient-to-br from-primary-500 to-primary-700 rounded-2xl p-8 text-white text-center shadow-2xl">
        <div className="text-sm font-medium opacity-90 mb-2">PREDICTED PRICE</div>
        <div className="text-5xl font-bold mb-2">{formattedPrice}</div>
        <div className={`inline-block px-4 py-1 rounded-full bg-white/20 text-sm font-semibold`}>
          {category.label} Category
        </div>
      </div>

      {/* Price Breakdown */}
      <div className="bg-gray-50 rounded-xl p-6 space-y-4">
        <h4 className="font-semibold text-gray-700 mb-4">Price Breakdown</h4>
        
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Base Estimate</span>
            <span className="font-semibold text-gray-800">{formattedPrice}</span>
          </div>
          
          <div className="border-t pt-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Estimated Range</span>
              <span className="font-semibold text-gray-800">
                ${(price * 0.9).toLocaleString(undefined, {maximumFractionDigits: 0})} - ${(price * 1.1).toLocaleString(undefined, {maximumFractionDigits: 0})}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">±10% margin</p>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <div className="flex items-start space-x-3">
          <span className="text-xl">ℹ️</span>
          <div className="text-sm text-blue-800">
            <p className="font-semibold mb-1">Note:</p>
            <p>This is an ML model prediction based on similar houses. Actual market value may vary based on location, condition, and market conditions.</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => window.print()}
          className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors font-medium"
        >
          📄 Print
        </button>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-primary-100 hover:bg-primary-200 text-primary-700 rounded-lg transition-colors font-medium"
        >
          🔄 New Prediction
        </button>
      </div>
    </div>
  )
}
