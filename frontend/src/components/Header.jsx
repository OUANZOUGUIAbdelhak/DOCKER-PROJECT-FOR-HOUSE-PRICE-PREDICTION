export default function Header() {
  return (
    <header className="glass-effect shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
              <span className="text-2xl">🏠</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">House Price Predictor</h1>
              <p className="text-sm text-gray-600">ML-Powered Real Estate Valuation</p>
            </div>
          </div>
          <div className="hidden md:flex items-center space-x-4">
            <span className="px-4 py-2 bg-green-100 text-green-800 rounded-full text-sm font-semibold">
              ✓ Model Ready
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
