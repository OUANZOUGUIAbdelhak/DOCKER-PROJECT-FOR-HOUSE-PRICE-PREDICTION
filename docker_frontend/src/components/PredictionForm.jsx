import { useState, useEffect } from 'react'

export default function PredictionForm({ onSubmit, loading, featureOptions }) {
  const [formData, setFormData] = useState({
    LotArea: '',
    YearBuilt: '',
    YearRemodAdd: '',
    OverallQual: '',
    OverallCond: '',
    GrLivArea: '',
    TotalBsmtSF: '',
    FirstFlrSF: '',
    SecondFlrSF: '',
    FullBath: '',
    HalfBath: '',
    BsmtFullBath: '',
    BsmtHalfBath: '',
    BedroomAbvGr: '',
    TotRmsAbvGrd: '',
    GarageCars: '',
    GarageArea: '',
    Neighborhood: '',
    HouseStyle: '',
    BldgType: '',
    ExterQual: '',
    KitchenQual: '',
    Fireplaces: '',
    PoolArea: ''
  })

  // Default options if API fails to load
  const defaultOptions = {
    Neighborhood: ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste", "Gilbert"],
    HouseStyle: ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Fin", "2.5Unf", "Split", "SplitFoyer"],
    BldgType: ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
    ExterQual: ["Ex", "Gd", "TA", "Fa", "Po"],
    KitchenQual: ["Ex", "Gd", "TA", "Fa", "Po"]
  }

  // Use featureOptions if available, otherwise use defaults
  const options = featureOptions || defaultOptions

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Convert form data to proper types
    const submitData = {}
    for (const [key, value] of Object.entries(formData)) {
      if (value === '') {
        // Skip empty optional fields
        if (['YearRemodAdd', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 
             'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea',
             'Fireplaces', 'PoolArea'].includes(key)) {
          continue
        }
        submitData[key] = null
      } else if (['YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond',
                   'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                   'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'Fireplaces'].includes(key)) {
        submitData[key] = parseInt(value)
      } else if (['LotArea', 'GrLivArea', 'TotalBsmtSF', 'FirstFlrSF', 
                   'SecondFlrSF', 'GarageArea', 'PoolArea'].includes(key)) {
        submitData[key] = parseFloat(value)
      } else {
        submitData[key] = value
      }
    }
    
    onSubmit(submitData)
  }

  const handleQuickFill = () => {
    setFormData({
      LotArea: '8450',
      YearBuilt: '2003',
      YearRemodAdd: '2003',
      OverallQual: '7',
      OverallCond: '5',
      GrLivArea: '1710',
      TotalBsmtSF: '856',
      FirstFlrSF: '856',
      SecondFlrSF: '854',
      FullBath: '2',
      HalfBath: '1',
      BsmtFullBath: '1',
      BsmtHalfBath: '0',
      BedroomAbvGr: '3',
      TotRmsAbvGrd: '8',
      GarageCars: '2',
      GarageArea: '548',
      Neighborhood: 'Veenker',
      HouseStyle: '2Story',
      BldgType: '1Fam',
      ExterQual: 'Gd',
      KitchenQual: 'Gd',
      Fireplaces: '0',
      PoolArea: '0'
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Quick Fill Button */}
      <button
        type="button"
        onClick={handleQuickFill}
        className="w-full py-2 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors text-sm font-medium"
      >
        📋 Fill Sample Data
      </button>

      {/* Property Basics */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Property Basics</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Lot Area (sq ft) *
            </label>
            <input
              type="number"
              name="LotArea"
              value={formData.LotArea}
              onChange={handleChange}
              required
              className="input-field"
              placeholder="8450"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Year Built *
            </label>
            <input
              type="number"
              name="YearBuilt"
              value={formData.YearBuilt}
              onChange={handleChange}
              required
              min="1800"
              max="2024"
              className="input-field"
              placeholder="2003"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Remodel Year
            </label>
            <input
              type="number"
              name="YearRemodAdd"
              value={formData.YearRemodAdd}
              onChange={handleChange}
              min="1800"
              max="2024"
              className="input-field"
              placeholder="2003"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Overall Quality (1-10) *
            </label>
            <input
              type="number"
              name="OverallQual"
              value={formData.OverallQual}
              onChange={handleChange}
              required
              min="1"
              max="10"
              className="input-field"
              placeholder="7"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Overall Condition (1-10) *
          </label>
          <input
            type="number"
            name="OverallCond"
            value={formData.OverallCond}
            onChange={handleChange}
            required
            min="1"
            max="10"
            className="input-field"
            placeholder="5"
          />
        </div>
      </div>

      {/* Living Space */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Living Space</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Above Grade Area (sq ft) *
            </label>
            <input
              type="number"
              name="GrLivArea"
              value={formData.GrLivArea}
              onChange={handleChange}
              required
              className="input-field"
              placeholder="1710"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Total Basement (sq ft)
            </label>
            <input
              type="number"
              name="TotalBsmtSF"
              value={formData.TotalBsmtSF}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="856"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              First Floor (sq ft)
            </label>
            <input
              type="number"
              name="FirstFlrSF"
              value={formData.FirstFlrSF}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="856"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Second Floor (sq ft)
            </label>
            <input
              type="number"
              name="SecondFlrSF"
              value={formData.SecondFlrSF}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="854"
            />
          </div>
        </div>
      </div>

      {/* Bathrooms & Bedrooms */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Bathrooms & Bedrooms</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Full Bathrooms *
            </label>
            <input
              type="number"
              name="FullBath"
              value={formData.FullBath}
              onChange={handleChange}
              required
              min="0"
              className="input-field"
              placeholder="2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Half Bathrooms
            </label>
            <input
              type="number"
              name="HalfBath"
              value={formData.HalfBath}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="1"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Basement Full Bath
            </label>
            <input
              type="number"
              name="BsmtFullBath"
              value={formData.BsmtFullBath}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Basement Half Bath
            </label>
            <input
              type="number"
              name="BsmtHalfBath"
              value={formData.BsmtHalfBath}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="0"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Bedrooms Above Grade *
            </label>
            <input
              type="number"
              name="BedroomAbvGr"
              value={formData.BedroomAbvGr}
              onChange={handleChange}
              required
              min="0"
              className="input-field"
              placeholder="3"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Total Rooms Above Grade *
            </label>
            <input
              type="number"
              name="TotRmsAbvGrd"
              value={formData.TotRmsAbvGrd}
              onChange={handleChange}
              required
              min="0"
              className="input-field"
              placeholder="8"
            />
          </div>
        </div>
      </div>

      {/* Garage */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Garage</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Garage Cars
            </label>
            <input
              type="number"
              name="GarageCars"
              value={formData.GarageCars}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Garage Area (sq ft)
            </label>
            <input
              type="number"
              name="GarageArea"
              value={formData.GarageArea}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="548"
            />
          </div>
        </div>
      </div>

      {/* Location & Type */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Location & Type</h3>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Neighborhood *
          </label>
          <select
            name="Neighborhood"
            value={formData.Neighborhood}
            onChange={handleChange}
            required
            className="input-field"
          >
            <option value="">Select Neighborhood</option>
            {options.Neighborhood && options.Neighborhood.map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              House Style *
            </label>
            <select
              name="HouseStyle"
              value={formData.HouseStyle}
              onChange={handleChange}
              required
              className="input-field"
            >
              <option value="">Select Style</option>
              {options.HouseStyle && options.HouseStyle.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Building Type *
            </label>
            <select
              name="BldgType"
              value={formData.BldgType}
              onChange={handleChange}
              required
              className="input-field"
            >
              <option value="">Select Type</option>
              {options.BldgType && options.BldgType.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Quality Ratings */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Quality Ratings</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Exterior Quality *
            </label>
            <select
              name="ExterQual"
              value={formData.ExterQual}
              onChange={handleChange}
              required
              className="input-field"
            >
              <option value="">Select Quality</option>
              {options.ExterQual && options.ExterQual.map(q => (
                <option key={q} value={q}>{q} ({q === 'Ex' ? 'Excellent' : q === 'Gd' ? 'Good' : q === 'TA' ? 'Average' : q === 'Fa' ? 'Fair' : 'Poor'})</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Kitchen Quality *
            </label>
            <select
              name="KitchenQual"
              value={formData.KitchenQual}
              onChange={handleChange}
              required
              className="input-field"
            >
              <option value="">Select Quality</option>
              {options.KitchenQual && options.KitchenQual.map(q => (
                <option key={q} value={q}>{q} ({q === 'Ex' ? 'Excellent' : q === 'Gd' ? 'Good' : q === 'TA' ? 'Average' : q === 'Fa' ? 'Fair' : 'Poor'})</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Additional Features */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Additional Features</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Fireplaces
            </label>
            <input
              type="number"
              name="Fireplaces"
              value={formData.Fireplaces}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="0"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pool Area (sq ft)
            </label>
            <input
              type="number"
              name="PoolArea"
              value={formData.PoolArea}
              onChange={handleChange}
              min="0"
              className="input-field"
              placeholder="0"
            />
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading}
        className="btn-primary w-full text-lg"
      >
        {loading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Predicting...
          </span>
        ) : (
          '💰 Predict House Price'
        )}
      </button>
    </form>
  )
}
