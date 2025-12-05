import { useState } from 'react'
import AnalysisPage from './pages/AnalysisPage'
import MultiCameraPage from './pages/MultiCameraPage'
import { Video, Camera } from 'lucide-react'

function App() {
  const [mode, setMode] = useState('simple') // 'simple' or 'advanced'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Mode Toggle Header */}
      <div className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">
                Traffic Vehicle Analytics
              </h1>
              <p className="text-sm text-slate-400">
                {mode === 'simple' ? 'Single video analysis' : 'Multi-camera fusion with real-world coordinates'}
              </p>
            </div>
            
            {/* Mode Toggle */}
            <div className="flex items-center gap-2 bg-slate-700/50 rounded-lg p-1">
              <button
                onClick={() => setMode('simple')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition ${
                  mode === 'simple'
                    ? 'bg-gradient-to-r from-blue-500 to-emerald-500 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Video size={18} />
                Simple Mode
              </button>
              <button
                onClick={() => setMode('advanced')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition ${
                  mode === 'advanced'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Camera size={18} />
                Advanced Mode
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div>
        {mode === 'simple' ? <AnalysisPage /> : <MultiCameraPage />}
      </div>
    </div>
  )
}

export default App
