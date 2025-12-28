
import React, { useState, useEffect } from 'react';
import { 
  Send, 
  TrendingUp, 
  FileText, 
  Code, 
  Settings, 
  Bell, 
  Plus, 
  LayoutDashboard,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { MarketChart } from './components/MarketChart';
import { TradingSignal, SignalDirection, AnalysisReport, TelegramConfig } from './types';
import { generateTechnicalReport, generatePineScript } from './services/geminiService';
import { sendSignalToTelegram, sendReportToTelegram } from './services/telegramService';

const App: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'reports' | 'pinescript' | 'settings'>('dashboard');
  const [loading, setLoading] = useState(false);
  const [telegramConfig, setTelegramConfig] = useState<TelegramConfig>({
    botToken: localStorage.getItem('tg_bot_token') || '',
    chatId: localStorage.getItem('tg_chat_id') || ''
  });

  const [signal, setSignal] = useState<TradingSignal>({
    id: '',
    symbol: 'BTC/USDT',
    direction: SignalDirection.BUY,
    entry: 0,
    tp1: 0,
    tp2: 0,
    sl: 0,
    timeframe: '1H',
    strategy: 'RSI Divergence',
    timestamp: new Date()
  });

  const [analysisContext, setAnalysisContext] = useState('');
  const [currentReport, setCurrentReport] = useState<AnalysisReport | null>(null);
  const [pineScript, setPineScript] = useState('');
  const [pinePrompt, setPinePrompt] = useState('');

  // Save config
  useEffect(() => {
    localStorage.setItem('tg_bot_token', telegramConfig.botToken);
    localStorage.setItem('tg_chat_id', telegramConfig.chatId);
  }, [telegramConfig]);

  const handleBroadcastSignal = async () => {
    try {
      setLoading(true);
      await sendSignalToTelegram(telegramConfig, signal);
      alert('Signal Broadcasted Successfully!');
    } catch (err: any) {
      alert(`Broadcast failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!analysisContext) return;
    try {
      setLoading(true);
      const report = await generateTechnicalReport(signal.symbol, signal.timeframe, analysisContext);
      setCurrentReport(report);
    } catch (err: any) {
      alert(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBroadcastReport = async () => {
    if (!currentReport) return;
    try {
      setLoading(true);
      await sendReportToTelegram(telegramConfig, currentReport);
      alert('Report Broadcasted Successfully!');
    } catch (err: any) {
      alert(`Broadcast failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePine = async () => {
    if (!pinePrompt) return;
    try {
      setLoading(true);
      const code = await generatePineScript("Custom Strategy", pinePrompt);
      setPineScript(code);
    } catch (err: any) {
      alert(`Code generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#0f172a] text-slate-200">
      {/* Sidebar */}
      <nav className="w-full md:w-64 bg-[#1e293b] border-r border-slate-700 flex flex-col p-4">
        <div className="flex items-center gap-3 mb-10 px-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Zap className="text-white w-6 h-6" />
          </div>
          <h1 className="font-bold text-xl tracking-tight text-white">TradeCast Pro</h1>
        </div>

        <div className="space-y-2 flex-1">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'dashboard' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}
          >
            <LayoutDashboard size={20} />
            <span className="font-medium">Dashboard</span>
          </button>
          <button 
            onClick={() => setActiveTab('reports')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'reports' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}
          >
            <FileText size={20} />
            <span className="font-medium">Analysis Reports</span>
          </button>
          <button 
            onClick={() => setActiveTab('pinescript')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'pinescript' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}
          >
            <Code size={20} />
            <span className="font-medium">Pine Script v6</span>
          </button>
          <button 
            onClick={() => setActiveTab('settings')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === 'settings' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}
          >
            <Settings size={20} />
            <span className="font-medium">Settings</span>
          </button>
        </div>

        <div className="mt-auto pt-4 border-t border-slate-700">
          <div className="p-4 bg-slate-800 rounded-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Status</span>
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            </div>
            <p className="text-sm font-medium text-white truncate">{telegramConfig.chatId ? 'Connected to Telegram' : 'Configure Telegram'}</p>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8">
        <header className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
          <div>
            <h2 className="text-3xl font-bold text-white mb-1 capitalize">{activeTab}</h2>
            <p className="text-slate-400">Professional trading signal broadcasting engine</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="bg-slate-800 p-2 rounded-full text-slate-400 hover:text-white transition-colors relative">
              <Bell size={20} />
              <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <div className="h-10 w-10 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center font-bold text-white">
              TC
            </div>
          </div>
        </header>

        {/* Dynamic Content */}
        <div className="animate-in fade-in duration-500">
          {activeTab === 'dashboard' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 space-y-8">
                <MarketChart />
                
                <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-6">
                  <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <TrendingUp className="text-blue-500" />
                    Signal Generator
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Symbol</label>
                        <input 
                          type="text" 
                          value={signal.symbol}
                          onChange={(e) => setSignal({...signal, symbol: e.target.value})}
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-600 outline-none transition-all"
                          placeholder="e.g. BTC/USDT"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Entry Price</label>
                        <input 
                          type="number" 
                          value={signal.entry}
                          onChange={(e) => setSignal({...signal, entry: Number(e.target.value)})}
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:ring-2 focus:ring-blue-600"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-slate-400 mb-1">TP 1</label>
                          <input 
                            type="number" 
                            value={signal.tp1}
                            onChange={(e) => setSignal({...signal, tp1: Number(e.target.value)})}
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:ring-2 focus:ring-blue-600"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-slate-400 mb-1">TP 2</label>
                          <input 
                            type="number" 
                            value={signal.tp2}
                            onChange={(e) => setSignal({...signal, tp2: Number(e.target.value)})}
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:ring-2 focus:ring-blue-600"
                          />
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Direction</label>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => setSignal({...signal, direction: SignalDirection.BUY})}
                            className={`flex-1 py-2 rounded-lg font-bold transition-all ${signal.direction === SignalDirection.BUY ? 'bg-green-600 text-white shadow-lg shadow-green-900/30' : 'bg-slate-900 text-slate-400 border border-slate-700'}`}
                          >
                            BUY
                          </button>
                          <button 
                            onClick={() => setSignal({...signal, direction: SignalDirection.SELL})}
                            className={`flex-1 py-2 rounded-lg font-bold transition-all ${signal.direction === SignalDirection.SELL ? 'bg-red-600 text-white shadow-lg shadow-red-900/30' : 'bg-slate-900 text-slate-400 border border-slate-700'}`}
                          >
                            SELL
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Stop Loss</label>
                        <input 
                          type="number" 
                          value={signal.sl}
                          onChange={(e) => setSignal({...signal, sl: Number(e.target.value)})}
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:ring-2 focus:ring-blue-600"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-1">Timeframe</label>
                        <select 
                          value={signal.timeframe}
                          onChange={(e) => setSignal({...signal, timeframe: e.target.value})}
                          className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white outline-none focus:ring-2 focus:ring-blue-600"
                        >
                          <option>1M</option>
                          <option>5M</option>
                          <option>15M</option>
                          <option>1H</option>
                          <option>4H</option>
                          <option>1D</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  <button 
                    onClick={handleBroadcastSignal}
                    disabled={loading}
                    className="w-full mt-8 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:cursor-not-allowed py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-xl shadow-blue-900/20"
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Send size={20} />}
                    {loading ? 'Processing...' : 'Broadcast to Telegram'}
                  </button>
                </div>
              </div>

              <div className="space-y-8">
                <div className="bg-slate-800 rounded-2xl border border-slate-700 p-6">
                  <h3 className="text-lg font-bold text-white mb-4">Signal History</h3>
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="bg-slate-900 p-4 rounded-xl border border-slate-700 flex items-center justify-between">
                        <div>
                          <p className="font-bold text-white">ETH/USDT</p>
                          <p className="text-xs text-slate-500">2 hours ago • Buy</p>
                        </div>
                        <div className="text-right">
                          <p className="text-green-500 font-bold text-sm">+2.4%</p>
                          <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-1 rounded">COMPLETED</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-indigo-600 to-blue-700 rounded-2xl p-6 text-white shadow-xl shadow-blue-900/30">
                  <div className="flex items-center gap-3 mb-4">
                    <Zap className="fill-white" />
                    <h3 className="font-bold">Pro Account</h3>
                  </div>
                  <p className="text-sm text-blue-100 mb-6">Upgrade now to access advanced technical indicators and multi-channel broadcasting.</p>
                  <button className="w-full bg-white text-blue-700 py-3 rounded-xl font-bold hover:bg-blue-50 transition-colors">
                    Upgrade Now
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'reports' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-6">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <FileText className="text-blue-500" />
                  Analysis Generator
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-400 mb-1">Asset Context</label>
                    <textarea 
                      value={analysisContext}
                      onChange={(e) => setAnalysisContext(e.target.value)}
                      placeholder="Paste chart observations, news snippets, or key levels here..."
                      className="w-full h-48 bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-blue-600 outline-none resize-none"
                    />
                  </div>
                  <button 
                    onClick={handleGenerateReport}
                    disabled={loading || !analysisContext}
                    className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg"
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Zap size={18} />}
                    {loading ? 'Analyzing...' : 'Generate Technical Report'}
                  </button>
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-6 flex flex-col">
                <h3 className="text-xl font-bold text-white mb-6">Preview Report</h3>
                {currentReport ? (
                  <div className="flex-1 space-y-6 overflow-y-auto max-h-[500px] pr-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-2xl font-bold text-white">{currentReport.title}</h4>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                        currentReport.outlook === 'BULLISH' ? 'bg-green-500/20 text-green-500 border border-green-500/30' :
                        currentReport.outlook === 'BEARISH' ? 'bg-red-500/20 text-red-500 border border-red-500/30' :
                        'bg-slate-500/20 text-slate-400 border border-slate-500/30'
                      }`}>
                        {currentReport.outlook}
                      </span>
                    </div>
                    <div className="p-4 bg-slate-900 rounded-xl border border-slate-700">
                      <p className="text-sm font-semibold text-blue-400 mb-2 uppercase tracking-wide">Summary</p>
                      <p className="text-slate-300 leading-relaxed">{currentReport.summary}</p>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-blue-400 mb-2 uppercase tracking-wide">Technical Details</p>
                      <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line">{currentReport.technicalDetails}</p>
                    </div>
                    <button 
                      onClick={handleBroadcastReport}
                      disabled={loading}
                      className="w-full mt-4 bg-green-600 hover:bg-green-500 py-3 rounded-xl font-bold flex items-center justify-center gap-2"
                    >
                      <Send size={18} />
                      Send to Telegram
                    </button>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-slate-500 space-y-4 border-2 border-dashed border-slate-700 rounded-xl">
                    <AlertCircle size={48} className="opacity-20" />
                    <p>No report generated yet. Use the tool on the left.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'pinescript' && (
            <div className="space-y-8">
              <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-6">
                <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                  <Code className="text-blue-500" />
                  Pine Script v6 Editor
                </h3>
                <p className="text-slate-400 mb-6">Prompt the AI to create custom TradingView indicators and strategies in Pine Script v6.</p>
                <div className="space-y-4">
                  <div className="flex gap-4">
                    <input 
                      value={pinePrompt}
                      onChange={(e) => setPinePrompt(e.target.value)}
                      placeholder="e.g. Create a RSI and MACD crossover strategy with volume confirmation..."
                      className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-blue-600 outline-none"
                    />
                    <button 
                      onClick={handleGeneratePine}
                      disabled={loading || !pinePrompt}
                      className="bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 px-8 rounded-xl font-bold transition-all shadow-lg flex items-center gap-2"
                    >
                      {loading ? <Loader2 size={18} className="animate-spin" /> : <Plus size={20} />}
                      Generate
                    </button>
                  </div>
                </div>
              </div>

              {pineScript && (
                <div className="bg-slate-900 rounded-2xl border border-slate-700 overflow-hidden shadow-2xl">
                  <div className="bg-slate-800 px-6 py-3 border-b border-slate-700 flex justify-between items-center">
                    <span className="text-sm font-mono text-slate-300">strategy.pine</span>
                    <button 
                      onClick={() => {
                        navigator.clipboard.writeText(pineScript);
                        alert('Copied to clipboard!');
                      }}
                      className="text-xs bg-slate-700 hover:bg-slate-600 text-white px-3 py-1 rounded-md transition-all"
                    >
                      Copy Code
                    </button>
                  </div>
                  <pre className="p-6 font-mono text-sm overflow-x-auto text-emerald-400 leading-relaxed max-h-[600px]">
                    <code>{pineScript}</code>
                  </pre>
                </div>
              )}
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="max-w-2xl mx-auto space-y-8">
              <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-8">
                <h3 className="text-2xl font-bold text-white mb-6">Telegram Integration</h3>
                <div className="space-y-6">
                  <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-xl flex gap-3 text-sm text-blue-200">
                    <AlertCircle size={20} className="shrink-0" />
                    <p>Credentials are stored locally in your browser. Ensure your bot is an admin in the target group/channel.</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-400 mb-2">Bot Token (from @BotFather)</label>
                    <input 
                      type="password" 
                      value={telegramConfig.botToken}
                      onChange={(e) => setTelegramConfig({...telegramConfig, botToken: e.target.value})}
                      placeholder="e.g. 123456789:ABCdefG..."
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white focus:ring-2 focus:ring-blue-600 outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-400 mb-2">Channel/Chat ID</label>
                    <input 
                      type="text" 
                      value={telegramConfig.chatId}
                      onChange={(e) => setTelegramConfig({...telegramConfig, chatId: e.target.value})}
                      placeholder="e.g. @mytradingchannel or -100123456789"
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white focus:ring-2 focus:ring-blue-600 outline-none"
                    />
                  </div>
                  <div className="flex items-center gap-2 p-4 bg-slate-900/50 rounded-xl border border-slate-700">
                    <CheckCircle className="text-green-500" size={20} />
                    <span className="text-sm font-medium">Automatic save enabled</span>
                  </div>
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-8">
                <h3 className="text-2xl font-bold text-white mb-6">System Preferences</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-slate-900 rounded-xl">
                    <div>
                      <p className="font-bold text-white">Default Timeframe</p>
                      <p className="text-xs text-slate-500">Auto-select for new signals</p>
                    </div>
                    <select className="bg-slate-800 border-none rounded-lg px-4 py-2 text-sm text-white">
                      <option>15M</option>
                      <option>1H</option>
                      <option>4H</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-slate-900 rounded-xl">
                    <div>
                      <p className="font-bold text-white">Language</p>
                      <p className="text-xs text-slate-500">Broadcasting language</p>
                    </div>
                    <select className="bg-slate-800 border-none rounded-lg px-4 py-2 text-sm text-white">
                      <option>English</option>
                      <option>Spanish</option>
                      <option>German</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
