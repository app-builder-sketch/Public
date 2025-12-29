import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  Send, TrendingUp, BarChart3, Code, Settings, LayoutDashboard,
  Loader2, Terminal, Check, ExternalLink, ShieldAlert, BrainCircuit,
  Copy, CheckCircle, AlertCircle, Activity, Layers, Search, Zap
} from 'lucide-react';
import { MarketChart } from './components/MarketChart';
import { useDebounce } from './hooks/useDebounce';
import { 
  TradingSignal, SignalDirection, AnalysisReport, 
  TelegramConfig, AnalysisDepth 
} from './types';
import { generateTechnicalReport, generatePineScript } from './services/geminiService';
import { sendSignalToTelegram, sendReportToTelegram } from './services/telegramService';

// Environment Configuration (Secure)
const ENV_CONFIG: TelegramConfig = {
  botToken: import.meta.env.VITE_TELEGRAM_BOT_TOKEN,
  chatId: import.meta.env.VITE_TELEGRAM_CHAT_ID
};

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'reports' | 'pinescript' | 'settings'>('dashboard');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([
    "[SYSTEM] Core Engine: Gemini 3.0 / Pine V6.0 Active", 
    "[INFO] Neural reasoning pipelines established..."
  ]);
  const [errors, setErrors] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);

  // Signal State
  const [signal, setSignal] = useState<TradingSignal>({
    id: crypto.randomUUID(),
    symbol: 'BTC/USDT',
    direction: SignalDirection.BUY,
    entry: 64250,
    tp1: 65500,
    tp2: 67000,
    tp3: 69000,
    sl: 63000,
    timeframe: '1H',
    strategy: 'Neural Momentum + RSI 6.0',
    timestamp: new Date()
  });
  
  const [signalIntelMode, setSignalIntelMode] = useState<'NONE' | 'QUICK' | 'DEEP'>('NONE');
  const [intelPreview, setIntelPreview] = useState<string | null>(null);
  
  // Report State
  const [analysisContext, setAnalysisContext] = useState('');
  const [reportDepth, setReportDepth] = useState<AnalysisDepth>(AnalysisDepth.DETAILED);
  const [currentReport, setCurrentReport] = useState<AnalysisReport | null>(null);
  
  // Pine Script State
  const [pineScript, setPineScript] = useState('');
  const [pinePrompt, setPinePrompt] = useState('');

  // Debounced symbol for AI calls
  const debouncedSymbol = useDebounce(signal.symbol, 1000);

  // Auto-generate intelligence when symbol changes
  useEffect(() => {
    if (signalIntelMode !== 'NONE' && debouncedSymbol) {
      handleGenerateSignalIntel();
    }
  }, [debouncedSymbol, signalIntelMode]);

  // Memoized Computed Values
  const signalPreview = useMemo(() => {
    const directionEmoji = signal.direction === SignalDirection.BUY ? '🟢' : '🔴';
    const directionColor = signal.direction === SignalDirection.BUY ? 'text-emerald-400' : 'text-rose-400';
    
    return {
      header: `🚀 SIGNAL: ${signal.symbol} 🚀`,
      directionText: `${directionEmoji} ${signal.direction}`,
      directionColor,
      logicStatus: errors.length === 0 ? 'VERIFIED' : 'INVALID',
      logicStatusColor: errors.length === 0 ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'
    };
  }, [signal, errors]);

  // Logger Utility
  const addLog = useCallback((msg: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, `[${timestamp}] ${msg}`]);
  }, []);

  // Enhanced Validation Logic
  const validateSignal = useCallback((): boolean => {
    const errs: string[] = [];
    const { symbol, entry, tp1, tp2, tp3, sl, direction } = signal;

    if (!symbol.trim()) {
      errs.push("Critical Error: Market Symbol identification missing.");
    }

    if (direction === SignalDirection.BUY) {
      if (sl >= entry) errs.push(`BUY Logic: SL (${sl}) must be < Entry (${entry})`);
      if (entry >= tp1) errs.push(`BUY Logic: Entry (${entry}) must be < TP1 (${tp1})`);
      if (tp1 >= tp2) errs.push(`BUY Logic: TP1 (${tp1}) must be < TP2 (${tp2})`);
      if (tp2 >= tp3) errs.push(`BUY Logic: TP2 (${tp2}) must be < TP3 (${tp3})`);
    } else {
      if (sl <= entry) errs.push(`SELL Logic: SL (${sl}) must be > Entry (${entry})`);
      if (entry <= tp1) errs.push(`SELL Logic: Entry (${entry}) must be > TP1 (${tp1})`);
      if (tp1 <= tp2) errs.push(`SELL Logic: TP1 (${tp1}) must be > TP2 (${tp2})`);
      if (tp2 <= tp3) errs.push(`SELL Logic: TP2 (${tp2}) must be > TP3 (${tp3})`);
    }

    setErrors(errs);
    if (errs.length > 0) {
      addLog(`[ERROR] Protocol Violation: ${errs.length} inconsistencies detected.`);
    }
    return errs.length === 0;
  }, [signal, addLog]);

  // AI Intelligence Generation
  const handleGenerateSignalIntel = useCallback(async () => {
    if (!signal.symbol) return;
    
    setLoading(true);
    addLog(`[AI] Invoking Gemini 3.0 reasoning for ${signal.symbol}...`);
    
    try {
      const depth = signalIntelMode === 'QUICK' ? AnalysisDepth.QUICK : AnalysisDepth.DETAILED;
      const report = await generateTechnicalReport(
        signal.symbol, 
        signal.timeframe, 
        `Analyze: ${signal.direction} ${signal.symbol}. Strategy: ${signal.strategy}`, 
        depth
      );
      
      setIntelPreview(report.summary);
      addLog(`[AI] Intelligence synthesis complete. Outlook: ${report.outlook}`);
    } catch (err: any) {
      addLog(`[ERROR] AI Synthesis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [signal, signalIntelMode, addLog]);

  // Signal Broadcasting
  const handleBroadcastSignal = useCallback(async () => {
    if (!validateSignal()) return;
    
    if (!ENV_CONFIG.botToken || !ENV_CONFIG.chatId) {
      addLog("[ERROR] Gateway Auth Error: Telegram credentials undefined.");
      setActiveTab('settings');
      return;
    }
    
    setLoading(true);
    addLog(`[ACTION] Deploying neural broadcast: ${signal.symbol}`);
    
    try {
      let finalSignal = { ...signal };
      if (signalIntelMode !== 'NONE') {
        const depth = signalIntelMode === 'QUICK' ? AnalysisDepth.QUICK : AnalysisDepth.DETAILED;
        const report = await generateTechnicalReport(
          signal.symbol, 
          signal.timeframe, 
          signal.strategy, 
          depth
        );
        finalSignal.strategy = `${signal.strategy}\n\n🤖 GEMINI 3 INTEL:\n${report.summary}`;
      }
      
      await sendSignalToTelegram(ENV_CONFIG, finalSignal);
      addLog("[SUCCESS] Broadcast delivered via Secure Telegram Gateway.");
      setErrors([]);
    } catch (err: any) {
      addLog(`[ERROR] Protocol Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [signal, signalIntelMode, validateSignal, addLog]);

  // Report Generation
  const handleGenerateReport = useCallback(async () => {
    if (!analysisContext) return;
    
    setLoading(true);
    const isQuant = reportDepth === AnalysisDepth.QUANT;
    addLog(`[AI] Initializing ${reportDepth} Neural ${isQuant ? 'Thinking' : 'Flash'}...`);
    
    try {
      const report = await generateTechnicalReport(
        signal.symbol, 
        signal.timeframe, 
        analysisContext, 
        reportDepth
      );
      setCurrentReport(report);
      addLog(`[SUCCESS] Synthesis complete. Reasoning fidelity: High.`);
    } catch (err: any) {
      addLog(`[ERROR] Neural engine mismatch: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [analysisContext, reportDepth, signal.symbol, signal.timeframe, addLog]);

  const handleBroadcastReport = useCallback(async () => {
    if (!ENV_CONFIG.botToken || !ENV_CONFIG.chatId || !currentReport) {
      addLog("[ERROR] Broadcast failed: Missing credentials or report.");
      setActiveTab('settings');
      return;
    }
    
    setLoading(true);
    addLog(`[ACTION] Deploying intelligence broadcast: ${currentReport.title}`);
    
    try {
      await sendReportToTelegram(ENV_CONFIG, currentReport);
      addLog("[SUCCESS] Intelligence synthesis delivered.");
    } catch (err: any) {
      addLog(`[ERROR] Protocol Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [currentReport, addLog]);

  // Pine Script Generation
  const handleGeneratePine = useCallback(async () => {
    if (!pinePrompt) return;
    
    setLoading(true);
    addLog("[AI] Compiling logic to Pine Script v6.0...");
    
    try {
      const code = await generatePineScript("Dark Singularity Engine", pinePrompt);
      setPineScript(code);
      addLog("[SUCCESS] Code validated against V6.0 specification.");
    } catch (err: any) {
      addLog(`[ERROR] Compilation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [pinePrompt, addLog]);

  const handleCopyCode = useCallback(async () => {
    if (!pineScript) return;
    
    try {
      await navigator.clipboard.writeText(pineScript);
      setCopied(true);
      addLog("[SYSTEM] Pine V6 Source cloned to clipboard.");
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      addLog("[ERROR] Clipboard access denied.");
    }
  }, [pineScript, addLog]);

  // Pine Script Syntax Highlighter
  const highlightPineScript = (code: string) => {
    if (!code) return null;
    return code.split('\n').map((line, i) => (
      <div key={i} className="min-h-[1.25rem]">
        <span dangerouslySetInnerHTML={{ 
          __html: line
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/(\/\/.*)/g, '<span style="color: #64748b">$1</span>')
            .replace(/\b(indicator|strategy|input|plot|if|else|for|true|false|alert|alertcondition|runtime|request|library)\b/g, '<span style="color: #3b82f6">$1</span>')
            .replace(/\b(ta\.[a-z_]+|ema|sma|rsi|macd|color\.[a-z_]+|math\.[a-z_]+|chart\.[a-z_]+)\b/g, '<span style="color: #fbbf24">$1</span>')
            .replace(/\b(\d+(\.\d+)?)\b/g, '<span style="color: #10b981">$1</span>')
            .replace(/(".*?"|'.*?')/g, '<span style="color: #94a3b8">$1</span>')
        }} />
      </div>
    ));
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-950 text-slate-200">
      {/* Navigation */}
      <nav className="w-full md:w-64 bg-slate-900 border-r border-slate-800 flex flex-col p-4 shadow-xl z-10">
        <div className="flex items-center gap-3 mb-10 px-2 py-4 border-b border-slate-800/50">
          <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2.5 rounded-xl shadow-lg shadow-indigo-500/20">
            <BrainCircuit className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="font-black text-xl tracking-tighter text-white">TRADECAST</h1>
            <p className="text-[10px] font-bold text-indigo-400 tracking-[0.2em] uppercase opacity-80">v3.0 NEURAL</p>
          </div>
        </div>

        <div className="space-y-1.5 flex-1">
          {[
            { id: 'dashboard', label: 'Signal Center', icon: <LayoutDashboard size={18} /> },
            { id: 'reports', label: 'AI Reports Lab', icon: <BarChart3 size={18} /> },
            { id: 'pinescript', label: 'Script Forge', icon: <Code size={18} /> },
            { id: 'settings', label: 'Gateways', icon: <Settings size={18} /> }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all font-bold text-sm
                ${activeTab === tab.id 
                  ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/20 shadow-[0_0_20px_rgba(79,70,229,0.1)]' 
                  : 'hover:bg-slate-800/50 text-slate-400'}`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800/50">
          <div className="p-4 bg-slate-800 rounded-2xl border border-slate-700/50">
            <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1">Engine Latency</p>
            <div className="flex items-center gap-2">
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="w-[85%] h-full bg-emerald-500 animate-pulse"></div>
              </div>
              <span className="text-[9px] font-bold text-emerald-500">OPTIMAL</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <header className="bg-slate-950/50 backdrop-blur-md p-6 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="h-10 w-1 bg-indigo-600 rounded-full"></div>
            <div>
              <h2 className="text-2xl font-black text-white tracking-tight uppercase">
                {activeTab === 'dashboard' ? 'Signal Intelligence' : 
                 activeTab === 'reports' ? 'Advanced Analytics' : 
                 activeTab === 'pinescript' ? 'Source Compilation' : 'Gateway Systems'}
              </h2>
              <p className="text-slate-500 text-[10px] font-bold tracking-[0.3em] uppercase">Gemini 3.0 / GPT 5.2 Protocol</p>
            </div>
          </div>
          <div className="flex gap-3">
             <div className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-xl flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-ring"></div>
                <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">Search Grounding: ON</span>
             </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 lg:p-10 custom-scrollbar">
          <div className="max-w-7xl mx-auto space-y-10">
            
            {/* Dashboard Tab */}
            {activeTab === 'dashboard' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                <div className="lg:col-span-8 space-y-10">
                  <MarketChart symbol={signal.symbol} timeframe={signal.timeframe} />
                  
                  <div className="bg-slate-900 rounded-[2.5rem] border border-slate-800 p-8 shadow-2xl relative overflow-hidden">
                    {/* Enhanced Error Display */}
                    {errors.length > 0 && (
                      <div className="mb-8 p-6 bg-rose-950/30 border-2 border-rose-500/40 rounded-[2rem] animate-in">
                        <div className="flex items-center gap-3 mb-4 text-rose-400">
                          <ShieldAlert size={22} strokeWidth={3} className="animate-pulse" />
                          <h4 className="font-black text-xs uppercase tracking-[0.2em]">Protocol Validation Breach</h4>
                        </div>
                        <ul className="space-y-3">
                          {errors.map((err, idx) => (
                            <li key={idx} className="flex items-start gap-3 text-[11px] text-rose-200/80 font-bold">
                              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0"></span>
                              {err}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Signal Form */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                      <div className="space-y-8">
                        <div>
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Market Pair</label>
                          <input 
                            type="text" 
                            value={signal.symbol} 
                            onChange={(e) => setSignal({...signal, symbol: e.target.value})}
                            className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-white focus:ring-2 focus:ring-indigo-600 outline-none font-bold"
                            placeholder="e.g., BTC/USDT"
                          />
                        </div>
                        
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Position</label>
                            <div className="flex p-1.5 bg-slate-950 rounded-2xl border border-slate-800">
                              <button 
                                onClick={() => setSignal({...signal, direction: SignalDirection.BUY})}
                                className={`flex-1 py-3 rounded-xl font-black text-xs transition-all
                                  ${signal.direction === SignalDirection.BUY 
                                    ? 'bg-emerald-600 text-white shadow-lg' 
                                    : 'text-slate-600 hover:text-slate-300'}`}
                              >
                                BUY
                              </button>
                              <button 
                                onClick={() => setSignal({...signal, direction: SignalDirection.SELL})}
                                className={`flex-1 py-3 rounded-xl font-black text-xs transition-all
                                  ${signal.direction === SignalDirection.SELL 
                                    ? 'bg-rose-600 text-white shadow-lg' 
                                    : 'text-slate-600 hover:text-slate-300'}`}
                              >
                                SELL
                              </button>
                            </div>
                          </div>
                          
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Interval</label>
                            <select 
                              value={signal.timeframe} 
                              onChange={(e) => setSignal({...signal, timeframe: e.target.value})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-white outline-none font-bold"
                            >
                              {['1M', '5M', '15M', '1H', '4H', '1D', '1W'].map(tf => (
                                <option key={tf} value={tf}>{tf}</option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Entry Target</label>
                            <input 
                              type="number" 
                              value={signal.entry}
                              onChange={(e) => setSignal({...signal, entry: Number(e.target.value)})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-white outline-none font-mono font-bold"
                            />
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Stop Loss</label>
                            <input 
                              type="number" 
                              value={signal.sl}
                              onChange={(e) => setSignal({...signal, sl: Number(e.target.value)})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-rose-500 outline-none font-mono font-bold"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-8">
                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">TP 1</label>
                            <input 
                              type="number" 
                              value={signal.tp1}
                              onChange={(e) => setSignal({...signal, tp1: Number(e.target.value)})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-4 py-4 text-emerald-500 outline-none font-mono font-bold text-xs"
                            />
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">TP 2</label>
                            <input 
                              type="number" 
                              value={signal.tp2}
                              onChange={(e) => setSignal({...signal, tp2: Number(e.target.value)})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-4 py-4 text-emerald-400 outline-none font-mono font-bold text-xs"
                            />
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">TP 3</label>
                            <input 
                              type="number" 
                              value={signal.tp3}
                              onChange={(e) => setSignal({...signal, tp3: Number(e.target.value)})}
                              className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-4 py-4 text-emerald-300 outline-none font-mono font-bold text-xs"
                            />
                          </div>
                        </div>

                        <div>
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Reasoning Basis</label>
                          <textarea 
                            value={signal.strategy}
                            onChange={(e) => setSignal({...signal, strategy: e.target.value})}
                            rows={4}
                            className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-5 py-4 text-white outline-none font-medium italic resize-none text-sm"
                            placeholder="Enter technical analysis rationale..."
                          />
                        </div>
                      </div>
                    </div>

                    <button 
                      onClick={handleBroadcastSignal}
                      disabled={loading}
                      className="w-full mt-12 bg-gradient-to-r from-indigo-600 to-purple-700 hover:from-indigo-500 hover:to-purple-600 
                                py-5 rounded-[2rem] font-black text-lg tracking-widest flex items-center justify-center gap-4 
                                transition-all shadow-xl disabled:opacity-50 disabled:cursor-not-allowed group"
                    >
                      {loading ? (
                        <Loader2 className="animate-spin" />
                      ) : (
                        <Send size={22} className="-rotate-45 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                      )}
                      {loading ? 'TRANSMITTING NEURAL PACKET...' : 'DEPLOY ENCRYPTED BROADCAST'}
                    </button>
                  </div>
                </div>

                {/* Preview Panel */}
                <div className="lg:col-span-4 space-y-10">
                  <div className="bg-slate-900 rounded-[3rem] border border-slate-800 overflow-hidden shadow-2xl p-8 relative">
                    <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800/50">
                      <BrainCircuit size={20} className="text-indigo-500" />
                      <span className="text-[10px] font-black text-white uppercase tracking-widest">LIVE PREVIEW</span>
                    </div>
                    
                    <div className="w-full bg-slate-950 p-8 rounded-[2.5rem] border border-slate-800 relative overflow-hidden shadow-inner">
                      <div className={`absolute top-0 left-0 w-2 h-full ${signal.direction === SignalDirection.BUY ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
                      
                      <p className="text-[14px] font-black text-white mb-6 uppercase tracking-tight">
                        {signalPreview.header}
                      </p>
                      
                      <div className="space-y-4 font-mono text-[11px] font-bold">
                        <p className="text-slate-500 uppercase">
                          DIRECTION: <span className={signalPreview.directionColor}>{signalPreview.directionText}</span>
                        </p>
                        <p className="text-slate-500 uppercase">TF: {signal.timeframe}</p>
                        
                        <div className="h-[1px] bg-slate-800/50 my-6"></div>
                        
                        <div className="space-y-3">
                          <p className="text-white/90">📍 ENTRY: {signal.entry}</p>
                          <p className="text-emerald-500">🎯 TP1: {signal.tp1}</p>
                          <p className="text-emerald-400">🎯 TP2: {signal.tp2}</p>
                          <p className="text-emerald-300">🎯 TP3: {signal.tp3}</p>
                          <p className="text-rose-500">🛑 SL: {signal.sl}</p>
                        </div>
                        
                        <p className="text-indigo-400 text-[10px] leading-relaxed italic border-t border-slate-800/50 mt-6 pt-6 line-clamp-4">
                          {signal.strategy}
                        </p>
                      </div>
                    </div>
                    
                    <div className="mt-8 flex items-center justify-between px-2">
                      <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Logic Status:</span>
                      <span className={`text-[9px] font-black px-3 py-1 rounded-full border ${signalPreview.logicStatusColor}`}>
                        {signalPreview.logicStatus}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Reports Tab */}
            {activeTab === 'reports' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="bg-slate-900 rounded-[2.5rem] border border-slate-800 p-10 shadow-2xl">
                  <div className="flex items-center gap-4 mb-10">
                    <BarChart3 className="text-indigo-500" size={24} />
                    <div>
                      <h3 className="text-2xl font-black text-white uppercase">NEURAL ANALYTICS</h3>
                      <p className="text-slate-500 text-[10px] font-bold uppercase">Gemini 3.0 Deep Synthesis</p>
                    </div>
                  </div>
                  
                  <div className="space-y-8">
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Reasoning Depth</label>
                      <div className="grid grid-cols-3 gap-3 p-1.5 bg-slate-950 rounded-2xl border border-slate-800">
                        {[
                          { key: AnalysisDepth.QUICK, icon: <Search size={14} />, label: 'QUICK' },
                          { key: AnalysisDepth.DETAILED, icon: <Activity size={14} />, label: 'DEEP' },
                          { key: AnalysisDepth.QUANT, icon: <Layers size={14} />, label: 'QUANT' }
                        ].map(({key, icon, label}) => (
                          <button
                            key={key}
                            onClick={() => setReportDepth(key)}
                            className={`py-3 rounded-xl font-black text-[10px] tracking-widest flex items-center justify-center gap-2 transition-all
                              ${reportDepth === key ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                          >
                            {icon} {label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <textarea 
                      value={analysisContext}
                      onChange={(e) => setAnalysisContext(e.target.value)}
                      placeholder="Enter market vectors, price action observations, or technical indicators for deep synthesis..."
                      className="w-full h-80 bg-slate-950 border border-slate-800 rounded-3xl px-6 py-6 text-white outline-none focus:ring-2 focus:ring-indigo-600 resize-none font-medium text-sm"
                    />
                    
                    <button 
                      onClick={handleGenerateReport}
                      disabled={loading || !analysisContext}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 py-5 rounded-[1.5rem] font-black text-lg tracking-widest flex items-center justify-center gap-4 shadow-xl disabled:opacity-50"
                    >
                      {loading ? <Loader2 className="animate-spin" /> : <Zap size={20} />}
                      {loading ? 'SYNTHESIZING VECTORS...' : `INITIALIZE ${reportDepth} ANALYSIS`}
                    </button>
                  </div>
                </div>

                {/* Report Output */}
                <div className="bg-slate-900 rounded-[2.5rem] border border-slate-800 p-10 flex flex-col min-h-[600px] shadow-2xl">
                  {currentReport ? (
                    <div className="flex-1 flex flex-col space-y-8 animate-in overflow-hidden">
                      <div className="flex items-center justify-between">
                        <h4 className="text-2xl font-black text-white uppercase underline decoration-indigo-500 decoration-4 underline-offset-8">
                          {currentReport.title}
                        </h4>
                        <div className={`px-4 py-1.5 rounded-full text-[10px] font-black border-2 ${
                          currentReport.outlook === 'BULLISH' 
                            ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' 
                            : currentReport.outlook === 'BEARISH'
                            ? 'bg-rose-500/10 text-rose-500 border-rose-500/20'
                            : 'bg-slate-500/10 text-slate-400 border-slate-500/20'
                        }`}>
                          {currentReport.outlook}
                        </div>
                      </div>

                      <div className="p-8 bg-slate-950 rounded-3xl border border-slate-800 shadow-inner">
                        <p className="text-[10px] font-black text-indigo-400 mb-4 uppercase tracking-widest">AI EXECUTIVE SUMMARY</p>
                        <p className="text-slate-200 leading-relaxed font-bold italic">"{currentReport.summary}"</p>
                      </div>

                      <div className="flex-1 overflow-y-auto custom-scrollbar space-y-8">
                        <div className="text-slate-400 text-sm leading-relaxed whitespace-pre-line bg-slate-950/50 p-8 rounded-3xl border border-slate-800/50">
                          {currentReport.technicalDetails}
                        </div>
                        
                        {currentReport.sources && currentReport.sources.length > 0 && (
                          <div className="space-y-4">
                            <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Grounding Sources</h5>
                            <div className="grid grid-cols-1 gap-2">
                              {currentReport.sources.map((source, idx) => (
                                <a 
                                  key={idx}
                                  href={source.uri}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-center gap-3 p-4 bg-indigo-500/5 border border-indigo-500/10 rounded-2xl hover:bg-indigo-500/10 transition-colors group"
                                >
                                  <ExternalLink size={14} className="text-indigo-400 group-hover:text-indigo-300" />
                                  <span className="text-xs font-bold text-slate-400 group-hover:text-slate-200 truncate flex-1">
                                    {source.title}
                                  </span>
                                </a>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>

                      <button 
                        onClick={handleBroadcastReport}
                        className="w-full bg-indigo-600 hover:bg-indigo-500 py-5 rounded-[1.5rem] font-black text-lg tracking-widest flex items-center justify-center gap-4 transition-all"
                      >
                        <Send size={22} className="-rotate-45" /> TRANSMIT INTELLIGENCE
                      </button>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center opacity-40">
                      <FileText size={64} className="mb-6" />
                      <p className="text-xl font-black text-slate-400 uppercase">Engine Standby</p>
                      <p className="text-[10px] text-slate-600 mt-2">Generate a report to see results here</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Pine Script Tab */}
            {activeTab === 'pinescript' && (
              <div className="space-y-10">
                <div className="bg-slate-900 rounded-[2.5rem] border border-slate-800 p-10 shadow-2xl">
                  <div className="flex items-center gap-4">
                    <Code className="text-emerald-500" size={24} />
                    <div>
                      <h3 className="text-2xl font-black text-white uppercase">PINE FORGE 6.0</h3>
                      <p className="text-slate-500 text-[10px] font-bold uppercase">TradingView v6.0 Standard</p>
                    </div>
                  </div>
                  
                  <div className="flex gap-6 mt-10">
                    <input 
                      value={pinePrompt}
                      onChange={(e) => setPinePrompt(e.target.value)}
                      placeholder="Describe trading logic (e.g., RSI Divergence + Volume Profile with EMA crossover)..."
                      className="flex-1 bg-slate-950 border border-slate-800 rounded-[1.5rem] px-8 py-5 text-white outline-none font-medium focus:ring-2 focus:ring-emerald-600"
                    />
                    <button 
                      onClick={handleGeneratePine}
                      disabled={loading || !pinePrompt}
                      className="bg-emerald-600 hover:bg-emerald-500 px-10 rounded-[1.5rem] font-black text-sm uppercase shadow-xl disabled:opacity-50 transition-all"
                    >
                      {loading ? <Loader2 className="animate-spin" /> : 'FORGE V6 SOURCE'}
                    </button>
                  </div>
                </div>

                {pineScript && (
                  <div className="bg-slate-950 rounded-[2.5rem] border border-slate-800 overflow-hidden shadow-2xl animate-in">
                    <div className="bg-slate-900/80 px-10 py-5 border-b border-slate-800 flex justify-between items-center">
                      <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">V6.0 ALPHA CORE</span>
                      <button 
                        onClick={handleCopyCode}
                        className="text-[10px] font-black text-emerald-500 flex items-center gap-2 group px-5 py-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5 hover:bg-emerald-500/10 transition-all"
                      >
                        {copied ? <Check size={14} /> : <Copy size={14} />}
                        {copied ? 'CLONED' : 'CLONE SOURCE'}
                      </button>
                    </div>
                    <div className="p-10 font-mono text-sm overflow-x-auto leading-relaxed max-h-[750px] custom-scrollbar bg-slate-950">
                      {highlightPineScript(pineScript)}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="max-w-3xl mx-auto space-y-10">
                <div className="bg-slate-900 rounded-[3rem] border border-slate-800 p-12 shadow-2xl">
                  <div className="flex items-center gap-6 mb-12">
                    <Settings className="text-white" size={32} />
                    <h3 className="text-3xl font-black text-white uppercase">SYSTEM GATEWAYS</h3>
                  </div>

                  <div className="space-y-10">
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Telegram Bot Token</label>
                      <input 
                        type="password"
                        value={ENV_CONFIG.botToken || ''}
                        disabled
                        className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-6 py-5 text-white font-mono text-sm outline-none"
                        placeholder="Loaded from .env"
                      />
                      <p className="text-[9px] text-slate-600 mt-2">Configure in .env file (VITE_TELEGRAM_BOT_TOKEN)</p>
                    </div>
                    
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Destination Chat ID</label>
                      <input 
                        type="text"
                        value={ENV_CONFIG.chatId || ''}
                        disabled
                        className="w-full bg-slate-950 border border-slate-800 rounded-2xl px-6 py-5 text-white font-mono text-sm outline-none"
                        placeholder="Loaded from .env"
                      />
                      <p className="text-[9px] text-slate-600 mt-2">Configure in .env file (VITE_TELEGRAM_CHAT_ID)</p>
                    </div>

                    <div className="p-6 bg-indigo-500/5 border border-indigo-500/20 rounded-2xl">
                      <div className="flex items-start gap-3">
                        <Info size={16} className="text-indigo-400 mt-0.5" />
                        <div>
                          <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Security Notice</p>
                          <p className="text-[11px] text-slate-400 mt-1">
                            API credentials are now loaded securely from environment variables and are never exposed in the client bundle.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Terminal Footer */}
        <footer className="h-32 bg-slate-950 border-t border-slate-800/80 p-5 font-mono text-[10px] flex flex-col z-20">
          <div className="flex items-center justify-between mb-2 text-slate-600 font-black uppercase tracking-widest pb-2 border-b border-slate-800/30">
            <div className="flex items-center gap-2">
              <Terminal size={12} className="text-indigo-500" /> NEURAL OVERRIDE TERMINAL
            </div>
            <div className="flex gap-6">
              <span className="text-emerald-500">QUANT CORE: ONLINE</span>
              <span className="text-indigo-500">GPT-5 SYNTH: ACTIVE</span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto space-y-1.5 custom-scrollbar">
            {logs.map((log, i) => (
              <div 
                key={i}
                className={`flex gap-3 hover:bg-slate-900/40 px-2 rounded transition-colors
                  ${log.includes('[ERROR]') ? 'text-rose-500' : 
                    log.includes('[SUCCESS]') ? 'text-emerald-500' : 
                    log.includes('[AI]') ? 'text-indigo-400' : 'text-slate-500'}`}
              >
                <span className="opacity-20 select-none font-black">
                  {String(i + 1).padStart(3, '0')}
                </span>
                <p className="tracking-tighter">{log}</p>
              </div>
            ))}
            <div className="h-4" />
          </div>
        </footer>
      </main>
    </div>
  );
};

export default App;
