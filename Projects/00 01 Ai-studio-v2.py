
import React, { useState, useEffect, useRef } from 'react';
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
  Loader2,
  Terminal,
  History,
  Copy,
  Smartphone,
  BarChart3,
  Cpu,
  ExternalLink,
  ShieldAlert,
  Search,
  Activity,
  Layers,
  Check,
  ChevronDown,
  Info,
  BrainCircuit
} from 'lucide-react';
import { MarketChart } from './components/MarketChart';
import { TradingSignal, SignalDirection, AnalysisReport, TelegramConfig, AnalysisDepth } from './types';
import { generateTechnicalReport, generatePineScript } from './services/geminiService';
import { sendSignalToTelegram, sendReportToTelegram } from './services/telegramService';

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

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'reports' | 'pinescript' | 'settings'>('dashboard');
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>(["[SYSTEM] Core Engine: Gemini 3.0 / Pine V6.0 Active", "[INFO] Neural reasoning pipelines established..."]);
  const [errors, setErrors] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const [telegramConfig, setTelegramConfig] = useState<TelegramConfig>({
    botToken: localStorage.getItem('tg_bot_token') || '',
    chatId: localStorage.getItem('tg_chat_id') || ''
  });

  const [signal, setSignal] = useState<TradingSignal>({
    id: '', symbol: 'BTC/USDT', direction: SignalDirection.BUY, entry: 64250,
    tp1: 65500, tp2: 67000, tp3: 69000, sl: 63000, timeframe: '1H',
    strategy: 'Neural Momentum + RSI 6.0', timestamp: new Date()
  });
  
  const [signalIntelMode, setSignalIntelMode] = useState<'NONE' | 'QUICK' | 'DEEP'>('NONE');
  const [intelPreview, setIntelPreview] = useState<string | null>(null);
  const [analysisContext, setAnalysisContext] = useState('');
  const [reportDepth, setReportDepth] = useState<AnalysisDepth>(AnalysisDepth.DETAILED);
  const [currentReport, setCurrentReport] = useState<AnalysisReport | null>(null);
  const [pineScript, setPineScript] = useState('');
  const [pinePrompt, setPinePrompt] = useState('');

  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem('tg_bot_token', telegramConfig.botToken || '');
    localStorage.setItem('tg_chat_id', telegramConfig.chatId || '');
  }, [telegramConfig]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (msg: string) => setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleCopyCode = () => {
    if (!pineScript) return;
    navigator.clipboard.writeText(pineScript);
    setCopied(true);
    addLog("[SYSTEM] Pine V6 Source successfully cloned to clipboard.");
    setTimeout(() => setCopied(false), 2000);
  };

  const validateSignal = (): boolean => {
    const errs: string[] = [];
    const { symbol, entry, tp1, tp2, tp3, sl, direction } = signal;

    if (!symbol.trim()) errs.push("Symbol is required.");
    if (entry <= 0) errs.push("Entry price must be greater than 0.");
    if (tp1 <= 0) errs.push("TP1 must be greater than 0.");
    if (tp2 <= 0) errs.push("TP2 must be greater than 0.");
    if (tp3 <= 0) errs.push("TP3 must be greater than 0.");
    if (sl <= 0) errs.push("Stop Loss must be greater than 0.");

    if (direction === SignalDirection.BUY) {
      if (sl >= entry) errs.push("BUY Logic: Stop Loss must be strictly lower than Entry price.");
      if (entry >= tp1) errs.push("BUY Logic: Take Profit 1 must be strictly higher than Entry price.");
      if (tp1 >= tp2) errs.push("BUY Logic: Take Profit 2 must be strictly higher than Take Profit 1.");
      if (tp2 >= tp3) errs.push("BUY Logic: Take Profit 3 must be strictly higher than Take Profit 2.");
    } else {
      if (sl <= entry) errs.push("SELL Logic: Stop Loss must be strictly higher than Entry price.");
      if (entry <= tp1) errs.push("SELL Logic: Take Profit 1 must be strictly lower than Entry price.");
      if (tp1 <= tp2) errs.push("SELL Logic: Take Profit 2 must be strictly lower than Take Profit 1.");
      if (tp2 <= tp3) errs.push("SELL Logic: Take Profit 3 must be strictly lower than Take Profit 2.");
    }

    setErrors(errs);
    if (errs.length > 0) {
      addLog(`[ERROR] Signal Validation Failed: ${errs.length} issues found.`);
    }
    return errs.length === 0;
  };

  const handleGenerateSignalIntel = async () => {
    if (!signal.symbol) return;
    try {
      setLoading(true);
      addLog(`[AI] Invoking Gemini 3 reasoning for ${signal.symbol}...`);
      const depth = signalIntelMode === 'QUICK' ? AnalysisDepth.QUICK : AnalysisDepth.DETAILED;
      const report = await generateTechnicalReport(signal.symbol, signal.timeframe, `Analyze: ${signal.direction} ${signal.symbol}. Strategy: ${signal.strategy}`, depth);
      setIntelPreview(report.summary);
      addLog(`[AI] Intelligence Layer 1 complete. Outlook: ${report.outlook}`);
    } catch (err: any) {
      addLog(`[ERROR] AI Synthesis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBroadcastSignal = async () => {
    if (!validateSignal()) return;
    
    if (!telegramConfig.botToken || !telegramConfig.chatId) {
      addLog("[ERROR] Broadcast failed: Missing Credentials.");
      setActiveTab('settings');
      return;
    }
    
    try {
      setLoading(true);
      addLog(`[ACTION] Deploying neural broadcast: ${signal.symbol}`);
      let finalSignal = { ...signal };
      if (signalIntelMode !== 'NONE') {
        const depth = signalIntelMode === 'QUICK' ? AnalysisDepth.QUICK : AnalysisDepth.DETAILED;
        const report = await generateTechnicalReport(signal.symbol, signal.timeframe, signal.strategy, depth);
        finalSignal.strategy = `${signal.strategy}\n\n🤖 GEMINI 3 INTEL:\n${report.summary}`;
      }
      await sendSignalToTelegram(telegramConfig, finalSignal);
      addLog("[SUCCESS] Broadcast delivered via Secure Telegram Gateway.");
      setErrors([]); // Clear errors on success
    } catch (err: any) {
      addLog(`[ERROR] Protocol Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!analysisContext) return;
    try {
      setLoading(true);
      const isQuant = reportDepth === AnalysisDepth.QUANT;
      addLog(`[AI] Initializing ${reportDepth} Neural ${isQuant ? 'Thinking' : 'Flash'} for ${signal.symbol}...`);
      const report = await generateTechnicalReport(signal.symbol, signal.timeframe, analysisContext, reportDepth);
      setCurrentReport(report);
      addLog(`[AI] Synthesis complete. Reasoning fidelity: High.`);
    } catch (err: any) {
      addLog(`[ERROR] Neural engine mismatch: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleBroadcastReport = async () => {
    if (!telegramConfig.botToken || !telegramConfig.chatId || !currentReport) {
      addLog("[ERROR] Broadcast failed: Missing Credentials or Report.");
      if (!telegramConfig.botToken || !telegramConfig.chatId) setActiveTab('settings');
      return;
    }
    try {
      setLoading(true);
      addLog(`[ACTION] Deploying neural intelligence broadcast: ${currentReport.title}`);
      await sendReportToTelegram(telegramConfig, currentReport);
      addLog("[SUCCESS] Intelligence synthesis delivered via Secure Telegram Gateway.");
    } catch (err: any) {
      addLog(`[ERROR] Protocol Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePine = async () => {
    if (!pinePrompt) return;
    try {
      setLoading(true);
      addLog("[AI] Compiling logic to Pine Script v6.0 Standard...");
      const code = await generatePineScript("Dark Singularity Engine", pinePrompt);
      setPineScript(code);
      addLog("[SUCCESS] Code validated against V6.0 specification.");
    } catch (err: any) {
      addLog(`[ERROR] Compilation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const timeframes = [
    { value: '1M', label: '1 MIN' }, { value: '5M', label: '5 MIN' }, { value: '15M', label: '15 MIN' },
    { value: '1H', label: '1 HOUR' }, { value: '4H', label: '4 HOUR' }, { value: '1D', label: '1 DAY' },
    { value: '1W', label: '1 WEEK' }
  ];

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#020617] text-slate-200">
      <nav className="w-full md:w-64 bg-[#0f172a] border-r border-slate-800 flex flex-col p-4 shadow-xl z-10">
        <div className="flex items-center gap-3 mb-10 px-2 py-4 border-b border-slate-800/50">
          <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2.5 rounded-xl shadow-lg shadow-indigo-500/20">
            <BrainCircuit className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="font-black text-xl tracking-tighter text-white">TRADECAST</h1>
            <p className="text-[10px] font-bold text-indigo-400 tracking-[0.2em] uppercase opacity-80 underline decoration-indigo-500/50">v3.0 NEURAL</p>
          </div>
        </div>

        <div className="space-y-1.5 flex-1">
          <button onClick={() => setActiveTab('dashboard')} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${activeTab === 'dashboard' ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/20 shadow-[0_0_20px_rgba(79,70,229,0.1)]' : 'hover:bg-slate-800/50 text-slate-400'}`}>
            <LayoutDashboard size={18} />
            <span className="font-bold text-sm">Signal Center</span>
          </button>
          <button onClick={() => setActiveTab('reports')} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${activeTab === 'reports' ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/20' : 'hover:bg-slate-800/50 text-slate-400'}`}>
            <BarChart3 size={18} />
            <span className="font-bold text-sm">AI Reports Lab</span>
          </button>
          <button onClick={() => setActiveTab('pinescript')} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${activeTab === 'pinescript' ? 'bg-emerald-600/10 text-emerald-400 border border-emerald-500/20' : 'hover:bg-slate-800/50 text-slate-400'}`}>
            <Code size={18} />
            <span className="font-bold text-sm">Script Forge</span>
          </button>
          <button onClick={() => setActiveTab('settings')} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${activeTab === 'settings' ? 'bg-slate-800 text-white' : 'hover:bg-slate-800/50 text-slate-400'}`}>
            <Settings size={18} />
            <span className="font-bold text-sm">Gateways</span>
          </button>
        </div>

        <div className="mt-auto pt-6 border-t border-slate-800/50">
          <div className="p-4 bg-[#1e293b] rounded-2xl border border-slate-700/50">
            <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1">Engine Latency</p>
            <div className="flex items-center gap-2">
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="w-[85%] h-full bg-emerald-500"></div>
              </div>
              <span className="text-[9px] font-bold text-emerald-500">OPTIMAL</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="bg-[#020617]/50 backdrop-blur-md p-6 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="h-10 w-1 bg-indigo-600 rounded-full"></div>
            <div>
              <h2 className="text-2xl font-black text-white tracking-tight uppercase">
                {activeTab === 'dashboard' ? 'Signal Intelligence' : activeTab === 'reports' ? 'Advanced Analytics' : activeTab === 'pinescript' ? 'Source Compilation' : 'Gateway Systems'}
              </h2>
              <p className="text-slate-500 text-[10px] font-bold tracking-[0.3em] uppercase underline decoration-slate-800 decoration-2">Gemini 3.0 / GPT 5.2 Reasoning Protocol</p>
            </div>
          </div>
          <div className="flex gap-3">
             <div className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-xl flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">Search Grounding: ON</span>
             </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 lg:p-10 custom-scrollbar">
          <div className="max-w-7xl mx-auto space-y-10">
            
            {activeTab === 'dashboard' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                <div className="lg:col-span-8 space-y-10">
                  <MarketChart symbol={signal.symbol} timeframe={signal.timeframe} />
                  <div className="bg-[#0f172a] rounded-[2.5rem] border border-slate-800 p-8 shadow-2xl relative overflow-hidden">
                    <div className="flex items-center justify-between mb-10">
                       <h3 className="text-xl font-black text-white flex items-center gap-4">
                         <TrendingUp className="text-indigo-500" size={20} />
                         SIGNAL ARCHITECT
                       </h3>
                    </div>

                    {/* Logical Error Display */}
                    {errors.length > 0 && (
                      <div className="mb-8 p-6 bg-red-950/40 border-2 border-red-500/50 rounded-[1.5rem] animate-in zoom-in-95 duration-300">
                        <div className="flex items-center gap-3 mb-4 text-red-400">
                          <ShieldAlert size={20} strokeWidth={3} />
                          <h4 className="font-black text-xs uppercase tracking-widest">Protocol Validation Failure</h4>
                        </div>
                        <ul className="space-y-2">
                          {errors.map((err, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-xs text-red-400/90 font-bold">
                              <span className="mt-1 w-1 h-1 rounded-full bg-red-500 shrink-0"></span>
                              {err}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                      <div className="space-y-8">
                        <div>
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Market Pair</label>
                          <input type="text" value={signal.symbol} onChange={(e) => setSignal({...signal, symbol: e.target.value})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-white focus:ring-2 focus:ring-indigo-600 outline-none font-bold" />
                        </div>
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Position</label>
                            <div className="flex p-1.5 bg-[#020617] rounded-2xl border border-slate-800">
                              <button onClick={() => setSignal({...signal, direction: SignalDirection.BUY})} className={`flex-1 py-3 rounded-xl font-black text-xs ${signal.direction === SignalDirection.BUY ? 'bg-green-600 text-white' : 'text-slate-600'}`}>BUY</button>
                              <button onClick={() => setSignal({...signal, direction: SignalDirection.SELL})} className={`flex-1 py-3 rounded-xl font-black text-xs ${signal.direction === SignalDirection.SELL ? 'bg-red-600 text-white' : 'text-slate-600'}`}>SELL</button>
                            </div>
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Interval</label>
                            <select value={signal.timeframe} onChange={(e) => setSignal({...signal, timeframe: e.target.value})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-white outline-none focus:ring-2 focus:ring-indigo-600 font-bold appearance-none">
                              {timeframes.map(tf => <option key={tf.value} value={tf.value}>{tf.label}</option>)}
                            </select>
                          </div>
                        </div>
                        <div>
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Market Intel Layer</label>
                          <div className="grid grid-cols-3 gap-2">
                            {['NONE', 'QUICK', 'DEEP'].map(mode => (
                              <button key={mode} onClick={() => { setSignalIntelMode(mode as any); setIntelPreview(null); }} className={`py-3 rounded-xl border font-black text-[10px] tracking-widest transition-all ${signalIntelMode === mode ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg' : 'bg-[#020617] border-slate-800 text-slate-500'}`}>{mode}</button>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="space-y-8">
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Target 1</label>
                            <input type="number" value={signal.tp1} onChange={(e) => setSignal({...signal, tp1: Number(e.target.value)})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-green-500 outline-none font-mono font-bold" />
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Stop Loss</label>
                            <input type="number" value={signal.sl} onChange={(e) => setSignal({...signal, sl: Number(e.target.value)})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-red-500 outline-none font-mono font-bold" />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-6">
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Target 2</label>
                            <input type="number" value={signal.tp2} onChange={(e) => setSignal({...signal, tp2: Number(e.target.value)})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-green-400 outline-none font-mono font-bold" />
                          </div>
                          <div>
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Target 3</label>
                            <input type="number" value={signal.tp3} onChange={(e) => setSignal({...signal, tp3: Number(e.target.value)})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-green-300 outline-none font-mono font-bold" />
                          </div>
                        </div>
                        <div>
                          <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 block">Reasoning Basis</label>
                          <textarea value={signal.strategy} onChange={(e) => setSignal({...signal, strategy: e.target.value})} rows={3} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-5 py-4 text-white outline-none focus:ring-2 focus:ring-indigo-600 font-medium italic resize-none text-sm" placeholder="Paste structural data..." />
                        </div>
                        {signalIntelMode !== 'NONE' && (
                          <button onClick={handleGenerateSignalIntel} disabled={loading} className="w-full py-3 bg-slate-900 border border-slate-800 rounded-xl text-[10px] font-black text-indigo-400 flex items-center justify-center gap-2">
                            {loading ? <Loader2 size={12} className="animate-spin" /> : <Zap size={12} />}
                            {intelPreview ? 'REFRESH NEURAL PREVIEW' : 'SYNTHESIZE MARKET INTEL'}
                          </button>
                        )}
                      </div>
                    </div>

                    <button onClick={handleBroadcastSignal} disabled={loading} className="w-full mt-12 bg-gradient-to-r from-indigo-600 to-purple-700 hover:from-indigo-500 hover:to-purple-600 py-5 rounded-[1.5rem] font-black text-lg tracking-widest flex items-center justify-center gap-4 transition-all shadow-[0_20px_40px_-10px_rgba(79,70,229,0.4)]">
                      {loading ? <Loader2 className="animate-spin" /> : <Send size={22} className="-rotate-45" />}
                      {loading ? 'TRANSMITTING NEURAL PACKET...' : 'DEPLOY ENCRYPTED BROADCAST'}
                    </button>
                  </div>
                </div>

                <div className="lg:col-span-4 space-y-10">
                  <div className="bg-[#0f172a] rounded-[2.5rem] border border-slate-800 overflow-hidden shadow-2xl p-8">
                     <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800/50">
                        <Smartphone size={20} className="text-indigo-500" />
                        <span className="text-[10px] font-black text-white uppercase tracking-widest">GATEWAY PREVIEW</span>
                      </div>
                      <div className="w-full bg-[#020617] p-6 rounded-[2rem] border border-slate-800 relative overflow-hidden shadow-inner">
                        <div className={`absolute top-0 left-0 w-1.5 h-full ${signal.direction === SignalDirection.BUY ? 'bg-emerald-500 shadow-[4px_0_15px_rgba(16,185,129,0.3)]' : 'bg-red-500 shadow-[4px_0_15px_rgba(239,68,68,0.3)]'}`}></div>
                        <p className="text-[13px] font-black text-white mb-4 uppercase">🚀 SIGNAL: {signal.symbol} 🚀</p>
                        <div className="space-y-3 font-mono text-[11px] font-bold">
                          <p className="text-slate-500">DIRECTION: <span className={signal.direction === SignalDirection.BUY ? 'text-emerald-400' : 'text-red-400'}>{signal.direction === SignalDirection.BUY ? '🟢 BUY' : '🔴 SELL'}</span></p>
                          <p className="text-slate-500">TF: <span className="text-white">{signal.timeframe}</span></p>
                          <div className="h-[1px] bg-slate-800/50 my-4"></div>
                          <div className="grid grid-cols-1 gap-2">
                             <p className="text-white">📍 ENTRY: {signal.entry}</p>
                             <p className="text-emerald-500">🎯 TP1: {signal.tp1}</p>
                             <p className="text-red-500">🛑 SL: {signal.sl}</p>
                          </div>
                          <p className="text-indigo-400 text-[10px] leading-relaxed italic border-t border-slate-800/50 mt-4 pt-4">{signal.strategy}</p>
                        </div>
                      </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'reports' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <div className="bg-[#0f172a] rounded-[2.5rem] border border-slate-800 p-10 shadow-2xl relative overflow-hidden">
                  <div className="flex items-center gap-4 mb-10">
                    <BarChart3 className="text-indigo-500" size={24} />
                    <div>
                       <h3 className="text-2xl font-black text-white uppercase">NEURAL ANALYTICS</h3>
                       <p className="text-slate-500 text-[10px] font-bold tracking-widest uppercase">Gemini 3.0 Deep Synthesis</p>
                    </div>
                  </div>
                  
                  <div className="space-y-8">
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Reasoning Depth</label>
                      <div className="grid grid-cols-3 gap-3 p-1.5 bg-[#020617] rounded-2xl border border-slate-800">
                         {[{key: AnalysisDepth.QUICK, icon: <Search size={14} />}, {key: AnalysisDepth.DETAILED, icon: <Activity size={14} />}, {key: AnalysisDepth.QUANT, icon: <Layers size={14} />}].map(({key, icon}) => (
                           <button key={key} onClick={() => setReportDepth(key)} className={`py-3 rounded-xl font-black text-[10px] tracking-widest flex items-center justify-center gap-2 transition-all ${reportDepth === key ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' : 'text-slate-500'}`}>
                             {icon} {key}
                           </button>
                         ))}
                      </div>
                    </div>

                    <textarea value={analysisContext} onChange={(e) => setAnalysisContext(e.target.value)} placeholder="Provide market vectors for deep synthesis..." className="w-full h-80 bg-[#020617] border border-slate-800 rounded-3xl px-6 py-6 text-white outline-none focus:ring-2 focus:ring-indigo-600 resize-none font-medium text-sm" />
                    
                    <button onClick={handleGenerateReport} disabled={loading || !analysisContext} className="w-full bg-indigo-600 hover:bg-indigo-500 py-5 rounded-[1.5rem] font-black text-lg tracking-widest flex items-center justify-center gap-4 shadow-xl">
                      {loading ? <Loader2 className="animate-spin" /> : <Zap size={20} />}
                      {loading ? 'SYNTHESIZING VECTORS...' : `INITIALIZE ${reportDepth} ANALYSIS`}
                    </button>
                  </div>
                </div>

                <div className="bg-[#0f172a] rounded-[2.5rem] border border-slate-800 p-10 flex flex-col min-h-[600px] shadow-2xl">
                  {currentReport ? (
                    <div className="flex-1 flex flex-col space-y-8 animate-in fade-in zoom-in-95">
                      <div className="flex items-center justify-between">
                        <h4 className="text-2xl font-black text-white uppercase underline decoration-indigo-500 decoration-4 underline-offset-8">{currentReport.title}</h4>
                        <div className={`px-4 py-1.5 rounded-full text-[10px] font-black border-2 ${currentReport.outlook === 'BULLISH' ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' : 'bg-red-500/10 text-red-500 border-red-500/20'}`}>{currentReport.outlook}</div>
                      </div>
                      <div className="p-8 bg-[#020617] rounded-3xl border border-slate-800 shadow-inner">
                        <p className="text-[10px] font-black text-indigo-400 mb-4 uppercase">AI EXECUTIVE SUMMARY</p>
                        <p className="text-slate-200 leading-relaxed font-bold italic">"{currentReport.summary}"</p>
                      </div>
                      <div className="flex-1 overflow-y-auto custom-scrollbar space-y-8">
                        <div className="text-slate-400 text-sm leading-relaxed whitespace-pre-line bg-[#020617]/50 p-8 rounded-3xl border border-slate-800/50">{currentReport.technicalDetails}</div>
                      </div>
                      <button onClick={handleBroadcastReport} className="w-full bg-indigo-600 py-5 rounded-[1.5rem] font-black text-lg tracking-widest flex items-center justify-center gap-4">
                        <Send size={22} className="-rotate-45" /> TRANSMIT INTELLIGENCE
                      </button>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center opacity-40">
                      <FileText size={64} className="mb-6" />
                      <p className="text-xl font-black text-slate-400 uppercase">Engine Standby</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'pinescript' && (
              <div className="space-y-10">
                <div className="bg-[#0f172a] rounded-[2.5rem] border border-slate-800 p-10 shadow-2xl relative overflow-hidden">
                  <div className="flex items-center gap-4 mb-4">
                    <Code className="text-emerald-500" size={24} />
                    <div>
                      <h3 className="text-2xl font-black text-white uppercase">PINE FORGE 6.0</h3>
                      <p className="text-slate-500 text-[10px] font-black tracking-widest uppercase">TradingView v6.0 Standard Compilation</p>
                    </div>
                  </div>
                  <div className="flex gap-6 mt-10">
                    <input value={pinePrompt} onChange={(e) => setPinePrompt(e.target.value)} placeholder="Describe trading logic (e.g. RSI Divergence + Volume Profile)..." className="flex-1 bg-[#020617] border border-slate-800 rounded-[1.5rem] px-8 py-5 text-white outline-none font-medium" />
                    <button onClick={handleGeneratePine} disabled={loading || !pinePrompt} className="bg-emerald-600 px-10 rounded-[1.5rem] font-black text-sm uppercase shadow-xl">{loading ? <Loader2 className="animate-spin" /> : 'FORGE V6 SOURCE'}</button>
                  </div>
                </div>
                {pineScript && (
                  <div className="bg-[#020617] rounded-[2.5rem] border border-slate-800 overflow-hidden shadow-2xl animate-in slide-in-from-top-4">
                    <div className="bg-slate-900/80 px-10 py-5 border-b border-slate-800 flex justify-between items-center backdrop-blur-md">
                      <span className="text-[10px] font-black text-slate-500 tracking-[0.4em] uppercase">V6.0 ALPHA CORE</span>
                      <button onClick={handleCopyCode} className="text-[10px] font-black text-emerald-500 flex items-center gap-2 group px-5 py-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5">
                        {copied ? <Check size={14} /> : <Copy size={14} />} {copied ? 'CLONED' : 'CLONE SOURCE'}
                      </button>
                    </div>
                    <div className="p-10 font-mono text-sm overflow-x-auto leading-relaxed max-h-[750px] custom-scrollbar bg-[#020617]">{highlightPineScript(pineScript)}</div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'settings' && (
              <div className="max-w-3xl mx-auto space-y-10">
                <div className="bg-[#0f172a] rounded-[3rem] border border-slate-800 p-12 shadow-2xl relative">
                  <div className="flex items-center gap-6 mb-12">
                    <Settings className="text-white" size={32} />
                    <h3 className="text-3xl font-black text-white uppercase">SYSTEM GATEWAYS</h3>
                  </div>
                  <div className="space-y-10">
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Telegram API Key</label>
                      <input type="password" value={telegramConfig.botToken} onChange={(e) => setTelegramConfig({...telegramConfig, botToken: e.target.value})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-6 py-5 text-white font-mono text-sm outline-none" placeholder="0000000:AAxxxx..." />
                    </div>
                    <div>
                      <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 block">Destination Chat ID</label>
                      <input type="text" value={telegramConfig.chatId} onChange={(e) => setTelegramConfig({...telegramConfig, chatId: e.target.value})} className="w-full bg-[#020617] border border-slate-800 rounded-2xl px-6 py-5 text-white font-mono text-sm outline-none" placeholder="-100xxxx..." />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="h-32 bg-[#020617] border-t border-slate-800/80 p-5 font-mono text-[10px] flex flex-col z-20 overflow-hidden">
          <div className="flex items-center justify-between mb-2 text-slate-600 font-black uppercase tracking-widest pb-2 border-b border-slate-800/30">
            <div className="flex items-center gap-2"><Terminal size={12} className="text-indigo-500" /> NEURAL OVERRIDE TERMINAL</div>
            <div className="flex gap-6"><span className="text-emerald-500">QUANT CORE: ONLINE</span><span className="text-indigo-500">GPT-5 SYNTH: ACTIVE</span></div>
          </div>
          <div className="flex-1 overflow-y-auto space-y-1.5 custom-scrollbar">
            {logs.map((log, i) => (
              <div key={i} className={`flex gap-3 hover:bg-slate-900/40 px-2 rounded transition-colors ${log.includes('[ERROR]') ? 'text-red-500' : log.includes('[SUCCESS]') ? 'text-emerald-500' : log.includes('[AI]') ? 'text-indigo-400' : 'text-slate-500'}`}>
                <span className="opacity-20 select-none font-black">{String(i + 1).padStart(3, '0')}</span>
                <p className="tracking-tighter">{log}</p>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </footer>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 5px; height: 5px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #312e81; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #4338ca; }
        .animate-in { animation: fade-in 0.5s ease-out; }
        @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
};

export default App;
