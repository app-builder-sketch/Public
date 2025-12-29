import React, { useState, useEffect, useMemo } from 'react';
import { AssetClassMap, BroadcastConfig, DataPoint, FibLevels } from './types';
import { ASSET_CLASSES, TIMEFRAMES } from './constants';
import { generateData, getFundamentals } from './services/dataService';
import { calculateFibs, calculateVolumeProfile } from './services/quantService';
import { generateReport, REPORT_TYPES, ReportType } from './services/reportService';
import TickerMarquee from './components/TickerMarquee';
import LiveClock from './components/LiveClock';
import { PriceChart, EntropyChart, FluxChart, VolumeProfileChart } from './components/Charts';

function App() {
  // State
  const [selectedClass, setSelectedClass] = useState<string>("Crypto (Major)");
  const [selectedTicker, setSelectedTicker] = useState<string>("BTC-USD");
  const [timeframe, setTimeframe] = useState<string>("4h");
  const [data, setData] = useState<DataPoint[]>([]);
  const [fibs, setFibs] = useState<FibLevels | null>(null);
  const [activeTab, setActiveTab] = useState<string>("tech");
  
  // Broadcast & Report State
  const [broadcastMsg, setBroadcastMsg] = useState("");
  const [selectedReportType, setSelectedReportType] = useState<ReportType>("Quick Signal");
  const [broadcastLog, setBroadcastLog] = useState<BroadcastConfig[]>([]);

  // Mobile Toggle
  const [isMobile, setIsMobile] = useState(false);

  // Load Data
  useEffect(() => {
    const newData = generateData(selectedTicker, timeframe);
    setData(newData);
    setFibs(calculateFibs(newData));
  }, [selectedTicker, timeframe]);

  // Derived Metrics
  const last = data.length > 0 ? data[data.length - 1] : null;
  const fundamentals = useMemo(() => getFundamentals(selectedTicker), [selectedTicker]);
  
  // Volume Profile
  const { profile, poc } = useMemo(() => calculateVolumeProfile(data), [data]);

  // Handlers
  const handleGenerateReport = () => {
    if (!last || !fibs) return;
    const report = generateReport(selectedReportType, selectedTicker, data, fibs, fundamentals);
    setBroadcastMsg(report);
  };

  const handleCopy = () => {
    if(!broadcastMsg) return;
    navigator.clipboard.writeText(broadcastMsg);
    // Simple visual feedback could be added here
  }

  const handleBroadcast = () => {
    if (!broadcastMsg) return;
    const newBroadcast: BroadcastConfig = {
      name: `${selectedTicker} ${selectedReportType}`,
      message: broadcastMsg,
      scheduleTime: new Date().toLocaleTimeString(),
      status: 'sent'
    };
    setBroadcastLog([newBroadcast, ...broadcastLog]);
    setBroadcastMsg(""); 
  };

  // Determine Signal Bias for UI
  const signalBias = useMemo(() => {
      if(!last) return "NEUTRAL";
      if(last.nexus_signal === 1) return "LONG";
      if(last.nexus_signal === -1) return "SHORT";
      return "WAIT";
  }, [last]);

  // Confidence Score for Meter
  const confidenceScore = useMemo(() => {
    if(!last) return 0;
    let score = 0;
    const isBull = (last.gm_apex_base || 0) < last.close;
    if (last.nexus_signal === (isBull ? 1 : -1)) score += 50;
    if (Math.abs(last.vector_flux || 0) > 0.5) score += 25;
    if ((last.nexus_trend === 1 && isBull) || (last.nexus_trend === -1 && !isBull)) score += 25;
    return score;
  }, [last]);

  return (
    <div className={`min-h-screen ${isMobile ? 'p-2' : 'p-0'} text-gray-300 font-sans`}>
      {/* Sidebar (Desktop) / Drawer (Mobile) would go here. Using a simplified layout for SPA */}
      <div className="flex flex-col md:flex-row min-h-screen">
        
        {/* SIDEBAR */}
        <div className="w-full md:w-64 bg-[#080808] border-r border-[#222] p-4 flex flex-col gap-6">
          <div className="mb-4">
            <h1 className="text-2xl font-bold text-white tracking-tighter flex items-center gap-2">
              <span className="text-[#00F0FF]">💠</span> AXIOM
            </h1>
            <p className="text-xs text-gray-500 tracking-widest ml-8">TITAN EDITION</p>
          </div>

          <div className="flex items-center gap-2 mb-2">
            <label className="text-xs font-bold text-[#00F0FF]">MOBILE OPTIMIZED</label>
            <input 
              type="checkbox" 
              checked={isMobile} 
              onChange={(e) => setIsMobile(e.target.checked)}
              className="accent-[#00F0FF]"
            />
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-500 uppercase block mb-1">Sector</label>
              <select 
                value={selectedClass} 
                onChange={(e) => { setSelectedClass(e.target.value); setSelectedTicker(ASSET_CLASSES[e.target.value][0]); }}
                className="w-full bg-[#111] border border-[#333] text-[#00F0FF] p-2 rounded text-sm focus:border-[#00F0FF] outline-none"
              >
                {Object.keys(ASSET_CLASSES).map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 uppercase block mb-1">Ticker</label>
              <select 
                value={selectedTicker} 
                onChange={(e) => setSelectedTicker(e.target.value)}
                className="w-full bg-[#111] border border-[#333] text-[#00F0FF] p-2 rounded text-sm focus:border-[#00F0FF] outline-none"
              >
                {ASSET_CLASSES[selectedClass].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 uppercase block mb-1">Interval</label>
              <select 
                value={timeframe} 
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full bg-[#111] border border-[#333] text-[#00F0FF] p-2 rounded text-sm focus:border-[#00F0FF] outline-none"
              >
                {TIMEFRAMES.map(tf => <option key={tf} value={tf}>{tf}</option>)}
              </select>
            </div>
          </div>

          <div className="mt-auto border-t border-[#222] pt-4">
            <div className="text-xs text-gray-500 mb-2">API Credentials</div>
            <input type="password" placeholder="OpenAI Key" className="w-full bg-[#050505] border border-[#333] p-2 mb-2 text-xs rounded" />
            <input type="password" placeholder="Telegram Token" className="w-full bg-[#050505] border border-[#333] p-2 mb-2 text-xs rounded" />
            <input type="text" placeholder="Chat ID" className="w-full bg-[#050505] border border-[#333] p-2 text-xs rounded" />
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div className="flex-1 bg-[#050505] flex flex-col h-screen overflow-y-auto">
          <TickerMarquee />
          
          <div className="p-4 md:p-6 flex-1">
            <LiveClock />

            {/* METRICS */}
            <div className={`grid ${isMobile ? 'grid-cols-1' : 'grid-cols-4'} gap-4 mb-6`}>
              <MetricCard 
                label="Nexus Signal" 
                value={last?.nexus_signal === 1 ? "BUY" : last?.nexus_signal === -1 ? "SELL" : "WAIT"} 
                sub="TRINITY SYSTEM" 
                color={last?.nexus_signal === 1 ? "text-[#00E676]" : last?.nexus_signal === -1 ? "text-[#FF1744]" : "text-gray-400"} 
              />
              <MetricCard 
                label="Vector State" 
                value={last?.vector_state || "Neutral"} 
                sub={`Flux: ${last?.vector_flux?.toFixed(2)}`} 
                color="text-[#00F0FF]" 
              />
              <MetricCard 
                label="Entropy (CHEDO)" 
                value={last?.chedo?.toFixed(2) || "0.00"} 
                sub={Math.abs(last?.chedo || 0) > 0.7 ? "RISK" : "STABLE"} 
                color="text-[#D500F9]" 
              />
              <MetricCard 
                label="Risk Line" 
                value={`$${last?.nexus_risk?.toFixed(2) || "0.00"}`}
                sub="UT BOT TRAIL" 
                color={ (last?.close || 0) > (last?.nexus_risk || 0) ? "text-[#00E676]" : "text-[#FF1744]"} 
              />
            </div>

            {/* TABS */}
            <div className="mb-6 border-b border-[#222] flex gap-4 overflow-x-auto">
              {["tech", "macro", "ai", "broadcast"].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`pb-2 text-sm font-mono uppercase tracking-widest ${activeTab === tab ? 'text-white border-b-2 border-[#00F0FF]' : 'text-gray-600 hover:text-gray-400'}`}
                >
                  {tab === 'tech' ? 'Titan Tech' : tab === 'ai' ? 'Intelligence' : tab === 'broadcast' ? 'Signals & Broadcast' : tab}
                </button>
              ))}
            </div>

            {/* TAB CONTENT */}
            <div className="min-h-[500px]">
              {activeTab === 'tech' && (
                <div className="space-y-6 animate-in fade-in duration-500">
                  <PriceChart data={data} />
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <EntropyChart data={data} />
                    <FluxChart data={data} />
                  </div>
                </div>
              )}

              {activeTab === 'macro' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                   <div className="glass-panel p-6 rounded">
                      <h3 className="text-[#00F0FF] text-lg font-bold mb-4">Fundamentals</h3>
                      <div className="space-y-3 font-mono text-sm">
                        <div className="flex justify-between"><span>Market Cap</span><span className="text-white">{fundamentals.marketCap}</span></div>
                        <div className="flex justify-between"><span>P/E Ratio</span><span className="text-white">{fundamentals.peRatio}</span></div>
                        <div className="flex justify-between"><span>Rev Growth</span><span className="text-[#00E676]">{fundamentals.revGrowth}</span></div>
                        <div className="border-t border-[#333] pt-2 text-gray-400 text-xs italic">
                           {fundamentals.summary}
                        </div>
                      </div>
                   </div>
                   <div className="glass-panel p-6 rounded">
                      <h3 className="text-[#D500F9] text-lg font-bold mb-4">Fibonacci Targets</h3>
                       <div className="space-y-3 font-mono text-sm">
                        <div className="flex justify-between text-red-400"><span>Smart Stop</span><span>${fibs?.smart_stop.toFixed(2)}</span></div>
                        <div className="flex justify-between text-yellow-400"><span>TP1</span><span>${fibs?.tp1.toFixed(2)}</span></div>
                        <div className="flex justify-between text-green-400"><span>TP2</span><span>${fibs?.tp2.toFixed(2)}</span></div>
                        <div className="flex justify-between text-[#00E676]"><span>TP3</span><span>${fibs?.tp3.toFixed(2)}</span></div>
                      </div>
                   </div>
                </div>
              )}

              {activeTab === 'ai' && (
                 <div className="glass-panel p-6 rounded max-w-2xl mx-auto">
                    <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                       <span className="text-xl">🧠</span> AI Intelligence
                    </h3>
                    <select className="w-full bg-[#0a0a0a] border border-[#333] p-3 rounded mb-4 text-sm text-gray-300 outline-none focus:border-[#00F0FF]">
                        <option>Comprehensive Analysis</option>
                        <option>Technical Breakdown</option>
                        <option>Risk Assessment</option>
                    </select>
                    <button className="w-full bg-[#00F0FF] text-black font-bold p-3 rounded hover:bg-[#00c0cc] transition mb-4">
                        RUN ANALYSIS (OPENAI)
                    </button>
                    <div className="bg-[#0a0a0a] border border-[#333] p-4 rounded min-h-[200px] text-xs font-mono text-gray-400">
                        Analyzing {selectedTicker} using God Mode algorithms...
                        <br/>
                        > Checking CHEDO Entropy... Stable.
                        <br/>
                        > Flux Vector... Bullish Divergence detected.
                        <br/>
                        > Calculating SMC Order Blocks... Done.
                    </div>
                 </div>
              )}

              {activeTab === 'broadcast' && (
                <div className="space-y-6">
                    <VolumeProfileChart data={profile} poc={poc} />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="glass-panel p-6 rounded border-l-4 border-[#00F0FF]">
                            <div className="flex justify-between items-start mb-4">
                              <div>
                                <h3 className="text-white font-bold flex items-center gap-2">
                                  <span>📡 Signal Generator</span>
                                  <span className={`text-[10px] font-mono bg-[#111] px-2 py-1 rounded font-bold ${signalBias === 'LONG' ? 'text-[#00E676]' : signalBias === 'SHORT' ? 'text-[#FF1744]' : 'text-gray-500'}`}>
                                      {signalBias} BIAS
                                  </span>
                                </h3>
                                <div className="mt-2 flex items-center gap-2 text-xs">
                                  <span className="text-gray-500">CONFIDENCE</span>
                                  <div className="w-24 h-1.5 bg-[#111] rounded-full overflow-hidden">
                                    <div 
                                      className="h-full bg-gradient-to-r from-[#00F0FF] to-[#0055FF] transition-all duration-500"
                                      style={{ width: `${confidenceScore}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-[#00F0FF] font-mono">{confidenceScore}%</span>
                                </div>
                              </div>
                            </div>
                            
                            <div className="mb-4">
                            <label className="text-xs text-gray-500 uppercase block mb-1">Report Type</label>
                            <select 
                                value={selectedReportType} 
                                onChange={(e) => setSelectedReportType(e.target.value as ReportType)}
                                className="w-full bg-[#0a0a0a] border border-[#333] p-2 rounded text-sm text-[#00F0FF] outline-none focus:border-[#00F0FF]"
                            >
                                {REPORT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                            </div>

                            <button 
                            onClick={handleGenerateReport}
                            className="w-full bg-[#111] border border-[#333] text-white text-xs font-bold p-2 rounded hover:border-[#00F0FF] transition mb-4 uppercase tracking-widest hover:bg-[#1a1a1a]"
                            >
                                ⚡ Generate Signal Report
                            </button>

                            <div className="relative group">
                            <textarea 
                                className="w-full bg-[#0a0a0a] border border-[#333] p-3 rounded mb-4 text-xs text-[#00F0FF] font-mono h-64 focus:border-[#00F0FF] outline-none leading-relaxed custom-scrollbar"
                                value={broadcastMsg}
                                onChange={(e) => setBroadcastMsg(e.target.value)}
                                placeholder={`Generate a report above or type a custom signal...`}
                            />
                            {broadcastMsg && (
                                <button 
                                    onClick={handleCopy}
                                    className="absolute top-2 right-2 bg-[#222] hover:bg-[#333] text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition"
                                >
                                    COPY
                                </button>
                            )}
                            </div>
                            
                            <button 
                            onClick={handleBroadcast}
                            className="w-full bg-gradient-to-r from-[#00F0FF] to-[#0055FF] text-white font-bold p-3 rounded hover:opacity-90 transition shadow-[0_0_15px_rgba(0,240,255,0.3)]"
                            >
                                🚀 BROADCAST TO TELEGRAM
                            </button>
                        </div>

                        <div className="glass-panel p-6 rounded flex flex-col">
                            <h3 className="text-gray-400 font-bold mb-4 text-xs uppercase tracking-widest border-b border-[#222] pb-2 flex justify-between">
                              <span>Broadcast Log</span>
                              <span className="text-[#00F0FF]">{broadcastLog.length} SENT</span>
                            </h3>
                            <div className="space-y-4 flex-1 overflow-y-auto pr-2 custom-scrollbar bg-[#020202] p-2 rounded-inner shadow-inner">
                                {broadcastLog.length === 0 && <div className="text-gray-600 text-sm italic text-center mt-20">No signals transmitted.</div>}
                                {broadcastLog.map((log, i) => {
                                    const isBuy = log.message.includes("BUY") || log.message.includes("LONG") || log.message.includes("🟢");
                                    const isSell = log.message.includes("SELL") || log.message.includes("SHORT") || log.message.includes("🔴");
                                    const borderColor = isBuy ? "border-[#00E676]/30" : isSell ? "border-[#FF1744]/30" : "border-[#333]";
                                    
                                    return (
                                    <div key={i} className={`bg-[#0a0a0a] border ${borderColor} p-3 rounded-lg text-xs hover:bg-[#111] transition group relative max-w-[90%] ml-auto`}>
                                        <div className="flex justify-between text-gray-500 mb-1 font-mono uppercase tracking-tight text-[10px]">
                                            <span className="font-bold text-gray-300">{log.name}</span>
                                            <span>{log.scheduleTime}</span>
                                        </div>
                                        <div className="text-gray-400 font-mono whitespace-pre-wrap leading-relaxed opacity-90">
                                          {log.message.slice(0, 100)}...
                                        </div>
                                        <div className="absolute -right-2 top-3 w-0 h-0 border-t-[6px] border-t-transparent border-l-[8px] border-l-[#0a0a0a] border-b-[6px] border-b-transparent"></div>
                                    </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const MetricCard = ({ label, value, sub, color }: { label: string, value: string, sub: string, color: string }) => (
  <div className="bg-[#ffffff05] border-l-2 border-[#333] hover:border-[#00F0FF] hover:bg-[#ffffff10] transition-all p-4 backdrop-blur-sm group">
    <div className="text-gray-500 text-[10px] tracking-widest uppercase mb-1 font-bold">{label}</div>
    <div className={`text-2xl font-light ${color} font-mono mb-1`}>{value}</div>
    <div className="text-xs text-gray-600 group-hover:text-gray-400">{sub}</div>
  </div>
);

export default App;
