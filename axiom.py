# app.py
import json
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Axiom Quantitative | Titan Edition",
    page_icon="👁️",
)

# ==========================================
# SIDEBAR CONTROLS (sets initial state)
# ==========================================
st.sidebar.title("⚙️ Titan Controls")
ticker_options = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "SPY",
    "QQQ",
    "GC=F",
    "SI=F",
]
tf_options = ["15M", "1H", "4H", "1D", "1W"]

initial_ticker = st.sidebar.selectbox("Default Ticker", ticker_options, index=0)
initial_tf = st.sidebar.selectbox("Default Timeframe", tf_options, index=1)
iframe_height = st.sidebar.slider("UI Height", 700, 1400, 980, 10)

# Safely embed initial values into JS
INITIAL_TICKER_JS = json.dumps(initial_ticker)
INITIAL_TF_JS = json.dumps(initial_tf)

# ==========================================
# FULL EMBEDDED UI (React + Tailwind + Recharts)
# Rendered inside a Streamlit iframe via components.html
# ==========================================
HTML_APP = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Axiom Quantitative | Titan Edition</title>

    <script src="https://cdn.tailwindcss.com"></script>
    <link
      href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;700&display=swap"
      rel="stylesheet"
    />

    <style>
      body {{
        background-color: #050505;
        color: #e0e0e0;
        font-family: "Inter", sans-serif;
        margin: 0;
      }}
      .font-mono {{
        font-family: "Roboto Mono", monospace;
      }}
      /* Custom Scrollbar */
      ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
      }}
      ::-webkit-scrollbar-track {{
        background: #0a0a0a;
      }}
      ::-webkit-scrollbar-thumb {{
        background: #333;
        border-radius: 4px;
      }}
      ::-webkit-scrollbar-thumb:hover {{
        background: #00f0ff;
      }}
      .glass-panel {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
      }}
      .neon {{
        text-shadow: 0 0 16px rgba(0, 240, 255, 0.25);
      }}
      .ring-neon {{
        box-shadow: 0 0 0 1px rgba(0, 240, 255, 0.22), 0 0 22px rgba(0, 240, 255, 0.08);
      }}
    </style>

    <script type="importmap">
      {{
        "imports": {{
          "react/": "https://esm.sh/react@^19.2.3/",
          "react": "https://esm.sh/react@^19.2.3",
          "react-dom/": "https://esm.sh/react-dom@^19.2.3/",
          "recharts": "https://esm.sh/recharts@^3.6.0"
        }}
      }}
    </script>
  </head>

  <body>
    <div id="root"></div>

    <script type="module">
      import React, {{ useMemo, useState }} from "react";
      import {{ createRoot }} from "react-dom/client";
      import {{
        ResponsiveContainer,
        AreaChart,
        Area,
        XAxis,
        YAxis,
        Tooltip,
        CartesianGrid,
      }} from "recharts";

      // Initial state from Streamlit (sidebar)
      const STREAMLIT_INITIAL_TICKER = {INITIAL_TICKER_JS};
      const STREAMLIT_INITIAL_TF = {INITIAL_TF_JS};

      // -----------------------------
      // Mock Data Engine (wired)
      // -----------------------------
      function mulberry32(seed) {{
        let t = seed >>> 0;
        return function () {{
          t += 0x6D2B79F5;
          let r = Math.imul(t ^ (t >>> 15), 1 | t);
          r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
          return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
        }};
      }}

      function hashStringToSeed(s) {{
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) {{
          h ^= s.charCodeAt(i);
          h = Math.imul(h, 16777619);
        }}
        return h >>> 0;
      }}

      function genSeries({{ ticker, tf, points = 140 }}) {{
        const seed = hashStringToSeed(`${{ticker}}::${{tf}}`);
        const rnd = mulberry32(seed);

        const tfScale =
          tf === "15M" ? 1 :
          tf === "1H"  ? 2 :
          tf === "4H"  ? 3 :
          tf === "1D"  ? 4 :
          tf === "1W"  ? 6 : 3;

        let price = 100 + (seed % 9000) / 100;
        let vol = 0.6 * tfScale + rnd() * 1.6;

        const now = Date.now();
        const stepMs =
          tf === "15M" ? 15 * 60 * 1000 :
          tf === "1H"  ? 60 * 60 * 1000 :
          tf === "4H"  ? 4  * 60 * 60 * 1000 :
          tf === "1D"  ? 24 * 60 * 60 * 1000 :
          tf === "1W"  ? 7  * 24 * 60 * 60 * 1000 :
          60 * 60 * 1000;

        const data = [];
        for (let i = points - 1; i >= 0; i--) {{
          const drift = (rnd() - 0.48) * vol;
          const shock = (rnd() - 0.5) * (vol * 0.8);
          price = Math.max(1, price + drift + shock);

          const ts = now - i * stepMs;
          data.push({{
            t: ts,
            time: new Date(ts).toLocaleString(undefined, {{
              month: "short",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            }}),
            price: Number(price.toFixed(2)),
          }});
        }}

        const last = data[data.length - 1]?.price ?? 0;
        const prev = data[data.length - 2]?.price ?? last;
        const change = last - prev;
        const changePct = prev ? (change / prev) * 100 : 0;

        return {{ data, last, change, changePct }};
      }}

      // -----------------------------
      // UI Building Blocks
      // -----------------------------
      function clsx(...xs) {{
        return xs.filter(Boolean).join(" ");
      }}

      function Panel({{ title, subtitle, right, children, className }}) {{
        return (
          <div className={{clsx("glass-panel rounded-2xl p-4 ring-neon", className)}}>
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <div className="text-sm font-mono tracking-wide text-white/80">{{title}}</div>
                {{subtitle ? (
                  <div className="text-xs text-white/45 mt-1">{{subtitle}}</div>
                ) : null}}
              </div>
              {{right ? <div className="shrink-0">{{right}}</div> : null}}
            </div>
            {{children}}
          </div>
        );
      }}

      function Chip({{ children }}) {{
        return (
          <span className="inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-mono bg-white/5 border border-white/10 text-white/75">
            {{children}}
          </span>
        );
      }}

      function Button({{ children, onClick, variant = "primary" }}) {{
        const base =
          "inline-flex items-center justify-center rounded-xl px-3 py-2 text-sm font-mono border transition active:scale-[0.99]";
        const styles =
          variant === "primary"
            ? "bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/7 text-white/85"
            : "bg-transparent border-white/10 hover:border-white/20 text-white/70";
        return (
          <button className={{clsx(base, styles)}} onClick={{onClick}} type="button">
            {{children}}
          </button>
        );
      }}

      function Select({{ value, onChange, options }}) {{
        return (
          <select
            value={{value}}
            onChange={{(e) => onChange(e.target.value)}}
            className="font-mono text-sm bg-white/5 border border-white/10 rounded-xl px-3 py-2 outline-none hover:border-white/20"
          >
            {{options.map((o) => (
              <option key={{o.value}} value={{o.value}} className="bg-black">
                {{o.label}}
              </option>
            ))}}
          </select>
        );
      }}

      function TickerBanner({{ ticker, tf, last, changePct }}) {{
        const isUp = changePct >= 0;
        return (
          <div className="glass-panel rounded-2xl px-4 py-3 ring-neon flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-2.5 h-2.5 rounded-full" style={{{{ background: "rgba(0,240,255,0.85)" }}}} />
              <div className="min-w-0">
                <div className="font-mono text-white/90 tracking-wide neon truncate">
                  {{ticker}} <span className="text-white/45">/</span> {{tf}}
                </div>
                <div className="text-xs text-white/45 truncate">
                  Axiom Quantitative • Titan Edition • Live banner wired to selection
                </div>
              </div>
            </div>

            <div className="flex items-end gap-3 shrink-0">
              <div className="text-right">
                <div className="font-mono text-white/90 text-sm">{{last?.toFixed?.(2) ?? "—"}}</div>
                <div className={{clsx("font-mono text-[11px]", isUp ? "text-white/70" : "text-white/70")}}>
                  {{isUp ? "▲" : "▼"}} {{changePct?.toFixed?.(2) ?? "0.00"}}%
                </div>
              </div>
              <Chip>STREAM</Chip>
            </div>
          </div>
        );
      }}

      function PriceTooltip({{ active, payload, label }}) {{
        if (!active || !payload?.length) return null;
        const p = payload[0]?.payload;
        return (
          <div className="glass-panel rounded-xl px-3 py-2 border border-white/10">
            <div className="font-mono text-xs text-white/80">{{p?.time ?? label}}</div>
            <div className="font-mono text-sm text-white/90 mt-1">
              {{p?.price?.toFixed?.(2) ?? "—"}}
            </div>
          </div>
        );
      }}

      // -----------------------------
      // Main App
      // -----------------------------
      function App() {{
        const NAV = [
          {{ id: "overview", label: "Overview" }},
          {{ id: "charts", label: "Charts" }},
          {{ id: "signals", label: "Signals" }},
          {{ id: "scanner", label: "Scanner" }},
          {{ id: "ai", label: "AI" }},
          {{ id: "settings", label: "Settings" }},
          {{ id: "logs", label: "Logs" }},
        ];

        const TICKERS = [
          {{ value: "BTC-USD", label: "BTC-USD" }},
          {{ value: "ETH-USD", label: "ETH-USD" }},
          {{ value: "SOL-USD", label: "SOL-USD" }},
          {{ value: "XRP-USD", label: "XRP-USD" }},
          {{ value: "SPY", label: "SPY" }},
          {{ value: "QQQ", label: "QQQ" }},
          {{ value: "GC=F", label: "Gold (GC=F)" }},
          {{ value: "SI=F", label: "Silver (SI=F)" }},
        ];

        const TIMEFRAMES = [
          {{ value: "15M", label: "15M" }},
          {{ value: "1H", label: "1H" }},
          {{ value: "4H", label: "4H" }},
          {{ value: "1D", label: "1D" }},
          {{ value: "1W", label: "1W" }},
        ];

        const [active, setActive] = useState("overview");
        const [ticker, setTicker] = useState(STREAMLIT_INITIAL_TICKER || "BTC-USD");
        const [tf, setTf] = useState(STREAMLIT_INITIAL_TF || "1H");

        const series = useMemo(() => genSeries({{ ticker, tf, points: 160 }}), [ticker, tf]);

        const kpis = useMemo(() => {{
          const last = series.last ?? 0;
          const changePct = series.changePct ?? 0;
          const regime =
            Math.abs(changePct) < 0.15 ? "Neutral" : changePct > 0 ? "Bull" : "Bear";
          const conf =
            regime === "Neutral" ? 54 :
            regime === "Bull" ? 71 : 69;

          return {{
            last,
            changePct,
            regime,
            conf,
            volatility: (0.8 + (Math.abs(changePct) * 2)).toFixed(2),
            trend: changePct > 0 ? "Uptrend" : changePct < 0 ? "Downtrend" : "Range",
          }};
        }}, [series]);

        const signals = useMemo(() => {{
          const isBull = kpis.regime === "Bull";
          const isBear = kpis.regime === "Bear";
          return [
            {{
              ts: "Most Recent",
              side: isBull ? "LONG" : isBear ? "SHORT" : "WAIT",
              reason: "Composite regime + momentum proxy (mock)",
              conf: `${{kpis.conf}}%`,
              level: isBull ? "Buy pullbacks" : isBear ? "Sell rallies" : "Hold levels",
            }},
            {{
              ts: "Prev",
              side: isBull ? "LONG" : isBear ? "SHORT" : "WAIT",
              reason: "Trend state + volatility gate (mock)",
              conf: `${{Math.max(50, kpis.conf - 7)}}%`,
              level: "Key S/R zones",
            }},
          ];
        }}, [kpis]);

        const logs = useMemo(
          () => [
            `[BOOT] UI loaded`,
            `[STATE] ticker=${{ticker}} tf=${{tf}}`,
            `[DATA] series_points=${{series.data.length}}`,
            `[SIGNALS] generated=${{signals.length}}`,
          ],
          [ticker, tf, series.data.length, signals.length]
        );

        return (
          <div className="min-h-screen">
            {/* Top Header */}
            <div className="sticky top-0 z-30 bg-[#050505]/80 backdrop-blur border-b border-white/5">
              <div className="max-w-[1600px] mx-auto px-4 py-3 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl glass-panel ring-neon flex items-center justify-center font-mono text-white/90">
                    AQ
                  </div>
                  <div className="min-w-0">
                    <div className="font-mono text-white/85 tracking-wide neon truncate">
                      Axiom Quantitative <span className="text-white/35">|</span> Titan Edition
                    </div>
                    <div className="text-xs text-white/45 truncate">
                      Streamlit embed • React + Tailwind + Recharts
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Select value={ticker} onChange={setTicker} options={TICKERS} />
                  <Select value={tf} onChange={setTf} options={TIMEFRAMES} />
                  <Button onClick={() => alert("Mock refresh — wire this to your data fetch later.")}>
                    Refresh
                  </Button>
                </div>
              </div>
            </div>

            <div className="max-w-[1600px] mx-auto px-4 py-4 grid grid-cols-12 gap-4">
              {/* Sidebar */}
              <aside className="col-span-12 lg:col-span-2">
                <div className="glass-panel ring-neon rounded-2xl p-3">
                  <div className="font-mono text-xs text-white/55 mb-2">NAV</div>
                  <nav className="flex lg:flex-col gap-2 flex-wrap">
                    {NAV.map((n) => (
                      <button
                        key={n.id}
                        onClick={() => setActive(n.id)}
                        className={clsx(
                          "text-left font-mono text-sm rounded-xl px-3 py-2 border transition",
                          active === n.id
                            ? "bg-white/7 border-white/20 text-white/90"
                            : "bg-transparent border-white/10 text-white/65 hover:border-white/20 hover:bg-white/5"
                        )}
                      >
                        {n.label}
                      </button>
                    ))}
                  </nav>

                  <div className="mt-4 pt-3 border-t border-white/5">
                    <div className="font-mono text-xs text-white/55 mb-2">STATUS</div>
                    <div className="flex items-center justify-between gap-2">
                      <Chip>CONNECTED</Chip>
                      <Chip>MOCK DATA</Chip>
                    </div>
                  </div>
                </div>
              </aside>

              {/* Main */}
              <main className="col-span-12 lg:col-span-10 space-y-4">
                <TickerBanner ticker={ticker} tf={tf} last={kpis.last} changePct={kpis.changePct} />

                {/* KPI Row */}
                <div className="grid grid-cols-12 gap-4">
                  <Panel className="col-span-12 md:col-span-3" title="Regime" subtitle="Market state (mock)">
                    <div className="mt-1 flex items-end justify-between">
                      <div className="font-mono text-2xl text-white/90">{kpis.regime}</div>
                      <Chip>{kpis.trend}</Chip>
                    </div>
                    <div className="mt-2 text-xs text-white/45">
                      Hook this to your real regime engine when ready.
                    </div>
                  </Panel>

                  <Panel className="col-span-12 md:col-span-3" title="Confidence" subtitle="Composite score (mock)">
                    <div className="mt-1 flex items-end justify-between">
                      <div className="font-mono text-2xl text-white/90">{kpis.conf}%</div>
                      <Chip>GATED</Chip>
                    </div>
                    <div className="mt-2 text-xs text-white/45">
                      Add bull/bear thresholds & guards here.
                    </div>
                  </Panel>

                  <Panel className="col-span-12 md:col-span-3" title="Volatility" subtitle="Proxy (mock)">
                    <div className="mt-1 flex items-end justify-between">
                      <div className="font-mono text-2xl text-white/90">{kpis.volatility}</div>
                      <Chip>ATR-ish</Chip>
                    </div>
                    <div className="mt-2 text-xs text-white/45">
                      Replace with ATR/realized vol.
                    </div>
                  </Panel>

                  <Panel className="col-span-12 md:col-span-3" title="Last Price" subtitle="Latest point">
                    <div className="mt-1 flex items-end justify-between">
                      <div className="font-mono text-2xl text-white/90">{kpis.last.toFixed(2)}</div>
                      <Chip>
                        {kpis.changePct >= 0 ? "▲" : "▼"} {kpis.changePct.toFixed(2)}%
                      </Chip>
                    </div>
                    <div className="mt-2 text-xs text-white/45">
                      Feed this from your provider (yfinance/binance/etc).
                    </div>
                  </Panel>
                </div>

                {/* Content Tabs */}
                {(active === "overview" || active === "charts") ? (
                  <div className="grid grid-cols-12 gap-4">
                    <Panel
                      className="col-span-12 lg:col-span-8"
                      title="Price (Mock Series)"
                      subtitle="Wired to ticker/timeframe selection"
                      right={<Chip>{ticker}</Chip>}
                    >
                      <div className="h-[360px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={series.data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                            <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                            <XAxis
                              dataKey="time"
                              tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 11, fontFamily: "Roboto Mono" }}
                              tickLine={false}
                              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                              minTickGap={24}
                            />
                            <YAxis
                              tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 11, fontFamily: "Roboto Mono" }}
                              tickLine={false}
                              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                              width={56}
                              domain={["auto", "auto"]}
                            />
                            <Tooltip content={<PriceTooltip />} />
                            <Area
                              type="monotone"
                              dataKey="price"
                              stroke="rgba(0,240,255,0.8)"
                              fill="rgba(0,240,255,0.12)"
                              strokeWidth={2}
                              dot={false}
                              activeDot={{ r: 3 }}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="mt-3 flex flex-wrap gap-2">
                        <Chip>EMA 50/200 (placeholder)</Chip>
                        <Chip>RSI (placeholder)</Chip>
                        <Chip>MACD (placeholder)</Chip>
                        <Chip>ATR (placeholder)</Chip>
                      </div>
                    </Panel>

                    <Panel className="col-span-12 lg:col-span-4" title="Signals" subtitle="Mock structure ready">
                      <div className="space-y-2">
                        {signals.map((s, i) => (
                          <div key={i} className="rounded-xl border border-white/10 bg-white/3 p-3">
                            <div className="flex items-center justify-between gap-2">
                              <div className="font-mono text-sm text-white/85">{s.side}</div>
                              <Chip>{s.conf}</Chip>
                            </div>
                            <div className="text-xs text-white/45 mt-1">{s.reason}</div>
                            <div className="text-xs text-white/60 mt-2 font-mono">Level: {s.level}</div>
                          </div>
                        ))}
                      </div>

                      <div className="mt-3 flex gap-2">
                        <Button variant="secondary" onClick={() => alert("Mock export: CSV")}>Export CSV</Button>
                        <Button variant="secondary" onClick={() => alert("Mock export: Markdown")}>Export MD</Button>
                      </div>
                    </Panel>
                  </div>
                ) : null}

                {active === "signals" ? (
                  <Panel title="Signals Table" subtitle="Mock table ready for real engine output">
                    <div className="overflow-auto rounded-xl border border-white/10">
                      <table className="w-full text-sm">
                        <thead className="bg-white/3">
                          <tr className="text-left">
                            <th className="p-3 font-mono text-xs text-white/60">Time</th>
                            <th className="p-3 font-mono text-xs text-white/60">Side</th>
                            <th className="p-3 font-mono text-xs text-white/60">Confidence</th>
                            <th className="p-3 font-mono text-xs text-white/60">Reason</th>
                            <th className="p-3 font-mono text-xs text-white/60">Level</th>
                          </tr>
                        </thead>
                        <tbody>
                          {signals.map((s, i) => (
                            <tr key={i} className="border-t border-white/5 hover:bg-white/2">
                              <td className="p-3 text-white/70 font-mono text-xs">{s.ts}</td>
                              <td className="p-3 text-white/85 font-mono">{s.side}</td>
                              <td className="p-3 text-white/70 font-mono">{s.conf}</td>
                              <td className="p-3 text-white/60">{s.reason}</td>
                              <td className="p-3 text-white/60">{s.level}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Panel>
                ) : null}

                {active === "scanner" ? (
                  <Panel title="Scanner" subtitle="Wire to your watchlists + filters">
                    <div className="text-white/55 text-sm">
                      Placeholder scanner panel. Add:
                      <ul className="list-disc ml-5 mt-2 space-y-1 text-white/45">
                        <li>Search + multi-select universe</li>
                        <li>Rules engine (trend/vol/RSI/regime)</li>
                        <li>Batch compute + ranked output</li>
                      </ul>
                    </div>
                  </Panel>
                ) : null}

                {active === "ai" ? (
                  <Panel title="AI" subtitle="Drop-in OpenAI/Gemini analysis panel">
                    <div className="grid grid-cols-12 gap-4">
                      <div className="col-span-12 lg:col-span-7">
                        <div className="rounded-xl border border-white/10 bg-white/3 p-3">
                          <div className="font-mono text-xs text-white/60 mb-2">Prompt</div>
                          <textarea
                            className="w-full h-40 bg-transparent outline-none font-mono text-sm text-white/75"
                            defaultValue={`Analyze ${ticker} on ${tf}. Provide: regime, key levels, invalidation, and a clean trade plan.`}
                          ></textarea>
                        </div>
                        <div className="mt-3 flex gap-2">
                          <Button onClick={() => alert("Mock AI run — wire to your API calls.")}>Run AI</Button>
                          <Button variant="secondary" onClick={() => alert("Mock send to Telegram")}>
                            Send to Telegram
                          </Button>
                        </div>
                      </div>
                      <div className="col-span-12 lg:col-span-5">
                        <div className="rounded-xl border border-white/10 bg-white/3 p-3 h-full">
                          <div className="font-mono text-xs text-white/60 mb-2">Output</div>
                          <div className="text-sm text-white/55 leading-relaxed">
                            Mock AI output area. When wired, render:
                            <ul className="list-disc ml-5 mt-2 space-y-1 text-white/45">
                              <li>Structured bullets</li>
                              <li>Signal summary</li>
                              <li>Risk parameters</li>
                              <li>Markdown export</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Panel>
                ) : null}

                {active === "settings" ? (
                  <Panel title="Settings" subtitle="Keys + integrations go here">
                    <div className="grid grid-cols-12 gap-4">
                      <div className="col-span-12 lg:col-span-6 rounded-xl border border-white/10 bg-white/3 p-3">
                        <div className="font-mono text-xs text-white/60 mb-2">Integrations</div>
                        <div className="space-y-2 text-sm text-white/55">
                          <div className="flex items-center justify-between gap-2">
                            <span>TradingView Banner</span>
                            <Chip>ENABLED (UI)</Chip>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>Telegram</span>
                            <Chip>PLACEHOLDER</Chip>
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <span>X.com Broadcast</span>
                            <Chip>PLACEHOLDER</Chip>
                          </div>
                        </div>
                      </div>

                      <div className="col-span-12 lg:col-span-6 rounded-xl border border-white/10 bg-white/3 p-3">
                        <div className="font-mono text-xs text-white/60 mb-2">UI</div>
                        <div className="text-sm text-white/55">
                          You can add theme toggles, density, table sizing, etc.
                        </div>
                      </div>
                    </div>
                  </Panel>
                ) : null}

                {active === "logs" ? (
                  <Panel title="Logs" subtitle="System / pipeline logs">
                    <div className="rounded-xl border border-white/10 bg-black/30 p-3 font-mono text-xs text-white/65 space-y-1">
                      {logs.map((l, i) => (
                        <div key={i}>{l}</div>
                      ))}
                    </div>
                  </Panel>
                ) : null}
              </main>
            </div>
          </div>
        );
      }}

      createRoot(document.getElementById("root")).render(<App />);
    </script>
  </body>
</html>
"""

# ==========================================
# STREAMLIT RENDER
# ==========================================
st.title("👁️ Axiom Quantitative | Titan Edition (Embedded UI)")
st.caption("This Streamlit app embeds your React/Tailwind/Recharts UI in an iframe.")

components.html(HTML_APP, height=iframe_height, scrolling=True)
