import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  Settings,
  Terminal,
  Activity,
  Users,
  Building2,
  DollarSign,
  Zap,
  Save,
  RotateCcw,
  BarChart3,
  Globe,
  TrendingUp,
  Triangle,
  Lock
} from 'lucide-react';

// --- STYLES ---
// "Oberon Command" Theme - Sharp, Technical, Cold
const techStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  .font-display { font-family: 'Chakra Petch', sans-serif; }
  .font-mono { font-family: 'JetBrains Mono', monospace; }

  .bg-tech-grid {
    background-color: #0b0c15;
    background-image: 
      linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
    background-size: 30px 30px;
  }

  .tech-panel {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.15);
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
  }

  /* The "Bracket" corners effect */
  .tech-corners {
    position: relative;
  }
  .tech-corners::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 10px; height: 10px;
    border-top: 2px solid #0ea5e9;
    border-left: 2px solid #0ea5e9;
    z-index: 10;
  }
  .tech-corners::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 10px; height: 10px;
    border-bottom: 2px solid #0ea5e9;
    border-right: 2px solid #0ea5e9;
    z-index: 10;
  }

  .btn-tech {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #38bdf8;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
  }
  
  .btn-tech:hover:not(:disabled) {
    background: rgba(14, 165, 233, 0.2);
    border-color: #38bdf8;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
  }

  .btn-tech:active:not(:disabled) {
    transform: scale(0.98);
  }

  .btn-tech.active {
    background: #0ea5e9;
    color: #000;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.5);
  }
  
  .btn-tech:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    border-color: #334155;
    color: #475569;
  }
  
  .btn-danger {
    border-color: rgba(239, 68, 68, 0.4);
    color: #f87171;
    background: rgba(239, 68, 68, 0.1);
  }
  .btn-danger:hover {
    border-color: #ef4444;
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
  }
  
  .btn-primary-large {
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid #0ea5e9;
    color: #38bdf8;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.2);
  }
  .btn-primary-large:hover {
    background: #0ea5e9;
    color: #000;
    box-shadow: 0 0 30px rgba(14, 165, 233, 0.6);
  }

  .progress-bar {
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.2);
    height: 8px;
    width: 100%;
    position: relative;
  }
  
  .progress-fill {
    background: #0ea5e9;
    height: 100%;
    box-shadow: 0 0 8px rgba(14, 165, 233, 0.6);
  }

  /* Custom range input */
  input[type=range] {
    -webkit-appearance: none;
    background: transparent;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 16px;
    width: 8px;
    background: #38bdf8;
    cursor: pointer;
    margin-top: -6px;
    box-shadow: 0 0 5px #38bdf8;
  }
  input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 4px;
    background: #1e293b;
    border: 1px solid #334155;
  }
`;

// --- COMPONENTS ---

const NavButton = ({ icon: Icon, label, isActive, onClick, disabled }) => (
  <button
    onClick={disabled ? null : onClick}
    disabled={disabled}
    className={`w-full p-4 flex flex-col items-center justify-center space-y-1 transition-all border-l-2 ${isActive
      ? 'border-sky-500 bg-sky-500/10 text-sky-400'
      : disabled
        ? 'border-transparent text-slate-700 cursor-not-allowed'
        : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
      }`}
  >
    {disabled ? <Lock size={20} className="mb-1 opacity-50" /> : <Icon size={24} />}
    <span className="text-[10px] uppercase font-display tracking-widest">{label}</span>
  </button>
);

const StatTile = ({ label, value, trend, suffix = "" }) => (
  <div className="tech-panel p-4 flex flex-col tech-corners group">
    <div className="flex justify-between items-start mb-2">
      <span className="text-xs uppercase tracking-widest text-slate-400 font-display">{label}</span>
      {trend && (
        <span className={`text-xs ${trend > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          {trend > 0 ? '▲' : '▼'} {Math.abs(trend)}%
        </span>
      )}
    </div>
    <div className="flex items-baseline space-x-1">
      <span className="text-3xl font-display font-bold text-slate-100 group-hover:text-sky-400 transition-colors">
        {value}
      </span>
      <span className="text-sm text-slate-500 font-mono">{suffix}</span>
    </div>
    <div className="w-full h-[2px] bg-slate-800 mt-3 relative overflow-hidden">
      <div className="absolute top-0 left-0 h-full bg-sky-500 w-1/3 animate-pulse"></div>
    </div>
  </div>
);

const TechSlider = ({ label, value, onChange, min, max, step, format = v => v }) => (
  <div className="mb-6">
    <div className="flex justify-between items-end mb-2 font-display">
      <label className="text-sm text-slate-300 font-medium tracking-wide">{label}</label>
      <span className="text-sky-400 font-mono bg-sky-950/30 px-2 py-0.5 rounded text-sm border border-sky-500/20">
        {format(value)}
      </span>
    </div>
    <div className="relative flex items-center">
      <div className="h-2 w-2 bg-slate-600 rounded-full mr-2"></div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="h-2 w-2 bg-sky-500 rounded-full ml-2 shadow-[0_0_5px_rgba(14,165,233,0.8)]"></div>
    </div>
  </div>
);

// --- REUSABLE CHART COMPONENT ---
const LineChart = ({ title, data, color, minScale = 0, suffix = "", formatValue = v => v.toFixed(1) }) => {
  // Normalize data to array of arrays for multi-line support
  const datasets = Array.isArray(data[0]) ? data : [data];
  const colors = Array.isArray(color) ? color : [color];

  // Check if we have enough data in the primary dataset
  if (!datasets[0] || datasets[0].length < 2) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-600 font-mono text-xs border-l border-b border-slate-700/50 bg-slate-900/20">
        {datasets[0] && datasets[0].length === 1 ? "CALCULATING..." : "WAITING..."}
      </div>
    );
  }

  // Calculate global min/max across all datasets
  const allValues = datasets.flatMap(d => d.map(p => p.value));
  const minVal = Math.min(...allValues, minScale);
  const maxVal = Math.max(...allValues, minScale + 0.1);
  const range = maxVal - minVal || 1;

  // Zero line Y position
  const zeroY = 100 - ((0 - minVal) / range) * 80 - 10;

  // Get last value of primary dataset for display
  const lastValue = datasets[0][datasets[0].length - 1].value;

  return (
    <div className="flex-1 relative border-l border-b border-slate-700/50 px-2 pb-0 overflow-hidden bg-slate-900/20 flex flex-col min-h-[140px]">
      <div className="absolute top-2 left-3 z-10">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{title}</h4>
        <div className="text-lg font-mono font-bold text-slate-200">
          {formatValue(lastValue)}{suffix}
        </div>
      </div>

      <div className="flex-1 relative w-full h-full">
        <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
          {/* Grid Lines */}
          {[0, 25, 50, 75, 100].map(p => (
            <line key={p} x1="0" y1={p} x2="100" y2={p} stroke="#1e293b" strokeWidth="0.5" />
          ))}

          {/* Zero Line */}
          {zeroY >= 0 && zeroY <= 100 && (
            <line x1="0" y1={zeroY} x2="100" y2={zeroY} stroke="#475569" strokeWidth="0.5" strokeDasharray="2 2" />
          )}

          {/* Render each dataset */}
          {datasets.map((dataset, dIdx) => {
            const lineColor = colors[dIdx % colors.length];
            const points = dataset.map((point, i) => {
              const x = (i / (dataset.length - 1)) * 100;
              const y = 100 - ((point.value - minVal) / range) * 80 - 10;
              return `${x},${y}`;
            }).join(' ');

            return (
              <g key={dIdx}>
                <polyline points={points} fill="none" stroke={lineColor} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                {dataset.map((point, i) => {
                  const x = (i / (dataset.length - 1)) * 100;
                  const y = 100 - ((point.value - minVal) / range) * 80 - 10;

                  return (
                    <g key={i} className="group">
                      {/* Invisible hit area for tooltip */}
                      <circle cx={x} cy={y} r="3" fill="transparent" className="cursor-crosshair" />
                      {/* Visible dot only on hover */}
                      <circle cx={x} cy={y} r="2" fill={lineColor} className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

                      {/* Tooltip (only for primary line to avoid clutter) */}
                      {dIdx === 0 && (
                        <foreignObject x={x - 20} y={y - 25} width="40" height="20" className="overflow-visible opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                          <div className="bg-slate-800 text-[10px] text-white px-1 py-0.5 rounded border border-slate-600 whitespace-nowrap text-center shadow-lg z-50">
                            {formatValue(point.value)}{suffix}
                          </div>
                        </foreignObject>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export default function EcoSimUI() {
  const [activeView, setActiveView] = useState('CONFIG'); // Start at CONFIG
  const [isInitialized, setIsInitialized] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);
  const ws = useRef(null);
  const [isInitializing, setIsInitializing] = useState(false);

  // Simulation State
  const [metrics, setMetrics] = useState({
    unemployment: 99.0,
    gdp: 0.0,
    govDebt: 0.0,
    govProfit: 0.0,
    happiness: 50,
    housingInv: 0,
    avgWage: 0.0,
    gdpHistory: [],
    unemploymentHistory: [],
    wageHistory: [],
    medianWageHistory: [],
    happinessHistory: [],
    healthHistory: [],
    govProfitHistory: [],
    govDebtHistory: [],
    firmCountHistory: [],
    priceHistory: { food: [], housing: [], services: [] },
    supplyHistory: { food: [], housing: [], services: [] }
  });

  const [config, setConfig] = useState({
    wageTax: 0.05,
    profitTax: 0.30,
    hiringSpeed: 0.15,
    housingCap: 1.05,
    firmStrategy: 0.4
  });

  // Setup State (for initialization)
  const [setupConfig, setSetupConfig] = useState({
    num_households: 1000,
    num_firms: 5,
    wage_tax: 0.15,
    profit_tax: 0.20
  });

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // WebSocket Connection
  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8002/ws");
    ws.current.onopen = () => console.log("WS Connected");
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "SETUP_COMPLETE") {
        setIsInitializing(false);
        setIsInitialized(true);
        setActiveView('DASHBOARD');
        setIsRunning(true);
        // Sync local config with setup
        setConfig(prev => ({
          ...prev,
          wageTax: setupConfig.wage_tax,
          profitTax: setupConfig.profit_tax
        }));
        // Add boot sequence logs
        setLogs([
          { tick: 0, type: 'SYS', txt: 'INITIALIZING KERNEL...' },
          { tick: 0, type: 'SYS', txt: 'LOADING CONFIGURATION MAP...' },
          { tick: 0, type: 'SYS', txt: `SPAWNING ${setupConfig.num_households} AGENTS...` },
          { tick: 0, type: 'ECO', txt: 'WARMUP PHASE STARTED' }
        ]);
        // Auto-start simulation after setup
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ command: "START" }));
        }
      } else if (data.type === "RESET") {
        setTick(0);
        setLogs([]);
        setMetrics({
          unemployment: 99.0,
          gdp: 0,
          govDebt: 0,
          govProfit: 0,
          happiness: 50,
          housingInv: 0,
          avgWage: 0,
          gdpHistory: [],
          unemploymentHistory: [],
          wageHistory: [],
          medianWageHistory: [],
          happinessHistory: [],
          healthHistory: [],
          govProfitHistory: [],
          govDebtHistory: [],
          housingHistory: [],
          foodHistory: [],
          servicesHistory: []
        });
        setIsRunning(false);
        setIsInitialized(false);
        setActiveView('CONFIG'); // Go back to config on reset
      } else if (data.metrics) {
        setTick(data.tick);
        // Merge with existing metrics to preserve defaults if backend is missing keys
        setMetrics(prev => ({
          ...prev,
          ...data.metrics,
          // Ensure nested objects/arrays are not overwritten with undefined if missing
          priceHistory: data.metrics.priceHistory || prev.priceHistory || { food: [], housing: [], services: [] },
          supplyHistory: data.metrics.supplyHistory || prev.supplyHistory || { food: [], housing: [], services: [] },
          netWorthHistory: data.metrics.netWorthHistory || prev.netWorthHistory || []
        }));
        if (data.logs && data.logs.length > 0) {
          setLogs(prev => [...prev.slice(-20), ...data.logs]);
        }
      } else if (data.error) {
        console.error("Simulation error:", data.error);
        setIsInitializing(false);
      }
    };
    ws.current.onclose = () => {
      console.log("WS Disconnected");
      setIsRunning(false);
    }
    return () => ws.current.close();
  }, [setupConfig]);

  const handleInitialize = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      setIsInitializing(true);
      ws.current.send(JSON.stringify({
        command: "SETUP",
        config: setupConfig
      }));
    }
  };

  const toggleRun = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      if (isRunning) {
        ws.current.send(JSON.stringify({ command: "STOP" }));
      } else {
        ws.current.send(JSON.stringify({ command: "START" }));
      }
      setIsRunning(!isRunning);
    }
  };

  const handleReset = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: "RESET" }));
    }
  };

  const handleConfigChange = (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    if (ws.current && ws.current.readyState === WebSocket.OPEN && isInitialized) {
      ws.current.send(JSON.stringify({ command: "CONFIG", config: newConfig }));
    }
  };

  // Helper to update setup config
  const handleSetupChange = (key, value) => {
    setSetupConfig(prev => ({ ...prev, [key]: value }));
    // Also update the runtime config preview
    if (key === 'wage_tax') setConfig(prev => ({ ...prev, wageTax: value }));
    if (key === 'profit_tax') setConfig(prev => ({ ...prev, profitTax: value }));
  };

  return (
    <div className="min-h-screen bg-black text-slate-300 font-display selection:bg-sky-500/30 overflow-hidden flex">
      <style>{techStyles}</style>

      {/* SIDEBAR NAVIGATION */}
      <nav className="w-20 bg-slate-900/50 backdrop-blur-md border-r border-slate-800 flex flex-col justify-between z-20">
        <div>
          <div className="h-20 flex items-center justify-center border-b border-slate-800 mb-2">
            <Triangle className="text-sky-500 fill-sky-500/20" size={32} strokeWidth={1.5} />
          </div>
          {/* CONFIG is always active, but others are disabled until initialized */}
          <NavButton icon={Settings} label="Config" isActive={activeView === 'CONFIG'} onClick={() => setActiveView('CONFIG')} />
          <NavButton icon={Activity} label="Dash" isActive={activeView === 'DASHBOARD'} onClick={() => setActiveView('DASHBOARD')} disabled={!isInitialized} />
          <NavButton icon={Terminal} label="Logs" isActive={activeView === 'LOGS'} onClick={() => setActiveView('LOGS')} disabled={!isInitialized} />
        </div>

        {isInitialized && (
          <div className="p-4 flex flex-col items-center space-y-4 mb-4 animate-in fade-in duration-500">
            <div className={`h-2 w-2 rounded-full ${isRunning ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-rose-500'}`}></div>
            <span className="text-[10px] font-mono text-slate-500">{isRunning ? 'ONLINE' : 'HALTED'}</span>
          </div>
        )}
      </nav>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 bg-tech-grid relative flex flex-col">
        {/* TOP BAR */}
        <header className="h-16 border-b border-slate-800/50 bg-slate-900/30 flex items-center justify-between px-8 backdrop-blur-sm z-10">
          <div className="flex items-center space-x-6">
            <h1 className="text-xl font-bold tracking-widest text-slate-100">
              ECO<span className="text-sky-500">SIM</span> // OBERON
            </h1>
            <div className="h-6 w-[1px] bg-slate-700"></div>
            <div className="font-mono text-sm text-sky-400">
              {isInitialized ? (
                <>TICK_CYCLE: <span className="text-white">{tick.toString().padStart(5, '0')}</span></>
              ) : (
                <span className="text-amber-500">AWAITING INITIALIZATION</span>
              )}
            </div>
          </div>

          {isInitialized && (
            <div className="flex items-center space-x-4 animate-in fade-in slide-in-from-right-4 duration-500">
              <button onClick={toggleRun} className={`btn-tech px-6 py-2 flex items-center space-x-2 ${isRunning ? 'active' : ''}`}>
                {isRunning ? <Pause size={18} /> : <Play size={18} />}
                <span>{isRunning ? 'SUSPEND' : 'EXECUTE'}</span>
              </button>
              <button onClick={handleReset} className="btn-tech btn-danger p-2">
                <RotateCcw size={18} />
              </button>
            </div>
          )}
        </header>

        {/* CONTENT SCROLLABLE */}
        <div className="flex-1 overflow-y-auto p-8 relative">

          {/* DASHBOARD VIEW */}
          {activeView === 'DASHBOARD' && (
            <div className="grid grid-cols-12 gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* KEY METRICS ROW - COMPACT */}
              <div className="col-span-12 grid grid-cols-8 gap-4 mb-2">
                <StatTile label="GDP Output" value={`$${metrics.gdp.toFixed(1)}M`} trend={2.4} />
                <StatTile label="Net Worth" value={`$${(metrics.netWorth || 0).toFixed(1)}M`} trend={1.5} />
                <StatTile label="Gov Profit" value={`$${(metrics.govProfit || 0).toFixed(2)}M`} trend={metrics.govProfit > 0 ? 1 : -1} />
                <StatTile label="Gov Debt" value={`$${metrics.govDebt.toFixed(1)}M`} trend={-0.8} />
                <StatTile label="Unemployment" value={`${metrics.unemployment.toFixed(1)}%`} trend={-1.2} />
                <StatTile label="Employment" value={`${(100 - metrics.unemployment).toFixed(1)}%`} trend={1.2} />
                <StatTile label="Avg Wage" value={`$${metrics.avgWage.toFixed(2)}`} trend={0.5} />
                <StatTile label="Happiness" value={`${metrics.happiness.toFixed(1)}`} trend={0.1} />
              </div>

              {/* MAIN VISUALIZER - MULTI-GRAPH GRID */}
              <div className="col-span-12 tech-panel min-h-[600px] p-4 flex flex-col">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-display font-bold text-lg text-slate-200 flex items-center">
                    <BarChart3 className="mr-2 text-sky-500" size={20} />
                    ECONOMIC MONITOR (50-TICK INTERVALS)
                  </h3>
                  <div className="flex space-x-2">
                    <span className="px-2 py-1 bg-slate-800 text-xs font-mono text-slate-400 border border-slate-700">REALTIME</span>
                    <span className="px-2 py-1 bg-slate-800 text-xs font-mono text-slate-400 border border-slate-700">MACRO</span>
                  </div>
                </div>

                <div className="flex-1 grid grid-cols-3 gap-4">
                  {/* 1. GDP GROWTH */}
                  <LineChart
                    title="GDP GROWTH"
                    data={metrics.gdpHistory}
                    color="#38bdf8" // Sky Blue
                    minScale={0.5}
                    suffix="M"
                    formatValue={v => `$${v.toFixed(2)}`}
                  />

                  {/* 2. WAGE TRENDS (Mean vs Median) */}
                  <LineChart
                    title="WAGE TRENDS (MEAN/MEDIAN)"
                    data={[metrics.wageHistory, metrics.medianWageHistory]}
                    color={["#10b981", "#fbbf24"]} // Emerald, Amber
                    minScale={0}
                    suffix=""
                    formatValue={v => `$${v.toFixed(2)}`}
                  />

                  {/* 3. UNEMPLOYMENT */}
                  <LineChart
                    title="UNEMPLOYMENT RATE"
                    data={metrics.unemploymentHistory}
                    color="#ef4444" // Red
                    minScale={0}
                    suffix="%"
                    formatValue={v => v.toFixed(1)}
                  />

                  {/* 4. TOTAL NET WORTH (Replaces Gov Debt) */}
                  <LineChart
                    title="TOTAL NET WORTH"
                    data={metrics.netWorthHistory || []}
                    color="#a855f7" // Purple
                    minScale={0}
                    suffix="M"
                    formatValue={v => `$${v.toFixed(2)}`}
                  />

                  {/* 5. HEALTH INDEX */}
                  <LineChart
                    title="HEALTH INDEX"
                    data={metrics.healthHistory}
                    color="#ec4899" // Pink
                    minScale={0}
                    suffix="/100"
                    formatValue={v => v.toFixed(1)}
                  />

                  {/* 6. MARKET PRICES (Food, Housing, Services) */}
                  <LineChart
                    title="MARKET PRICES (F/H/S)"
                    data={[
                      metrics.priceHistory?.food || [],
                      metrics.priceHistory?.housing || [],
                      metrics.priceHistory?.services || []
                    ]}
                    color={["#d97706", "#10b981", "#06b6d4"]} // Amber, Emerald, Cyan
                    minScale={0}
                    suffix=""
                    formatValue={v => `$${v.toFixed(2)}`}
                  />

                  {/* 7. MARKET SUPPLY (Food, Housing, Services) */}
                  <LineChart
                    title="TOTAL SUPPLY (F/H/S)"
                    data={[
                      metrics.supplyHistory?.food || [],
                      metrics.supplyHistory?.housing || [],
                      metrics.supplyHistory?.services || []
                    ]}
                    color={["#d97706", "#10b981", "#06b6d4"]} // Amber, Emerald, Cyan
                    minScale={0}
                    suffix=""
                    formatValue={v => Math.floor(v)}
                  />

                  {/* 8. FISCAL BALANCE (Profit) */}
                  <LineChart
                    title="FISCAL BALANCE (PROFIT)"
                    data={metrics.govProfitHistory}
                    color="#8b5cf6" // Violet
                    minScale={-5}
                    suffix="M"
                    formatValue={v => `$${v.toFixed(2)}`}
                  />

                  {/* 9. FIRM COUNT */}
                  <LineChart
                    title="ACTIVE FIRMS"
                    data={metrics.firmCountHistory}
                    color="#64748b" // Slate
                    minScale={0}
                    suffix=""
                    formatValue={v => Math.floor(v)}
                  />
                </div>
              </div>

              {/* SYSTEM ADVISORY FOOTER */}
              <div className="col-span-12">
                <div className="tech-panel p-3 border-l-2 border-amber-500 bg-amber-500/5 flex justify-between items-center">
                  <div className="flex items-start space-x-2">
                    <Zap className="text-amber-500 shrink-0 mt-0.5" size={14} />
                    <div>
                      <h4 className="text-amber-400 font-bold text-xs">SYSTEM ADVISORY</h4>
                      <p className="text-[10px] text-slate-400 mt-1 leading-relaxed">
                        Monitor inflation risk. Supply chain nominal.
                      </p>
                    </div>
                  </div>
                  <div className="flex space-x-4 text-xs font-mono text-slate-500">
                    <span>FIRMS: <span className="text-slate-300">{metrics.firmCountHistory && metrics.firmCountHistory.length > 0 ? metrics.firmCountHistory[metrics.firmCountHistory.length - 1].value : setupConfig.num_firms * 3}</span></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* CONFIG VIEW */}
          {activeView === 'CONFIG' && (
            <div className="grid grid-cols-2 gap-8 max-w-4xl mx-auto animate-in fade-in zoom-in-95 duration-300">
              <div className="col-span-2 mb-4">
                <h2 className="text-3xl font-display font-bold text-white mb-2">SIMULATION PARAMETERS</h2>
                <p className="text-slate-500">
                  {isInitialized
                    ? "Adjust macroeconomic variables. Changes apply on next tick cycle."
                    : "Input macroeconomic variables before initializing the physics engine."}
                </p>
              </div>

              {/* System Scale - Only visible during setup */}
              {!isInitialized && (
                <div className="col-span-2 tech-panel p-8 tech-corners mb-4">
                  <div className="flex items-center space-x-3 mb-8 pb-4 border-b border-slate-700/50">
                    <Users className="text-sky-400" />
                    <h3 className="text-xl font-bold text-slate-200">SYSTEM SCALE</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-8">
                    <TechSlider
                      label="Population Scale (Households)"
                      value={setupConfig.num_households}
                      min={0} max={3000} step={100}
                      onChange={v => handleSetupChange('num_households', v)}
                      format={v => v.toLocaleString()}
                    />
                    <TechSlider
                      label="Market Density (Firms/Category)"
                      value={setupConfig.num_firms}
                      min={1} max={50} step={1}
                      onChange={v => handleSetupChange('num_firms', v)}
                      format={v => v}
                    />
                  </div>
                </div>
              )}

              {/* Fiscal Controls */}
              <div className="tech-panel p-8 tech-corners">
                <div className="flex items-center space-x-3 mb-8 pb-4 border-b border-slate-700/50">
                  <Globe className="text-sky-400" />
                  <h3 className="text-xl font-bold text-slate-200">FISCAL POLICY</h3>
                </div>

                <TechSlider
                  label="Wage Tax Rate"
                  value={isInitialized ? config.wageTax : setupConfig.wage_tax}
                  min={0} max={0.5} step={0.01}
                  onChange={v => isInitialized ? handleConfigChange('wageTax', v) : handleSetupChange('wage_tax', v)}
                  format={v => `${(v * 100).toFixed(0)}%`}
                />
                <TechSlider
                  label="Corp Profit Tax"
                  value={isInitialized ? config.profitTax : setupConfig.profit_tax}
                  min={0} max={0.6} step={0.01}
                  onChange={v => isInitialized ? handleConfigChange('profitTax', v) : handleSetupChange('profit_tax', v)}
                  format={v => `${(v * 100).toFixed(0)}%`}
                />
              </div>

              {/* Physics Controls */}
              <div className="tech-panel p-8 tech-corners">
                <div className="flex items-center space-x-3 mb-8 pb-4 border-b border-slate-700/50">
                  <TrendingUp className="text-emerald-400" />
                  <h3 className="text-xl font-bold text-slate-200">MARKET PHYSICS</h3>
                </div>

                <TechSlider
                  label="Hiring Elasticity"
                  value={config.hiringSpeed}
                  min={0.01} max={0.5} step={0.01}
                  onChange={v => handleConfigChange('hiringSpeed', v)}
                  format={v => `x${v.toFixed(2)}`}
                />
                <TechSlider
                  label="Firm Aggression"
                  value={config.firmStrategy}
                  min={0} max={1} step={0.1}
                  onChange={v => handleConfigChange('firmStrategy', v)}
                  format={v => `${v * 100} / 100`}
                />
              </div>

              <div className="col-span-2 flex justify-end space-x-4 mt-6">
                {isInitialized ? (
                  <button className="btn-tech px-8 py-3 flex items-center space-x-2 active bg-sky-500 text-white shadow-lg shadow-sky-500/20">
                    <Save size={18} />
                    <span>UPDATE PARAMS</span>
                  </button>
                ) : (
                  <button
                    onClick={handleInitialize}
                    disabled={isInitializing}
                    className={`btn-tech btn-primary-large w-full py-6 flex items-center justify-center space-x-3 text-lg font-bold tracking-widest ${isInitializing ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isInitializing ? (
                      <>
                        <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                        <span>INITIALIZING PROTOCOL...</span>
                      </>
                    ) : (
                      <>
                        <Zap size={24} />
                        <span>INITIALIZE PROTOCOL</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* LOGS VIEW */}
          {activeView === 'LOGS' && (
            <div className="max-w-5xl mx-auto tech-panel h-[600px] flex flex-col p-0 tech-corners animate-in fade-in duration-300">
              <div className="bg-slate-900/80 p-3 border-b border-slate-700 flex justify-between items-center">
                <span className="font-mono text-sm text-sky-400 flex items-center">
                  <Terminal size={14} className="mr-2" />
                  /var/logs/ecosim_events.log
                </span>
                <span className="text-xs text-slate-500">AUTO-SCROLL: ON</span>
              </div>
              <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-1 bg-black/40">
                {logs.length === 0 && <div className="text-slate-600 italic">No events recorded in current session.</div>}
                {logs.map((log, i) => (
                  <div key={i} className="flex space-x-4 border-b border-slate-800/30 pb-1 mb-1 hover:bg-white/5 p-1 rounded">
                    <span className="text-slate-500 w-16 text-right">{log.tick ? log.tick.toString().padStart(4, '0') : '0000'}</span>
                    <span className={`w-12 font-bold ${log.type === 'WARN' ? 'text-amber-500' :
                      log.type === 'ECO' ? 'text-emerald-500' :
                        log.type === 'GOV' ? 'text-purple-400' :
                          log.type === 'SYS' ? 'text-slate-100' :
                            'text-sky-500'
                      }`}>{log.type}</span>
                    <span className="text-slate-300">{log.txt}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}

        </div>
      </main>

      {/* Background Decor */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-sky-500/5 rounded-full blur-[100px] pointer-events-none -z-10"></div>
    </div>
  );
}
