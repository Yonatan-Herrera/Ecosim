import React, { useState, useEffect, useRef } from 'react';
import NeuralAvatar from './NeuralAvatar';
import NeuralBuilding from './NeuralBuilding';
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
    className={`w-full p-2 flex flex-col items-center justify-center space-y-1 transition-all border-l-2 ${isActive
      ? 'border-sky-500 bg-sky-500/10 text-sky-400'
      : disabled
        ? 'border-transparent text-slate-700 cursor-not-allowed'
        : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
      }`}
  >
    {disabled ? <Lock size={20} className="mb-1 opacity-50" /> : <Icon size={24} />}
    <span className="text-[10px] uppercase font-display tracking-wider">{label}</span>
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

// --- WEALTH INEQUALITY VISUALIZATION ---
const WealthDistributionChart = ({ gini, top10, bottom50 }) => {
  // Reference inequality levels for context
  const references = [
    { label: "Perfect Equality", gini: 0.0, color: "#10b981" },
    { label: "Nordic Countries", gini: 0.27, color: "#84cc16" },
    { label: "Moderate", gini: 0.40, color: "#fbbf24" },
    { label: "USA Level", gini: 0.48, color: "#f97316" },
    { label: "High Inequality", gini: 0.60, color: "#ef4444" },
    { label: "Extreme Crisis", gini: 0.80, color: "#dc2626" }
  ];

  // Determine current state color
  const getCurrentColor = (g) => {
    if (g < 0.30) return "#10b981"; // Green - healthy
    if (g < 0.40) return "#84cc16"; // Light green
    if (g < 0.50) return "#fbbf24"; // Yellow - moderate
    if (g < 0.60) return "#f97316"; // Orange - concerning
    if (g < 0.70) return "#ef4444"; // Red - high
    return "#dc2626"; // Dark red - crisis
  };

  const currentColor = getCurrentColor(gini);
  const giniPercent = (gini * 100).toFixed(1);

  return (
    <div className="flex-1 relative border-l border-b border-slate-700/50 px-4 pb-4 bg-slate-900/20 flex flex-col min-h-[200px]">
      <div className="absolute top-2 left-3 z-10">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Wealth Inequality</h4>
        <div className="text-3xl font-mono font-bold" style={{ color: currentColor }}>
          {gini.toFixed(3)}
        </div>
        <div className="text-xs text-slate-500 mt-1">Gini Coefficient</div>
      </div>

      {/* Visual Bar Distribution */}
      <div className="flex-1 flex items-end justify-center space-x-1 pt-16 pb-6">
        {/* Bottom 50% */}
        <div className="flex flex-col items-center flex-1">
          <div
            className="w-full bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t-sm transition-all duration-500"
            style={{ height: `${Math.max(5, bottom50 * 2)}%` }}
          ></div>
          <div className="text-[10px] text-slate-400 mt-2 text-center">
            <div className="font-bold text-emerald-400">{bottom50.toFixed(1)}%</div>
            <div>Bottom 50%</div>
          </div>
        </div>

        {/* Middle 40% */}
        <div className="flex flex-col items-center flex-1">
          <div
            className="w-full bg-gradient-to-t from-amber-600 to-amber-400 rounded-t-sm transition-all duration-500"
            style={{ height: `${Math.max(10, (100 - top10 - bottom50) * 1.5)}%` }}
          ></div>
          <div className="text-[10px] text-slate-400 mt-2 text-center">
            <div className="font-bold text-amber-400">{(100 - top10 - bottom50).toFixed(1)}%</div>
            <div>Middle 40%</div>
          </div>
        </div>

        {/* Top 10% */}
        <div className="flex flex-col items-center flex-1">
          <div
            className="w-full bg-gradient-to-t from-rose-600 to-rose-400 rounded-t-sm transition-all duration-500 shadow-lg shadow-rose-500/30"
            style={{ height: `${Math.max(15, top10 * 1.2)}%` }}
          ></div>
          <div className="text-[10px] text-slate-400 mt-2 text-center">
            <div className="font-bold text-rose-400">{top10.toFixed(1)}%</div>
            <div>Top 10%</div>
          </div>
        </div>
      </div>

      {/* Reference Scale */}
      <div className="relative h-12 mt-2">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full h-4 bg-gradient-to-r from-emerald-500 via-amber-500 via-orange-500 to-rose-600 rounded-full opacity-30"></div>
        </div>

        {/* Current position indicator */}
        <div
          className="absolute top-0 flex flex-col items-center transition-all duration-500"
          style={{ left: `${gini * 100}%`, transform: 'translateX(-50%)' }}
        >
          <div className="w-1 h-4 bg-white shadow-lg"></div>
          <div className="text-[9px] text-white font-bold mt-1 bg-slate-800 px-2 py-0.5 rounded border border-slate-600 whitespace-nowrap">
            YOU ARE HERE
          </div>
        </div>

        {/* Reference markers */}
        <div className="absolute -bottom-6 left-0 text-[8px] text-emerald-400">Equal</div>
        <div className="absolute -bottom-6 left-1/4 text-[8px] text-lime-400">Nordic</div>
        <div className="absolute -bottom-6 left-1/2 text-[8px] text-amber-400 transform -translate-x-1/2">Moderate</div>
        <div className="absolute -bottom-6 right-1/4 text-[8px] text-orange-400">USA</div>
        <div className="absolute -bottom-6 right-0 text-[8px] text-rose-400">Crisis</div>
      </div>
    </div>
  );
};

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
  let minVal = Math.min(...allValues, minScale);
  let maxVal = Math.max(...allValues, minScale + 0.1);
  const rawRange = maxVal - minVal || 1;
  const padding = Math.max(rawRange * 0.05, 0.5);
  minVal -= padding;
  maxVal += padding;
  const range = maxVal - minVal || 1;

  // Zero line Y position
  const chartHeight = 90;
  const chartOffset = 5;
  const zeroY = 100 - ((0 - minVal) / range) * chartHeight - chartOffset;

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
        {/* SVG Layer - Lines Only */}
        <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
          {/* Grid Lines */}
          {[0, 25, 50, 75, 100].map(p => (
            <line key={p} x1="0" y1={p} x2="100" y2={p} stroke="#1e293b" strokeWidth="0.5" />
          ))}

          {/* Zero Line */}
          {zeroY >= 0 && zeroY <= 100 && (
            <line x1="0" y1={zeroY} x2="100" y2={zeroY} stroke="#475569" strokeWidth="0.5" strokeDasharray="2 2" />
          )}

          {/* Render Lines */}
          {datasets.map((dataset, dIdx) => {
            const lineColor = colors[dIdx % colors.length];
            const points = dataset.map((point, i) => {
              const x = (i / (dataset.length - 1)) * 100;
              const y = 100 - ((point.value - minVal) / range) * 50 - 25;
              return `${x},${y}`;
            }).join(' ');

            return (
              <polyline key={dIdx} points={points} fill="none" stroke={lineColor} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            );
          })}
        </svg>

        {/* HTML Overlay Layer - Tooltips & Dots */}
        <div className="absolute inset-0 pointer-events-none">
          {datasets[0].map((point, i) => {
            const x = (i / (datasets[0].length - 1)) * 100;
            const y = 100 - ((point.value - minVal) / range) * 50 - 25;
            const pointColor = colors[0];
            const isTop = y < 50;

            return (
              <div
                key={i}
                className="absolute group flex items-center justify-center"
                style={{ left: `${x}%`, top: `${y}%`, width: 0, height: 0 }}
              >
                {/* Hit Area */}
                <div className="absolute w-6 h-6 bg-transparent cursor-crosshair pointer-events-auto z-10 -translate-x-1/2 -translate-y-1/2"></div>

                {/* Visible Dot */}
                <div
                  className="absolute w-2 h-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-[0_0_8px_currentColor] -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                  style={{ backgroundColor: pointColor, color: pointColor }}
                ></div>

                {/* Tooltip */}
                <div className={`absolute ${isTop ? 'top-3' : 'bottom-3'} bg-slate-900/90 backdrop-blur border border-slate-600 rounded px-1.5 py-0.5 text-[9px] font-mono text-white whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none shadow-xl -translate-x-1/2`}>
                  {formatValue(point.value)}{suffix}
                </div>
              </div>
            );
          })}
        </div>
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
    giniCoefficient: 0.0,
    top10Share: 0.0,
    bottom50Share: 0.0,
    gdpHistory: [],
    unemploymentHistory: [],
    wageHistory: [],
    medianWageHistory: [],
    happinessHistory: [],
    healthHistory: [],
    govProfitHistory: [],
    govDebtHistory: [],
    firmCountHistory: [],
    giniHistory: [],
    top10ShareHistory: [],
    bottom50ShareHistory: [],
    priceHistory: { food: [], housing: [], services: [] },
    supplyHistory: { food: [], housing: [], services: [] },
    trackedSubjects: [],
    trackedFirms: []
  });

  const [activeSubjectIndex, setActiveSubjectIndex] = useState(0);
  const [activeFirmIndex, setActiveFirmIndex] = useState(0);
  const [firmStats, setFirmStats] = useState(null);

  const [config, setConfig] = useState({
    wageTax: 0.05,
    profitTax: 0.30,
    inflationRate: 0.02,
    birthRate: 0.01,
    minimumWage: 20.0,
    unemploymentBenefitRate: 0.4,
    universalBasicIncome: 0.0,
    wealthTaxThreshold: 50000,
    wealthTaxRate: 0.0
  });

  // Setup State (for initialization)
  const [setupConfig, setSetupConfig] = useState({
    num_households: 1000,
    num_firms: 5,
    wage_tax: 0.15,
    profit_tax: 0.20,
    disable_stabilizers: false,
    disabled_agents: []
  });
  const setupConfigRef = useRef(setupConfig);
  useEffect(() => {
    setupConfigRef.current = setupConfig;
  }, [setupConfig]);
  const stabilizerAgentOptions = [
    { key: 'households', label: 'Households' },
    { key: 'firms', label: 'Firms' },
    { key: 'government', label: 'Government' },
    { key: 'all', label: 'All Agents' }
  ];
  const subjectCount = metrics.trackedSubjects ? metrics.trackedSubjects.length : 0;
  const firmCount = metrics.trackedFirms ? metrics.trackedFirms.length : 0;

  useEffect(() => {
    if (subjectCount === 0 && activeSubjectIndex !== 0) {
      setActiveSubjectIndex(0);
    } else if (subjectCount > 0 && activeSubjectIndex >= subjectCount) {
      setActiveSubjectIndex(0);
    }
  }, [subjectCount, activeSubjectIndex]);

  useEffect(() => {
    if (firmCount === 0 && activeFirmIndex !== 0) {
      setActiveFirmIndex(0);
    } else if (firmCount > 0 && activeFirmIndex >= firmCount) {
      setActiveFirmIndex(0);
    }
  }, [firmCount, activeFirmIndex]);

  const formatCurrency = (value, decimals = 0) => {
    const num = Number(value || 0);
    return `$${num.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    })}`;
  };

  const formatCompact = (value) => Number(value || 0).toLocaleString();
  const selectedTrackedFirm = (metrics.trackedFirms && metrics.trackedFirms.length > 0 && metrics.trackedFirms[activeFirmIndex])
    ? metrics.trackedFirms[activeFirmIndex]
    : null;

  const renderFirmTable = (title, rows) => (
    <div className="tech-panel p-4 tech-corners">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-bold tracking-widest uppercase text-slate-300">{title}</h3>
        <span className="text-[10px] text-slate-500">{rows && rows.length ? rows.length : 0} tracked</span>
      </div>
      <div className="overflow-x-auto -mx-2">
        <table className="w-full text-[11px] text-slate-300 mx-2">
          <thead className="text-[9px] uppercase text-slate-500">
            <tr>
              <th className="text-left pb-1">Firm</th>
              <th className="text-left pb-1">Cat</th>
              <th className="text-right pb-1">Cash</th>
              <th className="text-right pb-1">Emp</th>
              <th className="text-right pb-1">Price</th>
              <th className="text-right pb-1">Wage</th>
              <th className="text-right pb-1">Profit</th>
            </tr>
          </thead>
          <tbody>
            {rows && rows.length ? rows.slice(0, 8).map(row => (
              <tr key={row.id} className="border-t border-slate-800/60">
                <td className="py-1 pr-2 font-display text-xs">{row.name}</td>
                <td className="py-1 pr-2 text-slate-500">{row.category}</td>
                <td className="py-1 pr-2 text-right">{formatCurrency(row.cash)}</td>
                <td className="py-1 pr-2 text-right">{row.employees}</td>
                <td className="py-1 pr-2 text-right">{formatCurrency(row.price, 2)}</td>
                <td className="py-1 pr-2 text-right">{formatCurrency(row.wageOffer, 2)}</td>
                <td className="py-1 pl-2 text-right">{formatCurrency(row.lastProfit, 2)}</td>
              </tr>
            )) : (
              <tr>
                <td colSpan={7} className="py-2 text-center text-slate-500 text-xs">No data yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );

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
        const cfg = setupConfigRef.current;
        // Sync local config with setup
        setConfig(prev => ({
          ...prev,
          wageTax: cfg.wage_tax,
          profitTax: cfg.profit_tax
        }));
        // Add boot sequence logs
        setLogs([
          { tick: 0, type: 'SYS', txt: 'INITIALIZING KERNEL...' },
          { tick: 0, type: 'SYS', txt: 'LOADING CONFIGURATION MAP...' },
          { tick: 0, type: 'SYS', txt: `SPAWNING ${cfg.num_households} AGENTS...` },
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
          giniCoefficient: 0.0,
          top10Share: 0.0,
          bottom50Share: 0.0,
          gdpHistory: [],
          unemploymentHistory: [],
          wageHistory: [],
          medianWageHistory: [],
          happinessHistory: [],
          healthHistory: [],
          govProfitHistory: [],
          govDebtHistory: [],
          giniHistory: [],
          top10ShareHistory: [],
          bottom50ShareHistory: [],
          housingHistory: [],
          foodHistory: [],
          servicesHistory: [],
          priceHistory: { food: [], housing: [], services: [] },
          supplyHistory: { food: [], housing: [], services: [] },
          trackedSubjects: [],
          trackedFirms: []
        });
        setActiveSubjectIndex(0);
        setActiveFirmIndex(0);
        setFirmStats(null);
        setIsRunning(false);
        setIsInitialized(false);
        setActiveView('CONFIG'); // Go back to config on reset
      } else if (data.type === "STABILIZERS_UPDATED") {
        console.log("Stabilizers updated:", data.state);
      } else if (data.metrics) {
        setTick(data.tick);
        // Merge with existing metrics to preserve defaults if backend is missing keys
        setMetrics(prev => ({
          ...prev,
          ...data.metrics,
          // Ensure nested objects/arrays are not overwritten with undefined if missing
          priceHistory: data.metrics.priceHistory || prev.priceHistory || { food: [], housing: [], services: [] },
          supplyHistory: data.metrics.supplyHistory || prev.supplyHistory || { food: [], housing: [], services: [] },
          netWorthHistory: data.metrics.netWorthHistory || prev.netWorthHistory || [],
          trackedSubjects: data.metrics.trackedSubjects || prev.trackedSubjects || [],
          trackedFirms: data.metrics.trackedFirms || prev.trackedFirms || []
        }));
        if (data.firm_stats) {
          setFirmStats(data.firm_stats);
        }
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
  }, []);

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
  const sendStabilizerCommand = (disableFlag, disabledAgents) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        command: "STABILIZERS",
        disable_stabilizers: disableFlag,
        disabled_agents: disabledAgents
      }));
    }
  };

  const handleSetupChange = (key, value) => {
    setSetupConfig(prev => {
      const next = { ...prev, [key]: value };
      if (key === 'disable_stabilizers' && value === false) {
        next.disabled_agents = [];
      }
      if (isInitialized && (key === 'disable_stabilizers')) {
        sendStabilizerCommand(next.disable_stabilizers, next.disabled_agents);
      }
      return next;
    });
    // Also update the runtime config preview
    if (key === 'wage_tax') setConfig(prev => ({ ...prev, wageTax: value }));
    if (key === 'profit_tax') setConfig(prev => ({ ...prev, profitTax: value }));
  };

  const toggleStabilizerAgent = (agentKey) => {
    setSetupConfig(prev => {
      const disabled = prev.disabled_agents || [];
      const exists = disabled.includes(agentKey);
      const updated = exists ? disabled.filter(a => a !== agentKey) : [...disabled, agentKey];
      const next = { ...prev, disabled_agents: updated };
      if (isInitialized) {
        sendStabilizerCommand(next.disable_stabilizers, next.disabled_agents);
      }
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-black text-slate-300 font-display selection:bg-sky-500/30 overflow-hidden flex">
      <style>{techStyles}</style>

      {/* SIDEBAR NAVIGATION */}
      <nav className="w-24 bg-slate-900/50 backdrop-blur-md border-r border-slate-800 flex flex-col justify-between z-20">
        <div>
          <div className="h-24 flex items-center justify-center border-b border-slate-800 mb-2">
            <Triangle className="text-sky-500 fill-sky-500/20" size={32} strokeWidth={1.5} />
          </div>
          {/* CONFIG is always active, but others are disabled until initialized */}
          <NavButton icon={Settings} label="Config" isActive={activeView === 'CONFIG'} onClick={() => setActiveView('CONFIG')} />
          <NavButton icon={Activity} label="Dash" isActive={activeView === 'DASHBOARD'} onClick={() => setActiveView('DASHBOARD')} disabled={!isInitialized} />
          <NavButton icon={Users} label="Subjects" isActive={activeView === 'SUBJECTS'} onClick={() => setActiveView('SUBJECTS')} disabled={!isInitialized} />
          <NavButton icon={Building2} label="Firms" isActive={activeView === 'FIRMS'} onClick={() => setActiveView('FIRMS')} disabled={!isInitialized} />
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
              ECO<span className="text-sky-500">SIM</span> // OPEN PROJECT
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

              {/* WEALTH INEQUALITY ROW */}
              <div className="col-span-12 grid grid-cols-3 gap-4 mb-2">
                <StatTile label="Gini Coefficient" value={`${(metrics.giniCoefficient || 0).toFixed(3)}`} suffix="/1.0" />
                <StatTile label="Top 10% Wealth Share" value={`${(metrics.top10Share || 0).toFixed(1)}%`} />
                <StatTile label="Bottom 50% Share" value={`${(metrics.bottom50Share || 0).toFixed(1)}%`} />
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

                  {/* 9. WEALTH INEQUALITY - Visual Distribution */}
                  <WealthDistributionChart
                    gini={metrics.giniCoefficient || 0}
                    top10={metrics.top10Share || 0}
                    bottom50={metrics.bottom50Share || 0}
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

          {/* SUBJECTS VIEW */}
          {activeView === 'SUBJECTS' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 h-full flex flex-col">
              <style>{`
                @keyframes hologram-spin {
                  0% { transform: rotateY(0deg); }
                  100% { transform: rotateY(360deg); }
                }
                .hologram-container {
                  perspective: 1000px;
                }
                .hologram-body {
                  animation: hologram-spin 10s linear infinite;
                  transform-style: preserve-3d;
                }
              `}</style>

              {/* TOP TABS - SUBJECT SELECTION */}
              <div className="flex space-x-2 mb-2 overflow-x-auto pb-1 shrink-0">
                {metrics.trackedSubjects && metrics.trackedSubjects.length > 0 ? (
                  metrics.trackedSubjects.map((subject, idx) => (
                    <button
                      key={subject.id}
                      onClick={() => setActiveSubjectIndex(idx)}
                      className={`flex-1 min-w-[120px] tech-panel p-2 text-left transition-all ${activeSubjectIndex === idx
                        ? 'border-sky-500 bg-sky-500/10'
                        : 'hover:bg-white/5 border-slate-700/50'
                        }`}
                    >
                      <div className="flex justify-between items-start mb-0.5">
                        <span className="text-[10px] font-mono text-slate-500">ID: {subject.id.toString().padStart(4, '0')}</span>
                        <div className={`h-1.5 w-1.5 rounded-full ${subject.state === 'WORKING' ? 'bg-emerald-500 shadow-[0_0_5px_#10b981]' :
                          subject.state === 'SLEEPING' ? 'bg-indigo-500' :
                            subject.state === 'STRESSED' ? 'bg-rose-500' :
                              'bg-amber-500'
                          }`}></div>
                      </div>
                      <div className="font-display font-bold text-xs text-slate-200 truncate">{subject.name}</div>
                      <div className="text-[9px] text-slate-400">{subject.state}</div>
                    </button>
                  ))
                ) : (
                  <div className="text-slate-500 italic p-4">Waiting for subject tracking data...</div>
                )}
              </div>

              {/* MAIN CONTENT GRID */}
              {metrics.trackedSubjects && metrics.trackedSubjects[activeSubjectIndex] && (
                <div className="flex-1 grid grid-cols-12 gap-4 min-h-0 overflow-hidden pb-2">

                  {/* LEFT COLUMN - BIO & EMPLOYMENT */}
                  <div className="col-span-3 flex flex-col space-y-2 overflow-y-auto pr-1 no-scrollbar">
                    {/* ID CARD */}
                    <div className="tech-panel p-2 tech-corners">
                      <h4 className="text-[9px] font-bold text-sky-400 uppercase tracking-widest mb-1 flex items-center">
                        <Users size={10} className="mr-1" /> Bio-Metric
                      </h4>
                      <div className="space-y-1">
                        <div className="flex justify-between items-center border-b border-slate-800 pb-0.5">
                          <span className="text-[9px] text-slate-500">AGE</span>
                          <span className="font-mono text-xs text-slate-200">{metrics.trackedSubjects[activeSubjectIndex].age}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-slate-800 pb-0.5">
                          <span className="text-[9px] text-slate-500">HEALTH</span>
                          <span className={`font-mono text-xs ${(metrics.trackedSubjects[activeSubjectIndex].health || 1) > 0.8 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {((metrics.trackedSubjects[activeSubjectIndex].health || 1) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[9px] text-slate-500">STATUS</span>
                          <span className="font-mono text-xs text-sky-400">{metrics.trackedSubjects[activeSubjectIndex].state}</span>
                        </div>
                      </div>
                    </div>

                    {/* EMPLOYMENT DATA */}
                    <div className="tech-panel p-2 tech-corners">
                      <h4 className="text-[9px] font-bold text-amber-400 uppercase tracking-widest mb-1 flex items-center">
                        <Building2 size={10} className="mr-1" /> Employment
                      </h4>
                      <div className="space-y-2">
                        <div>
                          <div className="text-[9px] text-slate-500 mb-0.5">EMPLOYER</div>
                          <div className="font-display text-sm text-slate-200 truncate">
                            {metrics.trackedSubjects[activeSubjectIndex].employer}
                          </div>
                        </div>
                        <div className="flex justify-between">
                          <div>
                            <div className="text-[9px] text-slate-500 mb-0.5">WAGE</div>
                            <div className="font-mono text-sm text-emerald-400">
                              ${metrics.trackedSubjects[activeSubjectIndex].wage.toFixed(2)}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[9px] text-slate-500 mb-0.5">SHIFT</div>
                            <div className="font-mono text-[10px] text-slate-300">
                              {metrics.trackedSubjects[activeSubjectIndex].state === 'WORKING' ? 'ACTIVE' : 'OFF'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* SKILLS & PERFORMANCE (RESTORED) */}
                    <div className="tech-panel p-2 tech-corners flex-1">
                      <h4 className="text-[9px] font-bold text-cyan-400 uppercase tracking-widest mb-1">Skills</h4>
                      <div className="space-y-2">
                        <div>
                          <div className="flex justify-between items-center mb-0.5">
                            <span className="text-[9px] text-slate-500">LEVEL</span>
                            <span className="font-mono text-xs text-slate-200">
                              {(metrics.trackedSubjects[activeSubjectIndex].skills * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-cyan-500" style={{ width: `${metrics.trackedSubjects[activeSubjectIndex].skills * 100}%` }}></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-0.5">
                            <span className="text-[9px] text-slate-500">MORALE</span>
                            <span className="font-mono text-xs text-slate-200">
                              {(metrics.trackedSubjects[activeSubjectIndex].morale * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-amber-500" style={{ width: `${metrics.trackedSubjects[activeSubjectIndex].morale * 100}%` }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* CENTER COLUMN - VISUALIZER */}
                  <div className="col-span-6 relative flex items-center justify-center overflow-hidden h-full rounded-lg border border-slate-800/50 bg-slate-900/20">

                    {/* Neural Avatar */}
                    <div className="absolute inset-0 z-0">
                      <NeuralAvatar
                        active={true}
                        mood={metrics.trackedSubjects[activeSubjectIndex].happiness > 0.7 ? 'happy' : 'neutral'}
                        variant="human"
                      />
                    </div>

                    {/* Header Overlay (Minimal) */}
                    <div className="absolute top-0 left-0 right-0 p-3 flex justify-between items-start z-10 bg-gradient-to-b from-slate-900/90 to-transparent">
                      <div>
                        <h2 className="text-2xl font-display font-bold text-white drop-shadow-md">
                          {metrics.trackedSubjects[activeSubjectIndex].name}
                        </h2>
                        <div className="text-xs font-mono text-sky-400 mt-0.5">
                          ID: {metrics.trackedSubjects[activeSubjectIndex].id}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-xl font-bold font-display drop-shadow-md ${metrics.trackedSubjects[activeSubjectIndex].state === 'WORKING' ? 'text-emerald-400' : 'text-sky-400'
                          }`}>
                          {metrics.trackedSubjects[activeSubjectIndex].state}
                        </div>
                      </div>
                    </div>

                    {/* Footer Stats (Floating) */}
                    <div className="absolute bottom-4 left-6 right-6 flex justify-between z-10 pointer-events-none">
                      <div className="text-center">
                        <div className="text-[9px] text-slate-400 uppercase tracking-widest mb-0.5">Happiness</div>
                        <div className="text-2xl font-display font-bold text-emerald-400 drop-shadow-md">
                          {((metrics.trackedSubjects[activeSubjectIndex].happiness || 0) * 100).toFixed(0)}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-[9px] text-slate-400 uppercase tracking-widest mb-0.5">Stress</div>
                        <div className="text-2xl font-display font-bold text-amber-400 drop-shadow-md">
                          {((1 - (metrics.trackedSubjects[activeSubjectIndex].happiness || 0)) * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* RIGHT COLUMN - FINANCIALS & NEEDS */}
                  <div className="col-span-3 flex flex-col space-y-2 overflow-y-auto pl-1 no-scrollbar">
                    {/* FINANCIAL HEALTH */}
                    <div className="tech-panel p-2 tech-corners">
                      <h4 className="text-[9px] font-bold text-rose-400 uppercase tracking-widest mb-1 flex items-center">
                        <DollarSign size={10} className="mr-1" /> Finances
                      </h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-end">
                          <span className="text-[9px] text-slate-500">LIQUID</span>
                          <span className="font-mono text-sm text-white">
                            ${metrics.trackedSubjects[activeSubjectIndex].cash.toFixed(0)}
                          </span>
                        </div>
                        <div className="flex justify-between items-end">
                          <span className="text-[9px] text-slate-500">NET WORTH</span>
                          <span className="font-mono text-sm text-purple-400">
                            ${metrics.trackedSubjects[activeSubjectIndex].netWorth.toFixed(0)}
                          </span>
                        </div>
                        {/* MEDICAL DEBT (RESTORED) */}
                        {metrics.trackedSubjects[activeSubjectIndex].medicalDebt > 0 && (
                          <div className="flex justify-between items-end">
                            <span className="text-[9px] text-slate-500">DEBT</span>
                            <span className="font-mono text-sm text-rose-400">
                              ${metrics.trackedSubjects[activeSubjectIndex].medicalDebt.toFixed(0)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* CHARTS (RESTORED) */}
                    <div className="tech-panel p-2 tech-corners flex-1 flex flex-col min-h-0">
                      <h4 className="text-[9px] font-bold text-sky-400 uppercase tracking-widest mb-1 shrink-0">Wealth</h4>
                      {metrics.trackedSubjects[activeSubjectIndex].history && metrics.trackedSubjects[activeSubjectIndex].history.cash.length > 1 ? (
                        <div className="flex-1 min-h-0 relative">
                          <div className="absolute inset-0">
                            <LineChart
                              title=""
                              data={metrics.trackedSubjects[activeSubjectIndex].history.cash}
                              color="#10b981"
                              minScale={0}
                              suffix=""
                              formatValue={v => `${v.toFixed(0)}`}
                            />
                          </div>
                        </div>
                      ) : <div className="text-[9px] text-slate-600 italic">No history</div>}
                    </div>

                    <div className="tech-panel p-2 tech-corners flex-1 flex flex-col min-h-0">
                      <h4 className="text-[9px] font-bold text-amber-400 uppercase tracking-widest mb-1 shrink-0">Wage</h4>
                      {metrics.trackedSubjects[activeSubjectIndex].history && metrics.trackedSubjects[activeSubjectIndex].history.wage.length > 1 ? (
                        <div className="flex-1 min-h-0 relative">
                          <div className="absolute inset-0">
                            <LineChart
                              title=""
                              data={metrics.trackedSubjects[activeSubjectIndex].history.wage}
                              color="#f59e0b"
                              minScale={0}
                              suffix=""
                              formatValue={v => `${v.toFixed(0)}`}
                            />
                          </div>
                        </div>
                      ) : <div className="text-[9px] text-slate-600 italic">No history</div>}
                    </div>

                    {/* INVENTORY */}
                    <div className="tech-panel p-2 tech-corners">
                      <h4 className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-1">Inventory</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] text-slate-500">FOOD</span>
                        <span className="font-mono text-xs text-slate-300">
                          {metrics.trackedSubjects[activeSubjectIndex].needs?.food?.toFixed(0)}
                        </span>
                      </div>
                    </div>
                  </div>

                </div>
              )}
            </div>
          )}

          {/* FIRMS VIEW */}
          {activeView === 'FIRMS' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <style>{`
                @keyframes hologram-spin {
                  0% { transform: rotateY(0deg); }
                  100% { transform: rotateY(360deg); }
                }
                .hologram-container {
                  perspective: 1000px;
                }
                .hologram-body {
                  animation: hologram-spin 18s linear infinite;
                  transform-style: preserve-3d;
                }
              `}</style>
              {!firmStats ? (
                <div className="tech-panel p-6 text-center text-slate-500 text-sm">
                  Awaiting firm telemetry...
                </div>
              ) : (
                <div className="grid grid-cols-12 gap-4 h-[calc(100vh-220px)] min-h-[620px]">
                  <div className="col-span-8 flex flex-col h-full space-y-4">
                    <div className="grid grid-cols-4 gap-4 shrink-0">
                      <StatTile label="Total Firms" value={formatCompact(firmStats.total_firms)} />
                      <StatTile label="Total Employees" value={formatCompact(firmStats.total_employees)} />
                      <StatTile label="Avg Wage Offer" value={formatCurrency(firmStats.avg_wage_offer || 0, 2)} />
                      <StatTile label="Struggling Firms" value={formatCompact(firmStats.struggling_firms || 0)} />
                    </div>

                    <div className="flex flex-col flex-1 min-h-0 space-y-4">
                      <div className="tech-panel tech-corners relative flex-1 min-h-[14rem] overflow-hidden">
                        <div className="absolute top-4 left-4 z-10">
                          <div className="text-[10px] uppercase text-slate-500 tracking-widest">Market Mood</div>
                          <div className="text-xl font-display text-white">
                            {firmStats.struggling_firms > 0.15 * firmStats.total_firms ? 'VOLATILE' : 'STABLE'}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            Avg price {formatCurrency(firmStats.avg_price || 0, 2)} | Avg quality {(firmStats.avg_quality || 0).toFixed(2)}
                          </div>
                        </div>
                        <div className="absolute top-4 right-4 text-right text-[10px] text-slate-500 z-10">
                          {firmStats.market_sentiment || 'Calm winds'}
                        </div>
                        <div className="absolute inset-0 flex items-center justify-center hologram-container pointer-events-none px-6">
                          <div className="w-full h-full max-w-full">
                            <NeuralBuilding
                              active
                              activityLevel={firmStats.struggling_firms > 0.15 * firmStats.total_firms ? 'high' : 'normal'}
                              tier={Math.min(3, Math.max(1, Math.round((firmStats.total_firms || 1) / 100)))}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-4">
                        <div className="tech-panel p-4 tech-corners">
                          <div className="flex justify-between items-center mb-3">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300">Sector Breakdown</h3>
                            <span className="text-[10px] text-slate-500">Avg price {formatCurrency(firmStats.avg_price || 0, 2)}</span>
                          </div>
                          {firmStats.categories && firmStats.categories.length ? (
                            <div className="grid grid-cols-3 gap-3">
                              {firmStats.categories.map(cat => (
                                <div key={cat.category} className="border border-slate-800 rounded-md p-3 bg-slate-900/30">
                                  <div className="text-xs font-display text-slate-200">{cat.category}</div>
                                  <div className="text-[10px] text-slate-500 mb-2">{cat.firm_count} firms</div>
                                  <div className="text-[11px] text-slate-400">Employees: <span className="text-slate-200">{formatCompact(cat.total_employees)}</span></div>
                                  <div className="text-[11px] text-slate-400">Avg Cash: <span className="text-slate-200">{formatCurrency(cat.avg_cash || 0)}</span></div>
                                  <div className="text-[11px] text-slate-400">Avg Price: <span className="text-slate-200">{formatCurrency(cat.avg_price || 0, 2)}</span></div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-slate-500 text-xs">No category data yet.</div>
                          )}
                        </div>

                        <div className="grid grid-cols-2 gap-4 pb-2">
                          {renderFirmTable("Top Cash Positions", firmStats.top_cash || [])}
                          {renderFirmTable("Top Employers", firmStats.top_employers || [])}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="col-span-4 flex flex-col space-y-4 min-h-0 h-full">
                    <div className="tech-panel p-3 tech-corners shrink-0">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-xs uppercase font-bold tracking-widest text-slate-300">Tracked Firms</h3>
                        <span className="text-[10px] text-slate-500">{firmCount} monitored</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {metrics.trackedFirms && metrics.trackedFirms.length ? (
                          metrics.trackedFirms.slice(0, 7).map((firm, idx) => (
                            <button
                              key={firm.id}
                              onClick={() => setActiveFirmIndex(idx)}
                              className={`px-3 py-1 text-[11px] rounded border truncate max-w-[8rem] ${activeFirmIndex === idx ? 'border-sky-500 text-sky-300 bg-sky-500/10' : 'border-slate-700 text-slate-400 hover:bg-white/5'}`}
                            >
                              {firm.name}
                            </button>
                          ))
                        ) : (
                          <div className="text-slate-500 text-xs">Sampling firms...</div>
                        )}
                      </div>
                    </div>

                    {selectedTrackedFirm ? (
                      <>
                        <div className="tech-panel p-4 tech-corners space-y-3 shrink-0">
                          <div className="flex justify-between items-center">
                            <div>
                              <h3 className="text-lg font-display text-white">{selectedTrackedFirm.name}</h3>
                              <div className="text-[11px] text-slate-500">{selectedTrackedFirm.category}</div>
                            </div>
                            <div className={`text-xs font-bold ${selectedTrackedFirm.state === 'DISTRESS' ? 'text-rose-400' : selectedTrackedFirm.state === 'SCALING' ? 'text-emerald-400' : 'text-sky-400'}`}>
                              {selectedTrackedFirm.state}
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Cash</div>
                              <div className="font-mono text-slate-200">{formatCurrency(selectedTrackedFirm.cash)}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Inventory</div>
                              <div className="font-mono text-slate-200">{selectedTrackedFirm.inventory?.toFixed(1)}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Employees</div>
                              <div className="font-mono text-slate-200">{selectedTrackedFirm.employees}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Quality</div>
                              <div className="font-mono text-slate-200">{(selectedTrackedFirm.quality || 0).toFixed(1)}</div>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Price</div>
                              <div className="font-mono text-emerald-400">{formatCurrency(selectedTrackedFirm.price, 2)}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Wage Offer</div>
                              <div className="font-mono text-amber-400">{formatCurrency(selectedTrackedFirm.wageOffer, 2)}</div>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Revenue</div>
                              <div className="font-mono text-slate-200">{formatCurrency(selectedTrackedFirm.lastRevenue, 2)}</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500 uppercase">Profit</div>
                              <div className={`font-mono ${selectedTrackedFirm.lastProfit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {formatCurrency(selectedTrackedFirm.lastProfit, 2)}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex-1 flex flex-col gap-3 min-h-0">
                          <div className="tech-panel p-3 tech-corners flex flex-col flex-1 min-h-[170px]">
                            <div className="text-[10px] font-bold tracking-widest uppercase text-slate-400 mb-2">Cash History</div>
                            {selectedTrackedFirm.history?.cash && selectedTrackedFirm.history.cash.length > 1 ? (
                              <div className="flex-1">
                                <LineChart
                                  title=""
                                  data={selectedTrackedFirm.history.cash}
                                  color="#0ea5e9"
                                  minScale={0}
                                  suffix=""
                                  formatValue={v => `$${v.toFixed(0)}`}
                                />
                              </div>
                            ) : <div className="text-[10px] text-slate-600">More ticks needed for cash history.</div>}
                          </div>
                          <div className="tech-panel p-3 tech-corners flex flex-col flex-1 min-h-[170px]">
                            <div className="text-[10px] font-bold tracking-widest uppercase text-slate-400 mb-2">Profit History</div>
                            {selectedTrackedFirm.history?.profit && selectedTrackedFirm.history.profit.length > 1 ? (
                              <div className="flex-1">
                                <LineChart
                                  title=""
                                  data={selectedTrackedFirm.history.profit}
                                  color="#f87171"
                                  minScale={-1}
                                  suffix=""
                                  formatValue={v => `$${v.toFixed(0)}`}
                                />
                              </div>
                            ) : <div className="text-[10px] text-slate-600">More ticks needed for profit history.</div>}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="tech-panel p-4 text-sm text-slate-500">
                        No tracked firms yet.
                      </div>
                    )}
                  </div>
                </div>
              )}
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
                <TechSlider
                  label="Inflation Rate"
                  value={config.inflationRate}
                  min={0} max={0.10} step={0.001}
                  onChange={v => handleConfigChange('inflationRate', v)}
                  format={v => `${(v * 100).toFixed(1)}% annual`}
                />
                <TechSlider
                  label="Birth Rate"
                  value={config.birthRate}
                  min={0} max={0.05} step={0.001}
                  onChange={v => handleConfigChange('birthRate', v)}
                  format={v => `${(v * 100).toFixed(1)}% per 36 ticks`}
                />
              </div>

              {/* Social Policy Controls */}
              <div className="tech-panel p-8 tech-corners">
                <div className="flex items-center space-x-3 mb-8 pb-4 border-b border-slate-700/50">
                  <Users className="text-emerald-400" />
                  <h3 className="text-xl font-bold text-slate-200">SOCIAL POLICY</h3>
                </div>

                <TechSlider
                  label="Minimum Wage Floor"
                  value={config.minimumWage}
                  min={0} max={100} step={1}
                  onChange={v => handleConfigChange('minimumWage', v)}
                  format={v => `$${v.toFixed(0)}`}
                />
                <TechSlider
                  label="Unemployment Benefits"
                  value={config.unemploymentBenefitRate}
                  min={0} max={1.0} step={0.05}
                  onChange={v => handleConfigChange('unemploymentBenefitRate', v)}
                  format={v => `${(v * 100).toFixed(0)}% of avg wage`}
                />
                <TechSlider
                  label="Universal Basic Income"
                  value={config.universalBasicIncome}
                  min={0} max={50} step={1}
                  onChange={v => handleConfigChange('universalBasicIncome', v)}
                  format={v => `$${v.toFixed(0)}/tick`}
                />
                <TechSlider
                  label="Wealth Tax Threshold"
                  value={config.wealthTaxThreshold}
                  min={10000} max={200000} step={10000}
                  onChange={v => handleConfigChange('wealthTaxThreshold', v)}
                  format={v => `$${(v/1000).toFixed(0)}k`}
                />
                <TechSlider
                  label="Wealth Tax Rate"
                  value={config.wealthTaxRate}
                  min={0} max={0.5} step={0.01}
                  onChange={v => handleConfigChange('wealthTaxRate', v)}
                  format={v => `${(v * 100).toFixed(0)}% above threshold`}
                />
              </div>

              {/* Stabilization Sandbox */}
              <div className="col-span-2 tech-panel p-8 tech-corners">
                <div className="flex items-center space-x-3 mb-6 pb-4 border-b border-slate-700/50">
                  <Activity className="text-rose-400" />
                  <h3 className="text-xl font-bold text-slate-200">STABILIZATION SANDBOX</h3>
                </div>
                <label className="flex items-center space-x-3 text-slate-300 text-sm font-display tracking-wide">
                  <input
                    type="checkbox"
                    checked={setupConfig.disable_stabilizers}
                    onChange={(e) => handleSetupChange('disable_stabilizers', e.target.checked)}
                    className="form-checkbox h-4 w-4 text-sky-500"
                  />
                  <span>Disable automatic stabilizers for selected agents</span>
                </label>
                <p className="text-xs text-slate-500 mt-2">
                  Use this to observe raw policy effects without safety nets. When enabled, choose which agents stop smoothing their decisions.
                </p>
                {setupConfig.disable_stabilizers && (
                  <div className="grid grid-cols-2 gap-3 mt-6">
                    {stabilizerAgentOptions.map(opt => {
                      const active = (setupConfig.disabled_agents || []).includes(opt.key);
                      return (
                        <button
                          type="button"
                          key={opt.key}
                          onClick={() => toggleStabilizerAgent(opt.key)}
                          className={`btn-tech px-4 py-2 text-sm ${active ? 'active' : ''}`}
                        >
                          {opt.label}
                        </button>
                      );
                    })}
                  </div>
                )}
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
      </main >

      {/* Background Decor */}
      < div className="absolute top-0 right-0 w-[500px] h-[500px] bg-sky-500/5 rounded-full blur-[100px] pointer-events-none -z-10" ></div >
    </div >
  );
}
