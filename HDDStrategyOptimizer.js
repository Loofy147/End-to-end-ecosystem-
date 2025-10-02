import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const HDDStrategyOptimizer = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [bestStrategy, setBestStrategy] = useState(null);
  const [equityCurve, setEquityCurve] = useState([]);
  const [logs, setLogs] = useState([]);
  const [currentGeneration, setCurrentGeneration] = useState(0);

  const DAYS = 450; // 200 + 100 + 150 for the perfect market storm
  const POP = 20;
  const GEN = 5;
  const ELITE_FRAC = 0.15;

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-9), `[${timestamp}] ${message}`]);
  };

  const normalRandom = () => {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z0;
  };

  const rollingMean = (arr, n) => {
    const result = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      if (i < n - 1) {
        result[i] = NaN;
      } else {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += arr[i - j];
        }
        result[i] = sum / n;
      }
    }
    return result;
  };

  const computeRSI = (arr, n = 14) => {
    const result = new Array(arr.length);
    result[0] = 50;
    let avgGain = 0;
    let avgLoss = 0;
    for (let i = 1; i <= n && i < arr.length; i++) {
      const change = arr[i] - arr[i-1];
      if (change > 0) avgGain += change;
      else avgLoss += Math.abs(change);
    }
    avgGain /= n;
    avgLoss /= n;
    for (let i = 1; i < arr.length; i++) {
      if (i <= n) {
        result[i] = 50;
        continue;
      }
      const change = arr[i] - arr[i-1];
      const gain = change > 0 ? change : 0;
      const loss = change < 0 ? Math.abs(change) : 0;
      avgGain = ((avgGain * (n-1)) + gain) / n;
      avgLoss = ((avgLoss * (n-1)) + loss) / n;
      const rs = avgGain / (avgLoss + 1e-12);
      result[i] = 100 - (100 / (1 + rs));
    }
    return result;
  };

  const computeATR = (high, low, close, n = 14) => {
    const tr = new Array(high.length);
    tr[0] = high[0] - low[0];
    for (let i = 1; i < high.length; i++) {
      const hl = high[i] - low[i];
      const hc = Math.abs(high[i] - close[i-1]);
      const lc = Math.abs(low[i] - close[i-1]);
      tr[i] = Math.max(hl, Math.max(hc, lc));
    }
    return rollingMean(tr, n);
  };

  const generateCandidate = () => {
    const components_pool = ["sma_cross", "rsi_filter", "momentum", "atr_stop", "take_profit", "volatility_filter", "ml_signal"];
    
    const core_signals = ["sma_cross", "rsi_filter", "momentum", "ml_signal"];
    const risk_management = ["atr_stop", "take_profit", "volatility_filter"];
    
    const k = Math.floor(Math.random() * 3) + 2; // 2-4 components
    const comps = [];
    
    const coreChoice = core_signals[Math.floor(Math.random() * core_signals.length)];
    comps.push(coreChoice);
    
    if (Math.random() < 0.7) {
      const riskChoice = risk_management[Math.floor(Math.random() * risk_management.length)];
      comps.push(riskChoice);
    }
    
    const remaining_pool = components_pool.filter(c => !comps.includes(c));
    while (comps.length < k && remaining_pool.length > 0) {
      const idx = Math.floor(Math.random() * remaining_pool.length);
      comps.push(remaining_pool.splice(idx, 1)[0]);
    }
    
    const params = {};
    
    if (comps.includes("sma_cross")) {
      const short_options = [5, 8, 10, 12, 15];
      const long_options = [20, 30, 50, 100];
      params.sma_short = short_options[Math.floor(Math.random() * short_options.length)];
      params.sma_long = long_options[Math.floor(Math.random() * long_options.length)];
      if (params.sma_short >= params.sma_long) {
        params.sma_short = Math.max(3, Math.floor(params.sma_long / 3));
      }
    }
    if (comps.includes("rsi_filter")) {
      params.rsi_buy = 25 + Math.floor(Math.random() * 20);
      params.rsi_sell = 60 + Math.floor(Math.random() * 15);
      if (params.rsi_sell - params.rsi_buy < 15) {
        params.rsi_sell = params.rsi_buy + 20;
      }
    }
    if (comps.includes("momentum")) {
      params.mom_period = [3, 5, 7, 10][Math.floor(Math.random() * 4)];
      params.mom_thresh = [0.001, 0.002, 0.003, 0.005, 0.008][Math.floor(Math.random() * 5)];
    }
    if (comps.includes("atr_stop")) {
      params.atr_mult = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5][Math.floor(Math.random() * 6)];
    }
    if (comps.includes("take_profit")) {
      params.tp_mult = [1.5, 2.0, 2.5, 3.0][Math.floor(Math.random() * 4)];
    }
    if (comps.includes("volatility_filter")) {
      params.vol_threshold = [0.001, 0.0015, 0.002, 0.003][Math.floor(Math.random() * 4)];
    }

    if (comps.includes("ml_signal")) {
      params.ml_buy_threshold = (0.6 + Math.random() * 0.2).toFixed(2);
      
      const model_layers = [];
      const num_conv_layers = Math.random() < 0.6 ? 1 : 2;
      let current_channels = 1;

      for (let i = 0; i < num_conv_layers; i++) {
        const out_channels = [4, 8, 12][Math.floor(Math.random() * 3)];
        model_layers.push({
          type: 'conv1d',
          in_channels: current_channels,
          out_channels: out_channels,
          kernel_size: [3, 5][Math.floor(Math.random() * 2)]
        });
        model_layers.push({ type: 'relu' });
        current_channels = out_channels;
      }
      model_layers.push({ type: 'flatten' });
      
      const num_linear_layers = Math.random() < 0.7 ? 1 : 2;
      for (let i = 0; i < num_linear_layers; i++) {
          model_layers.push({ type: 'linear', out_features: [16, 24][Math.floor(Math.random() * 2)] });
          model_layers.push({ type: 'relu' });
      }
      model_layers.push({ type: 'linear', out_features: 2 });

      params.model_config = {
        window_size: [15, 20, 25][Math.floor(Math.random() * 3)],
        learning_rate: [0.01, 0.005][Math.floor(Math.random() * 2)],
        epochs: 50,
        layers: model_layers
      };
    }
    
    params.position_size = [0.05, 0.1, 0.15][Math.floor(Math.random() * 3)];
    
    return { components: comps, params: params };
  };

  const backtestStrategy = (candidate, priceData, indicators, modelPredictions, initial_equity = 100000) => {
    try {
      const { close_arr, open_arr, high_arr, low_arr, sigma2 } = priceData;
      const { sma_cache, rsi_arr, atr_arr, fft_std } = indicators;
      const { components, params } = candidate;
      
      if (!close_arr || close_arr.length === 0 || !components || components.length === 0) {
        throw new Error("Invalid input data");
      }
      
      let equity = initial_equity;
      let position = 0;
      let entry_price = 0;
      let in_pos = false;
      const trades = [];
      const equity_curve = [];
      
      const commission = 0.0005;
      const slippage = 0.0005;
    
      for (let i = 0; i < DAYS; i++) {
        if (i >= close_arr.length) break;
        
        const openp = open_arr[i] || close_arr[i];
        const closep = close_arr[i];
        const highp = high_arr[i] || closep;
        const lowp = low_arr[i] || closep;
        
        let buy = false;
        let sell = false;
        
        try {
          if (components.includes("sma_cross")) {
            const s = params.sma_short || 5;
            const l = params.sma_long || 50;
            if (sma_cache[s] && sma_cache[l] && i < sma_cache[s].length && i < sma_cache[l].length) {
              const sma_s = sma_cache[s][i];
              const sma_l = sma_cache[l][i];
              if (!isNaN(sma_s) && !isNaN(sma_l) && sma_s > 0 && sma_l > 0) {
                if (sma_s > sma_l) buy = true;
                else if (sma_s < sma_l * 0.995) sell = true;
              }
            }
          }
          if (components.includes("rsi_filter")) {
            const r = (i < rsi_arr.length) ? rsi_arr[i] : 50;
            if (!isNaN(r) && r >= 0 && r <= 100) {
              const rsi_buy_level = params.rsi_buy || 40;
              const rsi_sell_level = params.rsi_sell || 65;
              if (r < rsi_buy_level) buy = buy || true;
              if (r > rsi_sell_level) sell = true;
            }
          }
          if (components.includes("momentum")) {
            const p = Math.max(1, Math.min(params.mom_period || 5, i));
            const thresh = Math.abs(params.mom_thresh || 0.002);
            if (i >= p && close_arr[i-p] > 0) {
              const mom = (closep - close_arr[i-p]) / close_arr[i-p];
              if (!isNaN(mom)) {
                if (mom > thresh) buy = buy || true;
                if (mom < -thresh) sell = true;
              }
            }
          }
          if (components.includes("volatility_filter")) {
            const vol = Math.sqrt(Math.max(1e-12, sigma2[Math.min(i, sigma2.length-1)]));
            const threshold = params.vol_threshold || 0.002;
            if (vol > threshold) buy = false;
          }

          if (components.includes("ml_signal")) {
              const prediction = modelPredictions[i];
              const buy_threshold = params.ml_buy_threshold || 0.7;
              
              if (prediction > buy_threshold) {
                  buy = buy || true;
              }
              if (prediction < 0.4) {
                  sell = true;
              }
          }
          
        } catch (error) {
          console.warn(`Error in signal generation at day ${i}:`, error);
        }
        
        if (!in_pos && buy && equity > 1000) {
            const pos_size = Math.min(0.2, Math.max(0.01, params.position_size || 0.1));
            entry_price = openp * (1 + slippage);
            const pos_value = equity * pos_size;
            const units = pos_value / entry_price;
            if (units > 0 && entry_price > 0) {
              equity -= pos_value * commission;
              position = units;
              in_pos = true;
              trades.push({ type: "buy", day: i, price: entry_price, units: units, equity: equity });
            }
        } else if (in_pos && position > 0) {
            let exit_flag = false;
            let exit_price = null;
            try {
              if (components.includes("atr_stop") && i < atr_arr.length) {
                const atr_val = atr_arr[i];
                if (!isNaN(atr_val) && atr_val > 0 && entry_price > 0) {
                  const atr_mult = Math.min(5.0, Math.max(0.5, params.atr_mult || 2.0));
                  const stop_price = entry_price * (1 - atr_mult * (atr_val / entry_price));
                  if (lowp <= stop_price) {
                    exit_flag = true;
                    exit_price = Math.max(stop_price * (1 - slippage), lowp * 0.99);
                  }
                }
              }
              if (components.includes("take_profit") && !exit_flag) {
                let tp;
                if (i < atr_arr.length && !isNaN(atr_arr[i]) && atr_arr[i] > 0) {
                  const tp_mult = Math.min(5.0, Math.max(1.0, params.tp_mult || 2.0));
                  tp = entry_price * (1 + tp_mult * (atr_arr[i] / entry_price));
                } else {
                  const tp_mult = Math.min(5.0, Math.max(1.0, params.tp_mult || 2.0));
                  tp = entry_price * (1 + tp_mult * 0.01);
                }
                if (highp >= tp) {
                  exit_flag = true;
                  exit_price = Math.min(tp * (1 - slippage), highp * 0.99);
                }
              }
              if (sell && !exit_flag) {
                exit_flag = true;
                exit_price = openp * (1 - slippage);
              }
              if (!exit_flag && entry_price > 0) {
                const unrealized_loss = (entry_price - closep) / entry_price;
                if (unrealized_loss > 0.15) {
                  exit_flag = true;
                  exit_price = closep * (1 - slippage);
                }
              }
            } catch (error) {
              console.warn(`Error in exit logic at day ${i}:`, error);
              exit_flag = true;
              exit_price = openp * (1 - slippage);
            }
            if (exit_flag && position > 0) {
              const final_exit_price = Math.max(0.01, exit_price || (openp * (1 - slippage)));
              const proceeds = position * final_exit_price;
              equity += proceeds - proceeds * commission;
              trades.push({ type: "sell", day: i, price: final_exit_price, units: position, equity: equity });
              position = 0;
              in_pos = false;
            }
        }
        equity_curve.push(Math.max(0, equity));
      }
      
      if (in_pos && position > 0) {
        const final_price = Math.max(0.01, close_arr[DAYS-1] * (1 - slippage));
        const proceeds = position * final_price;
        equity += proceeds - proceeds * commission;
        trades.push({ type: "sell", day: DAYS-1, price: final_price, units: position, equity: equity });
      }
      const final_equity = Math.max(0, equity);
      const total_return = final_equity / initial_equity - 1.0;
      const years = DAYS / 252.0;
      let cagr = 0;
      if (final_equity > 0 && years > 0) {
        cagr = Math.pow(final_equity / initial_equity, 1 / years) - 1.0;
      } else {
        cagr = -0.5;
      }
      const returns = [];
      let valid_returns = 0;
      for (let i = 1; i < equity_curve.length; i++) {
        if (equity_curve[i-1] > 0) {
          const daily_return = (equity_curve[i] / equity_curve[i-1]) - 1;
          if (!isNaN(daily_return) && isFinite(daily_return)) {
            returns.push(daily_return);
            valid_returns++;
          }
        }
      }
      let sharpe = 0;
      if (valid_returns > 10) {
        const avg_return = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((acc, r) => acc + Math.pow(r - avg_return, 2), 0) / returns.length;
        const std_return = Math.sqrt(variance);
        if (std_return > 0) {
          sharpe = (avg_return / std_return) * Math.sqrt(252);
        }
      }
      let peak = equity_curve[0] || initial_equity;
      let max_dd = 0;
      for (const eq of equity_curve) {
        if (eq > peak) peak = eq;
        if (peak > 0) {
          const dd = (peak - eq) / peak;
          if (dd > max_dd) max_dd = dd;
        }
      }
      let wins = 0;
      let losses = 0;
      let total_trades = 0;
      let total_pnl = 0;
      for (let j = 0; j < trades.length - 1; j += 2) {
        if (trades[j] && trades[j+1] && trades[j].type === "buy" && trades[j+1].type === "sell") {
          total_trades++;
          const pnl = (trades[j+1].price - trades[j].price) * trades[j+1].units;
          total_pnl += pnl;
          if (pnl > 0) {
            wins++;
          } else {
            losses++;
          }
        }
      }
      const win_rate = total_trades > 0 ? wins / total_trades : 0;
      return {
        equity_curve: equity_curve,
        final_equity: isNaN(final_equity) ? 0 : final_equity,
        cagr: isNaN(cagr) ? -0.5 : Math.max(-0.99, Math.min(5.0, cagr)),
        sharpe: isNaN(sharpe) ? -2 : Math.max(-5, Math.min(10, sharpe)),
        max_dd: isNaN(max_dd) ? 1.0 : Math.max(0, Math.min(1.0, max_dd)),
        total_trades: total_trades,
        win_rate: isNaN(win_rate) ? 0 : win_rate,
        total_return: isNaN(total_return) ? -1.0 : total_return,
        total_pnl: total_pnl,
        avg_trade: total_trades > 0 ? total_pnl / total_trades : 0
      };
    } catch (error) {
      console.error("Backtest error:", error);
      return {
        equity_curve: new Array(DAYS).fill(initial_equity * 0.5),
        final_equity: initial_equity * 0.5,
        cagr: -0.5,
        sharpe: -2,
        max_dd: 0.5,
        total_trades: 0,
        win_rate: 0,
        total_return: -0.5,
        total_pnl: -initial_equity * 0.5,
        avg_trade: 0
      };
    }
  };

  const runOptimization = async () => {
    setIsRunning(true);
    setProgress(0);
    setResults(null);
    setBestStrategy(null);
    setEquityCurve([]);
    setLogs([]);
    
    try {
      addLog("Starting FULL AutoML Optimization...");
      const priceData = generatePriceData();

      let population = [];
      for (let i = 0; i < POP; i++) {
        population.push(generateCandidate());
      }

      let bestOverallStrategy = null;
      let bestOverallFitness = -Infinity;

      for (let g = 0; g < GEN; g++) {
        setCurrentGeneration(g + 1);
        addLog(`--- Starting Generation ${g + 1}/${GEN} ---`);

        const fitnessScores = await Promise.all(population.map(async (cand, i) => {
          let modelPredictions = new Array(priceData.close_arr.length).fill(0.5);

          if (cand.components.includes("ml_signal")) {
            addLog(`Training custom model for candidate ${i}...`);
            const response = await fetch('http://localhost:5000/train_and_predict_dynamic', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prices: {
                    open: priceData.open_arr,
                    high: priceData.high_arr,
                    low: priceData.low_arr,
                    close: priceData.close_arr,
                },
                model_config: cand.params.model_config
              }),
            });
            if (response.ok) {
              const data = await response.json();
              modelPredictions = data.predictions;
            } else {
              addLog(`Model training failed for candidate ${i}. Using neutral preds.`);
            }
          }
          
          const result = backtestStrategy(cand, priceData, indicators, modelPredictions);
          setProgress(Math.floor(((g * POP + i + 1) / (GEN * POP)) * 100));
          
          const returnToDD = result.max_dd > 0 ? result.cagr / result.max_dd : result.cagr * 2;
          return (returnToDD * 0.6) + (result.sharpe * 0.4);
        }));

        const popWithFitness = population.map((p, i) => ({ candidate: p, fitness: fitnessScores[i] }))
            .sort((a, b) => b.fitness - a.fitness);

        if (popWithFitness[0].fitness > bestOverallFitness) {
            bestOverallFitness = popWithFitness[0].fitness;
            bestOverallStrategy = popWithFitness[0].candidate;
            addLog(`New best strategy found in Gen ${g+1} with fitness ${bestOverallFitness.toFixed(3)}`);
        }

        const eliteCount = Math.floor(POP * ELITE_FRAC);
        const newPopulation = popWithFitness.slice(0, eliteCount).map(p => p.candidate);

        while (newPopulation.length < POP) {
            const parent1 = popWithFitness[Math.floor(Math.random() * eliteCount)].candidate;
            const parent2 = popWithFitness[Math.floor(Math.random() * eliteCount)].candidate;
            const crossoverPoint = Math.floor(Math.random() * parent1.components.length);
            const childComps = [...parent1.components.slice(0, crossoverPoint), ...parent2.components.slice(crossoverPoint)];
            const childParams = {...parent1.params, ...parent2.params};
            let child = { components: [...new Set(childComps)], params: childParams };
            if (Math.random() < 0.3) { 
                child = generateCandidate(); 
            }
            newPopulation.push(child);
        }
        population = newPopulation;
      }
      
      addLog("Optimization finished. Analyzing best strategy...");
      const finalResults = backtestStrategy(bestOverallStrategy, priceData, indicators, modelPredictions);
      setResults(finalResults);
      setBestStrategy(bestOverallStrategy);
      
      const curve = finalResults.equity_curve.map((eq, i) => ({ day: i, equity: eq }));
      setEquityCurve(curve);
      
      addLog(`🏆 Best Strategy: ${bestOverallStrategy.components.join(', ')}`);
      addLog(`Final Equity: ${finalResults.final_equity.toLocaleString()}`);
      addLog(`CAGR: ${(finalResults.cagr * 100).toFixed(2)}% | Sharpe: ${finalResults.sharpe.toFixed(2)}`);

    } catch (error) {
      addLog(`An error occurred: ${error.message}`);
      console.error("Optimization failed:", error);
    } finally {
      setIsRunning(false);
    }
  };

  const generatePriceData = () => {
    const price0 = 50000.0;
    
    const bull_params = { mu: 0.0015, omega: 1e-7, alpha: 0.05, beta: 0.90 };
    const crash_params = { mu: -0.004, omega: 5e-6, alpha: 0.15, beta: 0.80 };
    const sideways_params = { mu: 0.0001, omega: 3e-6, alpha: 0.10, beta: 0.88 };

    const returns = new Array(DAYS);
    const sigma2 = new Array(DAYS);
    
    sigma2[0] = 0.0001;
    
    for (let t = 1; t < DAYS; t++) {
        let params;
        if (t < 200) {
            params = bull_params;
        } else if (t < 300) {
            params = crash_params;
        } else {
            params = sideways_params;
        }
        
        const prev_eps = returns[t-1] - (params.mu || 0); // Handle undefined mu for first iteration
        sigma2[t] = params.omega + params.alpha * (prev_eps ** 2) + params.beta * sigma2[t-1];
        returns[t] = params.mu + Math.sqrt(Math.max(1e-12, sigma2[t])) * normalRandom();
    }

    const logprice = new Array(DAYS);
    const close_arr = new Array(DAYS);
    const high_arr = new Array(DAYS);
    const low_arr = new Array(DAYS);
    const open_arr = new Array(DAYS);

    logprice[0] = Math.log(price0);
    for (let i = 1; i < DAYS; i++) {
        logprice[i] = logprice[i-1] + returns[i];
    }

    for (let i = 0; i < DAYS; i++) {
        close_arr[i] = Math.exp(logprice[i]);
        const volSpread = 0.003 + 2 * Math.sqrt(sigma2[i]);
        high_arr[i] = close_arr[i] * (1 + Math.abs(normalRandom()) * volSpread);
        low_arr[i] = close_arr[i] * (1 - Math.abs(normalRandom()) * volSpread);
        if (i === 0) {
          open_arr[i] = price0;
        } else {
          open_arr[i] = close_arr[i-1] * (normalRandom() * 0.001 + 1);
        }
    }
    
    return { close_arr, open_arr, high_arr, low_arr, sigma2 };
  };

  const sma_cache = {};
  const rsi_arr = [];
  const atr_arr = [];
  const indicators = { sma_cache, rsi_arr, atr_arr, fft_std: {} };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '1200px', margin: 'auto' }}>
      <h1>HDD Strategy Optimizer v4.0 (Feature Engineering)</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <button onClick={runOptimization} disabled={isRunning}>
          {isRunning ? "Evolving..." : "Run Full AutoML Search"}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <h2>Optimization Logs</h2>
          <div style={{ background: '#f0f0f0', padding: '10px', height: '300px', overflowY: 'scroll', fontSize: '0.8em' }}>
            {logs.map((log, index) => (
              <p key={index} style={{ margin: '0' }}>{log}</p>
            ))}
          </div>
          <p>Progress: {progress}%</p>
          <p>Current Generation: {currentGeneration}/{GEN}</p>
        </div>

        <div style={{ flex: 1 }}>
          <h2>Best Strategy Results</h2>
          {bestStrategy && results ? (
            <div>
              <p><strong>Components:</strong> {bestStrategy.components.join(', ')}</p>
              <p><strong>Parameters:</strong></p>
              <pre style={{ background: '#f0f0f0', padding: '10px', overflowX: 'auto' }}>
                {JSON.stringify(bestStrategy.params, null, 2)}
              </pre>
              <p><strong>Final Equity:</strong> {results.final_equity.toLocaleString()}</p>
              <p><strong>CAGR:</strong> {(results.cagr * 100).toFixed(2)}%</p>
              <p><strong>Sharpe Ratio:</strong> {results.sharpe.toFixed(2)}</p>
              <p><strong>Max Drawdown:</strong> {(results.max_dd * 100).toFixed(2)}%</p>
              <p><strong>Total Trades:</strong> {results.total_trades}</p>
              <p><strong>Win Rate:</strong> {(results.win_rate * 100).toFixed(2)}%</p>
            </div>
          ) : (
            <p>Run optimization to see results.</p>
          )}
        </div>
      </div>

      {equityCurve.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h2>Equity Curve</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={equityCurve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Line type="monotone" dataKey="equity" stroke="#8884d8" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default HDDStrategyOptimizer;


