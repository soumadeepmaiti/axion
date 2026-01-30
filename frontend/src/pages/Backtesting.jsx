import { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, Cell, ReferenceLine
} from "recharts";
import {
  Play, Square, RefreshCw, Activity, Clock, TrendingUp, TrendingDown,
  CalendarIcon, DollarSign, Target, BarChart3, Percent, AlertTriangle,
  CheckCircle, XCircle, ArrowUpRight, ArrowDownRight, Wallet, History
} from "lucide-react";
import { format } from "date-fns";
import { API } from "@/lib/api";

// API functions
const startBacktest = async (config) => {
  const response = await API.post('/backtest/start', config);
  return response.data;
};

const getBacktestStatus = async () => {
  const response = await API.get('/backtest/status');
  return response.data;
};

const getBacktestResult = async () => {
  const response = await API.get('/backtest/result');
  return response.data;
};

const stopBacktest = async () => {
  const response = await API.post('/backtest/stop');
  return response.data;
};

const getBacktestHistory = async () => {
  const response = await API.get('/backtest/history?limit=10');
  return response.data;
};

const Backtesting = () => {
  // Config state
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [positionSize, setPositionSize] = useState(0.1);
  const [useStopLoss, setUseStopLoss] = useState(true);
  const [stopLossPct, setStopLossPct] = useState(0.02);
  const [useTakeProfit, setUseTakeProfit] = useState(true);
  const [takeProfitPct, setTakeProfitPct] = useState(0.04);
  const [maxHoldTime, setMaxHoldTime] = useState(24);
  const [minConfidence, setMinConfidence] = useState(0.6);
  const [commission, setCommission] = useState(0.001);

  // Status and results
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const intervalRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const statusData = await getBacktestStatus();
      setStatus(statusData);
      
      // If backtest completed, fetch result
      if (!statusData.is_running && status?.is_running) {
        const resultData = await getBacktestResult();
        if (resultData && resultData.symbol) {
          setResult(resultData);
          toast.success("Backtest completed!");
        }
      }
    } catch (error) {
      console.error("Error fetching status:", error);
    }
  }, [status?.is_running]);

  const fetchHistory = useCallback(async () => {
    try {
      const data = await getBacktestHistory();
      setHistory(data.history || []);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchHistory();
  }, []);

  // Poll status when running
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    const pollInterval = status?.is_running ? 1000 : 10000;
    intervalRef.current = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(intervalRef.current);
  }, [status?.is_running, fetchStatus]);

  const handleStartBacktest = async () => {
    if (status?.is_running) {
      toast.error("Backtest already in progress");
      return;
    }

    setLoading(true);
    setResult(null);
    
    try {
      const config = {
        symbol,
        timeframe,
        start_date: startDate ? startDate.toISOString() : null,
        end_date: endDate ? endDate.toISOString() : null,
        initial_capital: initialCapital,
        position_size: positionSize,
        use_stop_loss: useStopLoss,
        stop_loss_pct: stopLossPct,
        use_take_profit: useTakeProfit,
        take_profit_pct: takeProfitPct,
        max_hold_time: maxHoldTime,
        min_confidence: minConfidence,
        commission
      };

      await startBacktest(config);
      toast.success("Backtest started!");
      setTimeout(fetchStatus, 500);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Failed to start backtest";
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleStopBacktest = async () => {
    try {
      await stopBacktest();
      toast.success("Backtest stopped");
      fetchStatus();
    } catch (error) {
      toast.error("Failed to stop backtest");
    }
  };

  const isRunning = status?.is_running;

  // Format currency
  const formatCurrency = (value) => {
    if (value === null || value === undefined) return "$0.00";
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  };

  // Format percentage
  const formatPercent = (value) => {
    if (value === null || value === undefined) return "0.00%";
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div data-testid="backtesting-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Backtesting</h1>
          <p className="text-muted-foreground mt-1">Test your strategy against historical data</p>
        </div>
        {isRunning && (
          <Badge className="bg-primary animate-pulse text-lg px-4 py-1">BACKTEST RUNNING</Badge>
        )}
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="config" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="config" className="gap-2"><Target className="w-4 h-4" /> Configuration</TabsTrigger>
          <TabsTrigger value="results" className="gap-2"><BarChart3 className="w-4 h-4" /> Results</TabsTrigger>
          <TabsTrigger value="history" className="gap-2"><History className="w-4 h-4" /> History</TabsTrigger>
        </TabsList>

        {/* ==================== CONFIGURATION TAB ==================== */}
        <TabsContent value="config" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            
            {/* Basic Settings */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary" />
                  Basic Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Symbol</Label>
                  <Select value={symbol} onValueChange={setSymbol} disabled={isRunning}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                      <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Timeframe</Label>
                  <Select value={timeframe} onValueChange={setTimeframe} disabled={isRunning}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5m">5 Minutes</SelectItem>
                      <SelectItem value="15m">15 Minutes</SelectItem>
                      <SelectItem value="1h">1 Hour</SelectItem>
                      <SelectItem value="4h">4 Hours</SelectItem>
                      <SelectItem value="1d">1 Day</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Date Range</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isRunning}>
                          <CalendarIcon className="mr-1 h-3 w-3" />
                          {startDate ? format(startDate, "MMM dd") : "Start"}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={startDate} onSelect={setStartDate} /></PopoverContent>
                    </Popover>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isRunning}>
                          <CalendarIcon className="mr-1 h-3 w-3" />
                          {endDate ? format(endDate, "MMM dd") : "End"}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={endDate} onSelect={setEndDate} /></PopoverContent>
                    </Popover>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Capital Settings */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Wallet className="w-4 h-4 text-primary" />
                  Capital Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Initial Capital</Label>
                    <span className="font-mono text-primary">{formatCurrency(initialCapital)}</span>
                  </div>
                  <Input 
                    type="number" 
                    value={initialCapital} 
                    onChange={(e) => setInitialCapital(parseFloat(e.target.value) || 10000)} 
                    className="bg-secondary border-border" 
                    disabled={isRunning}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Position Size</Label>
                    <span className="font-mono text-primary">{(positionSize * 100).toFixed(0)}%</span>
                  </div>
                  <Slider 
                    value={[positionSize * 100]} 
                    onValueChange={([v]) => setPositionSize(v / 100)} 
                    min={5} max={50} step={5} 
                    disabled={isRunning}
                  />
                  <p className="text-xs text-muted-foreground">Per-trade capital allocation</p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Commission</Label>
                    <span className="font-mono text-primary">{(commission * 100).toFixed(2)}%</span>
                  </div>
                  <Slider 
                    value={[commission * 1000]} 
                    onValueChange={([v]) => setCommission(v / 1000)} 
                    min={0} max={5} step={0.5} 
                    disabled={isRunning}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Risk Management */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-primary" />
                  Risk Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                  <div>
                    <Label className="text-xs">Stop Loss</Label>
                    <p className="text-xs text-muted-foreground">Auto-exit on loss</p>
                  </div>
                  <Switch checked={useStopLoss} onCheckedChange={setUseStopLoss} disabled={isRunning} />
                </div>

                {useStopLoss && (
                  <div className="space-y-2 pl-2 border-l-2 border-destructive/30">
                    <div className="flex justify-between text-xs">
                      <Label>Stop Loss %</Label>
                      <span className="font-mono text-destructive">{(stopLossPct * 100).toFixed(1)}%</span>
                    </div>
                    <Slider 
                      value={[stopLossPct * 100]} 
                      onValueChange={([v]) => setStopLossPct(v / 100)} 
                      min={0.5} max={10} step={0.5} 
                      disabled={isRunning}
                    />
                  </div>
                )}

                <div className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                  <div>
                    <Label className="text-xs">Take Profit</Label>
                    <p className="text-xs text-muted-foreground">Auto-exit on profit</p>
                  </div>
                  <Switch checked={useTakeProfit} onCheckedChange={setUseTakeProfit} disabled={isRunning} />
                </div>

                {useTakeProfit && (
                  <div className="space-y-2 pl-2 border-l-2 border-success/30">
                    <div className="flex justify-between text-xs">
                      <Label>Take Profit %</Label>
                      <span className="font-mono text-success">{(takeProfitPct * 100).toFixed(1)}%</span>
                    </div>
                    <Slider 
                      value={[takeProfitPct * 100]} 
                      onValueChange={([v]) => setTakeProfitPct(v / 100)} 
                      min={1} max={20} step={1} 
                      disabled={isRunning}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Trade Settings */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Clock className="w-4 h-4 text-primary" />
                  Trade Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Max Hold Time</Label>
                    <span className="font-mono text-primary">{maxHoldTime}h</span>
                  </div>
                  <Slider 
                    value={[maxHoldTime]} 
                    onValueChange={([v]) => setMaxHoldTime(v)} 
                    min={0} max={72} step={4} 
                    disabled={isRunning}
                  />
                  <p className="text-xs text-muted-foreground">{maxHoldTime === 0 ? "No time limit" : `Force exit after ${maxHoldTime} hours`}</p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Min Confidence</Label>
                    <span className="font-mono text-primary">{(minConfidence * 100).toFixed(0)}%</span>
                  </div>
                  <Slider 
                    value={[minConfidence * 100]} 
                    onValueChange={([v]) => setMinConfidence(v / 100)} 
                    min={50} max={90} step={5} 
                    disabled={isRunning}
                  />
                  <p className="text-xs text-muted-foreground">Only trade when model confidence exceeds this</p>
                </div>

                {/* Summary */}
                <div className="p-3 bg-primary/10 rounded border border-primary/30">
                  <h4 className="font-mono text-xs text-primary mb-2">Risk/Reward</h4>
                  <p className="text-xs">
                    {useStopLoss && useTakeProfit ? (
                      <span>R/R Ratio: <span className="font-mono text-primary">{(takeProfitPct / stopLossPct).toFixed(1)}:1</span></span>
                    ) : (
                      <span className="text-muted-foreground">Configure SL/TP for R/R</span>
                    )}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Start/Stop Buttons */}
          <div className="flex justify-center gap-4">
            <Button 
              data-testid="start-backtest-btn"
              size="lg"
              onClick={handleStartBacktest} 
              disabled={loading || isRunning}
              className="gap-2 bg-primary hover:bg-primary/90 text-white px-8 py-6 text-lg"
            >
              {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Play className="w-6 h-6" />}
              Start Backtest
            </Button>
            <Button 
              data-testid="stop-backtest-btn"
              variant="destructive"
              size="lg"
              onClick={handleStopBacktest}
              disabled={!isRunning}
              className="gap-2 px-8 py-6 text-lg"
            >
              <Square className="w-6 h-6" />
              Stop
            </Button>
          </div>

          {/* Progress */}
          {isRunning && (
            <Card className="bg-card border-2 border-primary">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-sm">Backtest Progress</span>
                    <span className="font-mono text-primary text-xl">{status?.progress || 0}%</span>
                  </div>
                  <Progress value={status?.progress || 0} className="h-3" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Date: {status?.current_date || '-'}</span>
                    <span>Trades: {status?.total_trades || 0}</span>
                    <span>PnL: {formatCurrency(status?.current_pnl || 0)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ==================== RESULTS TAB ==================== */}
        <TabsContent value="results" className="space-y-6">
          {result ? (
            <>
              {/* Performance Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Final Capital</p>
                    <p className={`font-mono text-2xl ${result.final_capital > result.initial_capital ? 'text-success' : 'text-destructive'}`}>
                      {formatCurrency(result.final_capital)}
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Total Return</p>
                    <p className={`font-mono text-2xl ${result.total_return_percent >= 0 ? 'text-success' : 'text-destructive'}`}>
                      {formatPercent(result.total_return_percent)}
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Sharpe Ratio</p>
                    <p className={`font-mono text-2xl ${result.sharpe_ratio >= 1 ? 'text-success' : result.sharpe_ratio >= 0 ? 'text-warning' : 'text-destructive'}`}>
                      {result.sharpe_ratio?.toFixed(2) || '0.00'}
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Max Drawdown</p>
                    <p className="font-mono text-2xl text-destructive">
                      {result.max_drawdown_percent?.toFixed(2) || '0.00'}%
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Win Rate</p>
                    <p className={`font-mono text-2xl ${result.win_rate >= 50 ? 'text-success' : 'text-warning'}`}>
                      {result.win_rate?.toFixed(1) || '0.0'}%
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4">
                    <p className="text-xs text-muted-foreground uppercase">Profit Factor</p>
                    <p className={`font-mono text-2xl ${result.profit_factor >= 1.5 ? 'text-success' : result.profit_factor >= 1 ? 'text-warning' : 'text-destructive'}`}>
                      {result.profit_factor?.toFixed(2) || '0.00'}
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Equity Curve */}
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="font-mono text-base flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-primary" />
                      Equity Curve
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={result.equity_curve || []}>
                          <defs>
                            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#00E5FF" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                          <XAxis dataKey="timestamp" tick={{ fill: '#A1A1AA', fontSize: 10 }} tickFormatter={(v) => v?.split('T')[0] || ''} />
                          <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} domain={['auto', 'auto']} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                          <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} formatter={(v) => formatCurrency(v)} />
                          <ReferenceLine y={result.initial_capital} stroke="#FF2E55" strokeDasharray="5 5" />
                          <Area type="monotone" dataKey="equity" stroke="#00E5FF" fill="url(#equityGrad)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                {/* Drawdown Chart */}
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="font-mono text-base flex items-center gap-2">
                      <TrendingDown className="w-4 h-4 text-destructive" />
                      Drawdown
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={result.drawdown_curve || []}>
                          <defs>
                            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#FF2E55" stopOpacity={0.5}/>
                              <stop offset="95%" stopColor="#FF2E55" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                          <XAxis dataKey="timestamp" tick={{ fill: '#A1A1AA', fontSize: 10 }} tickFormatter={(v) => v?.split('T')[0] || ''} />
                          <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} domain={[0, 'auto']} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                          <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} formatter={(v) => `${v.toFixed(2)}%`} />
                          <Area type="monotone" dataKey="drawdown" stroke="#FF2E55" fill="url(#ddGrad)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Trade Statistics */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Stats Grid */}
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="font-mono text-base flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-primary" />
                      Trade Statistics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Total Trades</span>
                          <span className="font-mono text-sm">{result.total_trades}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Winning Trades</span>
                          <span className="font-mono text-sm text-success">{result.winning_trades}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Losing Trades</span>
                          <span className="font-mono text-sm text-destructive">{result.losing_trades}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Avg Win</span>
                          <span className="font-mono text-sm text-success">{formatCurrency(result.avg_win)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Avg Loss</span>
                          <span className="font-mono text-sm text-destructive">{formatCurrency(result.avg_loss)}</span>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Best Trade</span>
                          <span className="font-mono text-sm text-success">{formatCurrency(result.best_trade)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Worst Trade</span>
                          <span className="font-mono text-sm text-destructive">{formatCurrency(result.worst_trade)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Avg Duration</span>
                          <span className="font-mono text-sm">{result.avg_trade_duration?.toFixed(1) || 0}h</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Max Win Streak</span>
                          <span className="font-mono text-sm text-success">{result.consecutive_wins}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">Max Loss Streak</span>
                          <span className="font-mono text-sm text-destructive">{result.consecutive_losses}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Monthly Returns */}
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="font-mono text-base flex items-center gap-2">
                      <CalendarIcon className="w-4 h-4 text-primary" />
                      Monthly Returns
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={result.monthly_returns || []}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                          <XAxis dataKey="month" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                          <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                          <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} formatter={(v) => `${v.toFixed(2)}%`} />
                          <ReferenceLine y={0} stroke="#A1A1AA" />
                          <Bar dataKey="return" radius={[4, 4, 0, 0]}>
                            {(result.monthly_returns || []).map((entry, index) => (
                              <Cell key={index} fill={entry.return >= 0 ? '#00FF94' : '#FF2E55'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Recent Trades Table */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base flex items-center gap-2">
                    <Activity className="w-4 h-4 text-primary" />
                    Recent Trades ({result.trades?.length || 0} total)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border text-muted-foreground">
                          <th className="text-left py-2 px-2">Entry</th>
                          <th className="text-left py-2 px-2">Exit</th>
                          <th className="text-left py-2 px-2">Direction</th>
                          <th className="text-right py-2 px-2">Entry Price</th>
                          <th className="text-right py-2 px-2">Exit Price</th>
                          <th className="text-right py-2 px-2">PnL</th>
                          <th className="text-left py-2 px-2">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(result.trades || []).slice(-20).reverse().map((trade, i) => (
                          <tr key={i} className="border-b border-border/50 hover:bg-secondary/30">
                            <td className="py-2 px-2 font-mono text-xs">{trade.entry_time?.split('T')[0] || '-'}</td>
                            <td className="py-2 px-2 font-mono text-xs">{trade.exit_time?.split('T')[0] || '-'}</td>
                            <td className="py-2 px-2">
                              <Badge className={trade.direction === 'long' ? 'bg-success' : 'bg-destructive'}>
                                {trade.direction === 'long' ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                {trade.direction}
                              </Badge>
                            </td>
                            <td className="py-2 px-2 text-right font-mono">{formatCurrency(trade.entry_price)}</td>
                            <td className="py-2 px-2 text-right font-mono">{formatCurrency(trade.exit_price)}</td>
                            <td className={`py-2 px-2 text-right font-mono ${trade.pnl >= 0 ? 'text-success' : 'text-destructive'}`}>
                              {formatCurrency(trade.pnl)}
                            </td>
                            <td className="py-2 px-2">
                              <Badge variant="outline" className="text-xs">
                                {trade.exit_reason === 'take_profit' && <CheckCircle className="w-3 h-3 mr-1 text-success" />}
                                {trade.exit_reason === 'stop_loss' && <XCircle className="w-3 h-3 mr-1 text-destructive" />}
                                {trade.exit_reason}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="bg-card border-border">
              <CardContent className="py-16 text-center">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">No Backtest Results</h3>
                <p className="text-muted-foreground">Run a backtest to see performance metrics and trade analysis</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ==================== HISTORY TAB ==================== */}
        <TabsContent value="history" className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-lg flex items-center gap-2">
                  <History className="w-5 h-5 text-primary" />
                  Backtest History
                </CardTitle>
                <Button variant="outline" size="sm" onClick={fetchHistory}>
                  <RefreshCw className="w-4 h-4 mr-1" /> Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {history.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {history.map((bt, i) => (
                    <div key={i} className="p-4 bg-secondary rounded-lg border border-border hover:border-primary/50 transition-all">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <p className="font-mono text-sm font-semibold">{bt.symbol}</p>
                          <p className="text-xs text-muted-foreground">{bt.created_at?.split('T')[0]}</p>
                        </div>
                        <Badge className={bt.result?.total_return_percent >= 0 ? 'bg-success' : 'bg-destructive'}>
                          {formatPercent(bt.result?.total_return_percent)}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-muted-foreground">Sharpe: </span>
                          <span className="font-mono">{bt.result?.sharpe_ratio?.toFixed(2) || '-'}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Win Rate: </span>
                          <span className="font-mono">{bt.result?.win_rate?.toFixed(1) || '-'}%</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Trades: </span>
                          <span className="font-mono">{bt.result?.total_trades || 0}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Max DD: </span>
                          <span className="font-mono text-destructive">{bt.result?.max_drawdown_percent?.toFixed(1) || '-'}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <History className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">No backtest history yet</p>
                  <p className="text-xs text-muted-foreground mt-1">Run your first backtest to see results here</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Backtesting;
