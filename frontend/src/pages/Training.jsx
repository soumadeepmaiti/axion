import { useState, useEffect, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Layers, Database,
  Activity, Clock, Target, Cpu, TrendingUp, TrendingDown,
  Calculator, GitBranch, Sigma, BarChart2
} from "lucide-react";
import { startTraining, getTrainingStatus, stopTraining, getTrainingHistory, getModelSummary } from "@/lib/api";

const Training = () => {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [lookbackDays, setLookbackDays] = useState(30);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [modelSummary, setModelSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mathAnalysis, setMathAnalysis] = useState(null);
  const [dataInfo, setDataInfo] = useState(null);
  const intervalRef = useRef(null);

  // Fast polling during training (every 500ms)
  const fetchStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      
      if (status.math_analysis) {
        setMathAnalysis(status.math_analysis);
      }
      if (status.data_info) {
        setDataInfo(status.data_info);
      }
    } catch (error) {
      console.error("Error fetching training status:", error);
    }
  }, []);

  const fetchInitialData = useCallback(async () => {
    try {
      const [history, summary] = await Promise.all([
        getTrainingHistory(),
        getModelSummary()
      ]);
      setTrainingHistory(history.history || []);
      setModelSummary(summary);
      
      // Get math analysis from latest training if available
      if (history.history && history.history.length > 0) {
        const latest = history.history[0];
        if (latest.math_analysis) setMathAnalysis(latest.math_analysis);
        if (latest.data_info) setDataInfo(latest.data_info);
      }
    } catch (error) {
      console.error("Error fetching initial data:", error);
    }
  }, []);

  useEffect(() => {
    fetchInitialData();
    fetchStatus();
  }, [fetchInitialData, fetchStatus]);

  // Dynamic polling interval based on training status
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // Poll every 500ms during training, every 5s when idle
    const pollInterval = trainingStatus?.is_training ? 500 : 5000;
    intervalRef.current = setInterval(fetchStatus, pollInterval);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [trainingStatus?.is_training, fetchStatus]);

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      const response = await startTraining(symbol, epochs, batchSize, lookbackDays);
      toast.success("Training started with ALL historical data!");
      
      if (response.math_analysis) {
        setMathAnalysis(response.math_analysis);
      }
      if (response.config?.data_range) {
        setDataInfo({ date_range: response.config.data_range, total_samples: response.config.total_samples });
      }
      
      fetchStatus();
    } catch (error) {
      console.error("Error starting training:", error);
      toast.error(error.response?.data?.detail || "Failed to start training");
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopTraining();
      toast.info("Training stopped");
      fetchStatus();
    } catch (error) {
      console.error("Error stopping training:", error);
      toast.error("Failed to stop training");
    }
  };

  const progressPercent = trainingStatus?.is_training && trainingStatus?.total_epochs > 0
    ? (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100
    : (trainingStatus?.current_epoch && trainingStatus?.total_epochs) 
      ? (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100 
      : 0;

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Model Training</h1>
          <p className="text-muted-foreground mt-1">Train with ALL historical data & Mathematical Modeling</p>
        </div>
        <div className="flex items-center gap-2">
          {trainingStatus?.is_training && (
            <Badge className="bg-success animate-pulse">LIVE</Badge>
          )}
          <Button
            variant="outline"
            size="icon"
            onClick={fetchStatus}
            className="border-border hover:border-primary/50"
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training Configuration */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              Training Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="symbol">Trading Pair</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger data-testid="training-symbol-select" className="bg-secondary border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                  <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="epochs">Epochs</Label>
              <Input
                data-testid="epochs-input"
                id="epochs"
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="bg-secondary border-border"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="batchSize">Batch Size</Label>
              <Input
                data-testid="batch-size-input"
                id="batchSize"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="bg-secondary border-border"
              />
            </div>

            <div className="flex gap-2 pt-4">
              {trainingStatus?.is_training ? (
                <Button
                  data-testid="stop-training-btn"
                  variant="destructive"
                  onClick={handleStopTraining}
                  className="flex-1"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </Button>
              ) : (
                <Button
                  data-testid="start-training-btn"
                  onClick={handleStartTraining}
                  disabled={loading}
                  className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-primary"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 mr-2" />
                  )}
                  Train with ALL Data
                </Button>
              )}
            </div>

            {/* Data Info */}
            {dataInfo && (
              <div className="mt-4 p-3 bg-secondary/50 rounded-lg text-sm">
                <p className="text-muted-foreground">Data Range:</p>
                <p className="font-mono text-xs mt-1">
                  {dataInfo.date_range?.start?.split(' ')[0]} → {dataInfo.date_range?.end?.split(' ')[0]}
                </p>
                {dataInfo.total_samples && (
                  <p className="text-primary font-semibold mt-1">{dataInfo.total_samples?.toLocaleString()} samples</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Training Progress - Real-time */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Training Progress
              {trainingStatus?.is_training && (
                <span className="ml-2 text-xs text-muted-foreground animate-pulse">● Updating every 500ms</span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  Epoch {trainingStatus?.current_epoch || 0} / {trainingStatus?.total_epochs || 0}
                </span>
                <span className="font-mono text-primary">
                  {progressPercent.toFixed(1)}%
                </span>
              </div>
              <Progress value={progressPercent} className="h-3" />
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Status</p>
                <Badge 
                  data-testid="training-status-badge"
                  className={trainingStatus?.is_training ? 'bg-success mt-2 animate-pulse' : 'bg-secondary mt-2'}
                >
                  {trainingStatus?.is_training ? 'Training' : 'Idle'}
                </Badge>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Loss</p>
                <p className="text-xl font-bold font-mono text-foreground mt-1">
                  {trainingStatus?.current_loss?.toFixed(4) || '0.0000'}
                </p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Accuracy</p>
                <p className="text-xl font-bold font-mono text-success mt-1">
                  {((trainingStatus?.current_accuracy || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Duration</p>
                <p className="text-sm font-mono text-foreground mt-2">
                  {trainingStatus?.start_time 
                    ? new Date(trainingStatus.start_time).toLocaleTimeString()
                    : '--:--:--'}
                </p>
              </div>
            </div>

            {/* Loss Curve - Real-time */}
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingStatus?.history || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                  <XAxis 
                    dataKey="epoch" 
                    tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    label={{ value: 'Epoch', position: 'bottom', fill: '#A1A1AA', fontSize: 10 }}
                  />
                  <YAxis 
                    tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F', borderRadius: '8px' }}
                    formatter={(value, name) => [value?.toFixed(4), name]}
                  />
                  <Line type="monotone" dataKey="loss" stroke="#FF2E55" strokeWidth={2} dot={false} name="Train Loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="#00E5FF" strokeWidth={2} dot={false} name="Val Loss" />
                  <Line type="monotone" dataKey="accuracy" stroke="#00FF94" strokeWidth={2} dot={false} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Mathematical Analysis Section */}
      {mathAnalysis && (
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Calculator className="w-5 h-5 text-primary" />
              Mathematical Modeling & Formulas Learned
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              
              {/* Trend Analysis */}
              {mathAnalysis.trend && (
                <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                  <div className="flex items-center gap-2 mb-3">
                    {mathAnalysis.trend.trend?.includes('bullish') ? (
                      <TrendingUp className="w-5 h-5 text-success" />
                    ) : mathAnalysis.trend.trend?.includes('bearish') ? (
                      <TrendingDown className="w-5 h-5 text-destructive" />
                    ) : (
                      <BarChart2 className="w-5 h-5 text-warning" />
                    )}
                    <h3 className="font-mono font-semibold">Trend Analysis</h3>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Trend</span>
                      <Badge className={
                        mathAnalysis.trend.trend?.includes('bullish') ? 'bg-success' :
                        mathAnalysis.trend.trend?.includes('bearish') ? 'bg-destructive' : 'bg-warning'
                      }>
                        {mathAnalysis.trend.trend?.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">ADX Strength</span>
                      <span className="font-mono text-primary">{mathAnalysis.trend.adx_strength}</span>
                    </div>
                    <Separator className="my-2" />
                    <div className="bg-background/50 p-2 rounded font-mono text-xs">
                      <p className="text-muted-foreground mb-1">Regression Formula:</p>
                      <p className="text-primary">
                        y = {mathAnalysis.trend.regression_line?.slope?.toFixed(2)}x + {mathAnalysis.trend.regression_line?.intercept?.toFixed(2)}
                      </p>
                      <p className="text-muted-foreground mt-1">
                        Slope: {mathAnalysis.trend.slope_pct?.toFixed(4)}%/candle
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Support & Resistance */}
              {mathAnalysis.support_resistance && (
                <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                  <div className="flex items-center gap-2 mb-3">
                    <Layers className="w-5 h-5 text-chart-4" />
                    <h3 className="font-mono font-semibold">Support & Resistance</h3>
                  </div>
                  <div className="space-y-3 text-sm">
                    <div>
                      <p className="text-muted-foreground mb-1">Resistance Levels:</p>
                      <div className="space-y-1">
                        {mathAnalysis.support_resistance.resistance?.map((level, i) => (
                          <div key={i} className="flex justify-between bg-destructive/10 px-2 py-1 rounded">
                            <span className="text-destructive">R{i + 1}</span>
                            <span className="font-mono">${level?.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-muted-foreground mb-1">Support Levels:</p>
                      <div className="space-y-1">
                        {mathAnalysis.support_resistance.support?.map((level, i) => (
                          <div key={i} className="flex justify-between bg-success/10 px-2 py-1 rounded">
                            <span className="text-success">S{i + 1}</span>
                            <span className="font-mono">${level?.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Fibonacci Retracement */}
              {mathAnalysis.fibonacci && (
                <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                  <div className="flex items-center gap-2 mb-3">
                    <Sigma className="w-5 h-5 text-warning" />
                    <h3 className="font-mono font-semibold">Fibonacci Levels</h3>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="bg-background/50 p-2 rounded mb-2">
                      <p className="text-xs text-muted-foreground">Range:</p>
                      <p className="font-mono text-xs">
                        ${mathAnalysis.fibonacci.low?.toLocaleString()} → ${mathAnalysis.fibonacci.high?.toLocaleString()}
                      </p>
                    </div>
                    {Object.entries(mathAnalysis.fibonacci.levels || {}).slice(0, 5).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-warning font-mono">{(parseFloat(key) * 100).toFixed(1)}%</span>
                        <span className="font-mono">${value?.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pattern Detection & Volatility */}
              <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                <div className="flex items-center gap-2 mb-3">
                  <GitBranch className="w-5 h-5 text-chart-5" />
                  <h3 className="font-mono font-semibold">Patterns & Volatility</h3>
                </div>
                <div className="space-y-3 text-sm">
                  {mathAnalysis.patterns && mathAnalysis.patterns.length > 0 && (
                    <div>
                      <p className="text-muted-foreground mb-1">Detected Patterns:</p>
                      <div className="space-y-1">
                        {mathAnalysis.patterns.map((pattern, i) => (
                          <Badge key={i} variant="outline" className={
                            pattern.type?.includes('bullish') ? 'border-success text-success' : 'border-destructive text-destructive'
                          }>
                            {pattern.name?.replace(/_/g, ' ')}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {mathAnalysis.volatility && (
                    <div className="bg-background/50 p-2 rounded">
                      <p className="text-xs text-muted-foreground mb-1">Volatility Regime:</p>
                      <Badge className={
                        mathAnalysis.volatility.regime === 'high_volatility' ? 'bg-destructive' :
                        mathAnalysis.volatility.regime === 'low_volatility' ? 'bg-success' : 'bg-warning'
                      }>
                        {mathAnalysis.volatility.regime?.replace(/_/g, ' ')}
                      </Badge>
                      <div className="mt-2 text-xs font-mono">
                        <p>Current: {mathAnalysis.volatility.current_vol}%</p>
                        <p>Average: {mathAnalysis.volatility.avg_vol}%</p>
                        <p>Ratio: {mathAnalysis.volatility.vol_ratio}x</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Formula Summary */}
            <div className="mt-6 p-4 bg-background/30 rounded-lg border border-primary/20">
              <h4 className="font-mono font-semibold text-primary mb-3 flex items-center gap-2">
                <Calculator className="w-4 h-4" />
                Model Learning Summary
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm font-mono">
                <div>
                  <p className="text-muted-foreground">Price Prediction Formula:</p>
                  <p className="text-foreground mt-1">
                    P(t+1) = LSTM(X<sub>t-50:t</sub>) × σ(W·h + b)
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Take Profit Formula:</p>
                  <p className="text-success mt-1">
                    TP = Price ± (2.5 × ATR<sub>14</sub>)
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Stop Loss Formula:</p>
                  <p className="text-destructive mt-1">
                    SL = Price ∓ (1.5 × ATR<sub>14</sub>)
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Architecture */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="font-mono text-lg flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            Model Architecture
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Branch A */}
            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-5 h-5 text-buy" />
                <h3 className="font-mono font-semibold">Branch A: Micro</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">Bi-LSTM with Attention for price sequences</p>
              <div className="space-y-2 text-sm font-mono bg-background/50 p-2 rounded">
                <p>Input: (50, 30) sequences</p>
                <p>LSTM: 128 → 64 units</p>
                <p>Attention: Self-attention</p>
              </div>
            </div>

            {/* Branch B */}
            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Database className="w-5 h-5 text-warning" />
                <h3 className="font-mono font-semibold">Branch B: Macro</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">30+ Technical Indicators</p>
              <div className="space-y-1 text-xs font-mono bg-background/50 p-2 rounded">
                <p>RSI, MACD, Stochastic</p>
                <p>Bollinger Bands, ATR</p>
                <p>EMA 9/21/50/200</p>
                <p>ADX, CCI, MFI, OBV</p>
              </div>
            </div>

            {/* Branch C */}
            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-5 h-5 text-chart-4" />
                <h3 className="font-mono font-semibold">Branch C: Math</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">Mathematical Pattern Recognition</p>
              <div className="space-y-1 text-xs font-mono bg-background/50 p-2 rounded">
                <p>Support/Resistance</p>
                <p>Fibonacci Retracement</p>
                <p>Trend Regression</p>
                <p>Pattern Detection</p>
              </div>
            </div>
          </div>

          {/* Model Stats */}
          <div className="mt-6 p-4 bg-secondary rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Status</p>
                <Badge className="mt-2">{modelSummary?.status || 'ready'}</Badge>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Parameters</p>
                <p className="text-lg font-bold font-mono text-foreground mt-1">
                  {(modelSummary?.total_params || 0).toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Dropout</p>
                <p className="text-lg font-bold font-mono text-foreground mt-1">
                  {modelSummary?.dropout_rate || 0.3}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Trained</p>
                <Badge className={trainingStatus?.current_accuracy > 0 ? 'bg-success mt-2' : 'bg-secondary mt-2'}>
                  {trainingStatus?.current_accuracy > 0 ? 'Yes' : 'No'}
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training History */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="font-mono text-lg flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            Training History
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trainingHistory.length > 0 ? (
            <div className="space-y-3">
              {trainingHistory.map((session, index) => (
                <div key={index} className="p-4 bg-secondary rounded-lg flex items-center justify-between">
                  <div>
                    <p className="font-mono font-semibold">{session.symbol}</p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(session.created_at).toLocaleString()}
                    </p>
                    {session.data_info && (
                      <p className="text-xs text-muted-foreground">
                        {session.data_info.total_5m_candles || 0} candles processed
                      </p>
                    )}
                  </div>
                  <div className="text-right">
                    <Badge className={session.result?.status === 'completed' ? 'bg-success' : 'bg-warning'}>
                      {session.result?.status || 'unknown'}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      Acc: {((session.result?.final_accuracy || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">No training history yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Training;
