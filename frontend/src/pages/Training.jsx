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
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Area, AreaChart
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Activity, Clock, Cpu,
  CalendarIcon, TrendingUp, Zap, Layers, Settings, Calculator,
  GitBranch, Sigma, Network, StopCircle, PlayCircle
} from "lucide-react";
import { format } from "date-fns";
import { API } from "@/lib/api";

const startAdvancedTraining = async (config) => {
  const response = await API.post('/training/start', config);
  return response.data;
};

const getTrainingStatus = async () => {
  const response = await API.get('/training/status');
  return response.data;
};

const getTrainingHistory = async () => {
  const response = await API.get('/training/history?limit=10');
  return response.data;
};

const stopTraining = async () => {
  const response = await API.post('/training/stop');
  return response.data;
};

const MATH_STRATEGIES = [
  { id: "mean_reversion", name: "Mean Reversion", description: "Price returns to average" },
  { id: "momentum", name: "Momentum", description: "Trend following strategy" },
  { id: "volatility_breakout", name: "Volatility Breakout", description: "Bollinger Bands style" },
  { id: "rsi", name: "RSI Divergence", description: "Oversold/overbought signals" },
  { id: "macd", name: "MACD Crossover", description: "Moving average convergence" },
  { id: "fibonacci", name: "Fibonacci Levels", description: "Key retracement levels" },
  { id: "support_resistance", name: "Support/Resistance", description: "Dynamic price levels" },
];

const NETWORK_TYPES = [
  { id: "lstm", name: "LSTM", description: "Long Short-Term Memory - Best for sequences" },
  { id: "gru", name: "GRU", description: "Gated Recurrent Unit - Faster training" },
  { id: "transformer", name: "Transformer", description: "Attention-based (coming soon)", disabled: true },
  { id: "cnn_lstm", name: "CNN + LSTM", description: "Conv + Recurrent hybrid (coming soon)", disabled: true },
];

const Training = () => {
  // Basic config
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  
  // Mode
  const [mode, setMode] = useState("pure_ml");
  
  // Network type
  const [networkType, setNetworkType] = useState("lstm");
  
  // Network architecture
  const [numLstmLayers, setNumLstmLayers] = useState(2);
  const [lstmUnits, setLstmUnits] = useState([128, 64, 32, 16]);
  const [numDenseLayers, setNumDenseLayers] = useState(2);
  const [denseUnits, setDenseUnits] = useState([64, 32, 16, 8]);
  const [dropoutRate, setDropoutRate] = useState(0.3);
  const [useAttention, setUseAttention] = useState(true);
  const [useBatchNorm, setUseBatchNorm] = useState(true);
  const [learningRate, setLearningRate] = useState(0.001);
  const [sequenceLength, setSequenceLength] = useState(50);
  
  // Strategies
  const [selectedStrategies, setSelectedStrategies] = useState([]);
  
  // Status
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [learnedPatterns, setLearnedPatterns] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const intervalRef = useRef(null);
  const timerRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      if (status.learned_patterns) setLearnedPatterns(status.learned_patterns);
    } catch (error) {
      console.error("Error fetching status:", error);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const history = await getTrainingHistory();
      setTrainingHistory(history.history || []);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchStatus();
  }, [fetchHistory, fetchStatus]);

  // Real-time status polling - every second when training
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    const pollInterval = trainingStatus?.is_training ? 1000 : 5000;
    intervalRef.current = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(intervalRef.current);
  }, [trainingStatus?.is_training, fetchStatus]);

  // Elapsed time counter
  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (trainingStatus?.is_training && trainingStatus?.start_time) {
      const startTime = new Date(trainingStatus.start_time).getTime();
      timerRef.current = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(timerRef.current);
  }, [trainingStatus?.is_training, trainingStatus?.start_time]);

  const handleStartTraining = async () => {
    if (trainingStatus?.is_training) {
      toast.error("Training already in progress");
      return;
    }
    
    setLoading(true);
    try {
      const config = {
        symbol,
        epochs,
        batch_size: batchSize,
        start_date: startDate ? startDate.toISOString() : null,
        end_date: endDate ? endDate.toISOString() : null,
        timeframe,
        mode,
        strategies: selectedStrategies,
        num_lstm_layers: numLstmLayers,
        lstm_units: lstmUnits.slice(0, numLstmLayers),
        num_dense_layers: numDenseLayers,
        dense_units: denseUnits.slice(0, numDenseLayers),
        dropout_rate: dropoutRate,
        use_attention: useAttention,
        use_batch_norm: useBatchNorm,
        learning_rate: learningRate,
        sequence_length: sequenceLength
      };
      
      const result = await startAdvancedTraining(config);
      toast.success(`Training started! ${result.config?.total_samples || 0} samples loaded`);
      
      // Immediately fetch status to update UI
      setTimeout(fetchStatus, 500);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Failed to start training";
      toast.error(errorMsg);
      console.error("Training start error:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    if (!trainingStatus?.is_training) {
      toast.info("No training in progress");
      return;
    }
    
    try {
      await stopTraining();
      toast.success("Training stopped");
      fetchStatus();
      fetchHistory();
    } catch (error) {
      toast.error("Failed to stop training");
      console.error("Training stop error:", error);
    }
  };

  const toggleStrategy = (strategyId) => {
    setSelectedStrategies(prev => 
      prev.includes(strategyId) 
        ? prev.filter(s => s !== strategyId)
        : [...prev, strategyId]
    );
  };

  const progressPercent = trainingStatus?.total_epochs > 0
    ? (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100
    : 0;

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const featureImportanceData = learnedPatterns?.feature_importance
    ? Object.entries(learnedPatterns.feature_importance).slice(0, 8)
        .map(([name, value]) => ({ name: name.replace(/_/g, ' '), importance: value }))
    : [];

  const isTraining = trainingStatus?.is_training;

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Model Training</h1>
          <p className="text-muted-foreground mt-1">Deep Learning Training Center</p>
        </div>
        {isTraining && (
          <Badge className="bg-success animate-pulse text-lg px-4 py-1">TRAINING IN PROGRESS</Badge>
        )}
      </div>

      {/* ==================== SECTION 1: CONFIGURATION ==================== */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Training Configuration */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Settings className="w-4 h-4 text-primary" />
              Training Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Mode Selection */}
            <div className="space-y-2">
              <Label className="text-xs">Training Mode</Label>
              <Select value={mode} onValueChange={setMode} disabled={isTraining}>
                <SelectTrigger className="bg-secondary border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pure_ml">Pure ML (Model Discovers)</SelectItem>
                  <SelectItem value="mathematical">Mathematical Only</SelectItem>
                  <SelectItem value="hybrid">Hybrid (ML + Math)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-xs">Symbol</Label>
                <Select value={symbol} onValueChange={setSymbol} disabled={isTraining}>
                  <SelectTrigger className="bg-secondary border-border h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                    <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Timeframe</Label>
                <Select value={timeframe} onValueChange={setTimeframe} disabled={isTraining}>
                  <SelectTrigger className="bg-secondary border-border h-9">
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
            </div>

            {/* Date Range */}
            <div className="space-y-2">
              <Label className="text-xs">Historical Data Range</Label>
              <div className="grid grid-cols-2 gap-2">
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isTraining}>
                      <CalendarIcon className="mr-2 h-3 w-3" />
                      {startDate ? format(startDate, "MMM dd, yy") : "Start"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={startDate} onSelect={setStartDate} /></PopoverContent>
                </Popover>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isTraining}>
                      <CalendarIcon className="mr-2 h-3 w-3" />
                      {endDate ? format(endDate, "MMM dd, yy") : "End"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={endDate} onSelect={setEndDate} /></PopoverContent>
                </Popover>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-xs">Epochs</Label>
                <Input type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 100)} className="bg-secondary border-border h-9" disabled={isTraining} />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Batch Size</Label>
                <Input type="number" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)} className="bg-secondary border-border h-9" disabled={isTraining} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Network Architecture */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Network className="w-4 h-4 text-primary" />
              Network Architecture
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Network Type */}
            <div className="space-y-2">
              <Label className="text-xs">Network Type</Label>
              <div className="grid grid-cols-2 gap-2">
                {NETWORK_TYPES.map((type) => (
                  <div
                    key={type.id}
                    className={`p-2 rounded border cursor-pointer transition-all text-center ${
                      type.disabled ? 'opacity-40 cursor-not-allowed' :
                      networkType === type.id
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-secondary/30 hover:border-primary/50'
                    }`}
                    onClick={() => !type.disabled && !isTraining && setNetworkType(type.id)}
                  >
                    <p className="font-mono text-xs font-semibold">{type.name}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Architecture Summary */}
            <div className="p-2 bg-primary/10 rounded border border-primary/30">
              <p className="text-xs font-mono text-primary">
                Input → {numLstmLayers}x Bi-LSTM({lstmUnits.slice(0, numLstmLayers).join(',')}) 
                {useAttention && ' → Attn'} → {numDenseLayers}x Dense({denseUnits.slice(0, numDenseLayers).join(',')}) → Out
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Network Layers */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Layers className="w-4 h-4 text-primary" />
              Network Layers
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* LSTM Layers */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs">LSTM Layers</Label>
                <Badge variant="outline" className="font-mono text-xs">{numLstmLayers}</Badge>
              </div>
              <Slider value={[numLstmLayers]} onValueChange={([v]) => setNumLstmLayers(v)} min={1} max={4} step={1} disabled={isTraining} />
              {[...Array(numLstmLayers)].map((_, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs w-10">L{i + 1}:</span>
                  <Slider
                    className="flex-1"
                    value={[lstmUnits[i] || 64]}
                    onValueChange={([v]) => {
                      const newUnits = [...lstmUnits];
                      newUnits[i] = v;
                      setLstmUnits(newUnits);
                    }}
                    min={16} max={256} step={16}
                    disabled={isTraining}
                  />
                  <span className="font-mono text-xs w-8 text-right text-primary">{lstmUnits[i] || 64}</span>
                </div>
              ))}
            </div>

            <Separator />

            {/* Dense Layers */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs">Dense Layers</Label>
                <Badge variant="outline" className="font-mono text-xs">{numDenseLayers}</Badge>
              </div>
              <Slider value={[numDenseLayers]} onValueChange={([v]) => setNumDenseLayers(v)} min={1} max={4} step={1} disabled={isTraining} />
              {[...Array(numDenseLayers)].map((_, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs w-10">L{i + 1}:</span>
                  <Slider
                    className="flex-1"
                    value={[denseUnits[i] || 32]}
                    onValueChange={([v]) => {
                      const newUnits = [...denseUnits];
                      newUnits[i] = v;
                      setDenseUnits(newUnits);
                    }}
                    min={8} max={128} step={8}
                    disabled={isTraining}
                  />
                  <span className="font-mono text-xs w-8 text-right text-primary">{denseUnits[i] || 32}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Hyperparameters */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Cpu className="w-4 h-4 text-primary" />
              Hyperparameters
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <Label>Dropout Rate</Label>
                <span className="font-mono text-primary">{dropoutRate.toFixed(2)}</span>
              </div>
              <Slider value={[dropoutRate * 100]} onValueChange={([v]) => setDropoutRate(v / 100)} min={0} max={50} step={5} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <Label>Learning Rate</Label>
                <span className="font-mono text-primary">{learningRate.toFixed(4)}</span>
              </div>
              <Slider value={[Math.log10(learningRate) + 4]} onValueChange={([v]) => setLearningRate(Math.pow(10, v - 4))} min={1} max={3} step={0.5} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <Label>Sequence Length</Label>
                <span className="font-mono text-primary">{sequenceLength}</span>
              </div>
              <Slider value={[sequenceLength]} onValueChange={([v]) => setSequenceLength(v)} min={10} max={100} step={10} disabled={isTraining} />
            </div>

            <Separator />

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Attention Mechanism</Label>
                <Switch checked={useAttention} onCheckedChange={setUseAttention} disabled={isTraining} />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-xs">Batch Normalization</Label>
                <Switch checked={useBatchNorm} onCheckedChange={setUseBatchNorm} disabled={isTraining} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Mathematical Strategies (conditional) */}
      {(mode === 'mathematical' || mode === 'hybrid') && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Calculator className="w-4 h-4 text-primary" />
              Mathematical Strategies
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
              {MATH_STRATEGIES.map((strategy) => (
                <div
                  key={strategy.id}
                  className={`p-3 rounded border cursor-pointer transition-all ${
                    selectedStrategies.includes(strategy.id)
                      ? 'border-primary bg-primary/10'
                      : 'border-border bg-secondary/30 hover:border-primary/50'
                  } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                  onClick={() => !isTraining && toggleStrategy(strategy.id)}
                >
                  <div className="flex items-center gap-2">
                    <Checkbox checked={selectedStrategies.includes(strategy.id)} disabled={isTraining} />
                    <span className="font-mono text-xs">{strategy.name}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ==================== START/STOP BUTTONS ==================== */}
      <div className="flex justify-center gap-4">
        <Button 
          data-testid="start-training-btn"
          size="lg"
          onClick={handleStartTraining} 
          disabled={loading || isTraining}
          className="gap-2 bg-success hover:bg-success/90 text-white px-8 py-6 text-lg"
        >
          {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <PlayCircle className="w-6 h-6" />}
          Start Training
        </Button>
        <Button 
          data-testid="stop-training-btn"
          variant="destructive"
          size="lg"
          onClick={handleStopTraining}
          disabled={!isTraining}
          className="gap-2 px-8 py-6 text-lg"
        >
          <StopCircle className="w-6 h-6" />
          Stop Training
        </Button>
      </div>

      {/* ==================== SECTION 2: TRAINING PROGRESS ==================== */}
      <Card className={`bg-card border-2 ${isTraining ? 'border-primary' : 'border-border'}`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Activity className={`w-5 h-5 ${isTraining ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
              Training Progress
              {isTraining && <Badge className="bg-success ml-2 animate-pulse">LIVE</Badge>}
            </CardTitle>
            <div className="flex items-center gap-4 text-sm">
              <span className="text-muted-foreground">Elapsed: <span className="font-mono text-primary">{formatTime(elapsedTime)}</span></span>
              <span className="text-muted-foreground">Mode: <Badge variant="outline">{trainingStatus?.mode || mode}</Badge></span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-mono">
                Epoch <span className="text-primary text-xl font-bold">{trainingStatus?.current_epoch || 0}</span> / {trainingStatus?.total_epochs || epochs}
              </span>
              <span className="font-mono text-primary text-xl font-bold">{progressPercent.toFixed(1)}%</span>
            </div>
            <Progress value={progressPercent} className="h-4" />
          </div>

          {/* Live Metrics */}
          <div className="grid grid-cols-5 gap-4">
            <div className="p-4 bg-secondary rounded-lg text-center">
              <p className="text-xs text-muted-foreground uppercase">Loss</p>
              <p className="font-mono text-2xl text-destructive">{trainingStatus?.current_loss?.toFixed(4) || '0.0000'}</p>
            </div>
            <div className="p-4 bg-secondary rounded-lg text-center">
              <p className="text-xs text-muted-foreground uppercase">Accuracy</p>
              <p className="font-mono text-2xl text-success">{((trainingStatus?.current_accuracy || 0) * 100).toFixed(1)}%</p>
            </div>
            <div className="p-4 bg-secondary rounded-lg text-center">
              <p className="text-xs text-muted-foreground uppercase">Val Loss</p>
              <p className="font-mono text-2xl text-warning">
                {trainingStatus?.history?.length > 0 
                  ? trainingStatus.history[trainingStatus.history.length - 1]?.val_loss?.toFixed(4) 
                  : '0.0000'}
              </p>
            </div>
            <div className="p-4 bg-secondary rounded-lg text-center">
              <p className="text-xs text-muted-foreground uppercase">Val Accuracy</p>
              <p className="font-mono text-2xl text-primary">
                {trainingStatus?.history?.length > 0 
                  ? ((trainingStatus.history[trainingStatus.history.length - 1]?.val_accuracy || 0) * 100).toFixed(1)
                  : '0.0'}%
              </p>
            </div>
            <div className="p-4 bg-secondary rounded-lg text-center">
              <p className="text-xs text-muted-foreground uppercase">Samples</p>
              <p className="font-mono text-2xl">{trainingStatus?.data_info?.training_samples?.toLocaleString() || 0}</p>
            </div>
          </div>

          {/* Live Charts */}
          <div className="grid grid-cols-2 gap-4">
            {/* Loss Chart */}
            <div className="p-4 bg-secondary/30 rounded-lg">
              <h4 className="text-sm font-mono text-muted-foreground mb-2">Loss Curve (Real-time)</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trainingStatus?.history || []}>
                    <defs>
                      <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#FF2E55" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#FF2E55" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="valLossGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#00E5FF" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                    <XAxis dataKey="epoch" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} />
                    <Area type="monotone" dataKey="loss" stroke="#FF2E55" fill="url(#lossGrad)" strokeWidth={2} name="Train Loss" />
                    <Area type="monotone" dataKey="val_loss" stroke="#00E5FF" fill="url(#valLossGrad)" strokeWidth={2} name="Val Loss" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-4 mt-2 text-xs">
                <span className="flex items-center gap-1"><span className="w-3 h-1 bg-[#FF2E55] rounded"></span> Train Loss</span>
                <span className="flex items-center gap-1"><span className="w-3 h-1 bg-[#00E5FF] rounded"></span> Val Loss</span>
              </div>
            </div>

            {/* Accuracy Chart */}
            <div className="p-4 bg-secondary/30 rounded-lg">
              <h4 className="text-sm font-mono text-muted-foreground mb-2">Accuracy Curve (Real-time)</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trainingStatus?.history || []}>
                    <defs>
                      <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00FF94" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#00FF94" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="valAccGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#B026FF" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#B026FF" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                    <XAxis dataKey="epoch" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} domain={[0, 1]} tickFormatter={(v) => `${(v*100).toFixed(0)}%`} />
                    <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} formatter={(v) => `${(v*100).toFixed(1)}%`} />
                    <Area type="monotone" dataKey="accuracy" stroke="#00FF94" fill="url(#accGrad)" strokeWidth={2} name="Train Acc" />
                    <Area type="monotone" dataKey="val_accuracy" stroke="#B026FF" fill="url(#valAccGrad)" strokeWidth={2} name="Val Acc" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-center gap-4 mt-2 text-xs">
                <span className="flex items-center gap-1"><span className="w-3 h-1 bg-[#00FF94] rounded"></span> Train Acc</span>
                <span className="flex items-center gap-1"><span className="w-3 h-1 bg-[#B026FF] rounded"></span> Val Acc</span>
              </div>
            </div>
          </div>

          {/* Learned Patterns */}
          {learnedPatterns?.model_equation && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-primary/10 rounded-lg border border-primary/30">
                <h4 className="font-mono text-sm text-primary mb-2 flex items-center gap-2">
                  <Zap className="w-4 h-4" /> Learned Equation
                </h4>
                <p className="font-mono text-sm">{learnedPatterns.model_equation}</p>
              </div>
              {featureImportanceData.length > 0 && (
                <div className="p-4 bg-secondary/30 rounded-lg">
                  <h4 className="font-mono text-sm text-muted-foreground mb-2">Top Features</h4>
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportanceData} layout="vertical">
                        <XAxis type="number" tick={{ fill: '#A1A1AA', fontSize: 9 }} />
                        <YAxis dataKey="name" type="category" tick={{ fill: '#A1A1AA', fontSize: 9 }} width={80} />
                        <Bar dataKey="importance" fill="#00E5FF" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ==================== SECTION 3: TRAINING HISTORY ==================== */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Clock className="w-5 h-5 text-primary" /> Training History
            </CardTitle>
            <Button variant="outline" size="sm" onClick={fetchHistory}>
              <RefreshCw className="w-4 h-4 mr-1" /> Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {trainingHistory.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {trainingHistory.map((session, i) => (
                <div key={i} className="p-4 bg-secondary rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-mono text-sm font-semibold">{session.symbol}</p>
                      <p className="text-xs text-muted-foreground">{new Date(session.created_at).toLocaleString()}</p>
                      <div className="flex gap-1 mt-2">
                        <Badge className="text-xs" variant="outline">{session.config?.mode || 'pure_ml'}</Badge>
                        <Badge className="text-xs" variant="outline">{session.config?.timeframe || '1h'}</Badge>
                      </div>
                    </div>
                    <Badge className={session.result?.status === 'completed' ? 'bg-success' : 'bg-warning'}>
                      {((session.result?.best_accuracy || session.result?.final_accuracy || 0) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">No training history yet. Start training to see results here.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Training;
