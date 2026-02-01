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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Area, AreaChart
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Activity, Clock, Cpu,
  CalendarIcon, TrendingUp, Zap, Layers, Settings, Calculator,
  GitBranch, Sigma, Network, StopCircle, PlayCircle, Save,
  FolderOpen, Database, Sparkles, Target, BarChart3, Shuffle
} from "lucide-react";
import { format } from "date-fns";
import { API } from "@/lib/api";

// API functions
const startAdvancedTraining = async (config) => {
  const response = await API.post('/training/advanced/start', config);
  return response.data;
};

const getAdvancedTrainingStatus = async () => {
  const response = await API.get('/training/advanced/status');
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

const getSavedModels = async () => {
  const response = await API.get('/models/saved');
  return response.data;
};

const loadSavedModel = async (modelPath) => {
  const response = await API.post('/models/load', null, { params: { model_path: modelPath } });
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
  { id: "lstm", name: "LSTM", description: "Long Short-Term Memory - Best for sequences", icon: "🧠" },
  { id: "gru", name: "GRU", description: "Gated Recurrent Unit - Faster training", icon: "⚡" },
  { id: "transformer", name: "Transformer", description: "Multi-head Attention - State of art", icon: "🔮" },
  { id: "cnn_lstm", name: "CNN + LSTM", description: "Convolutional + Recurrent hybrid", icon: "🔗" },
  { id: "ensemble", name: "Ensemble", description: "LSTM + XGBoost + RandomForest", icon: "🎯" },
  { id: "tft", name: "TFT", description: "Temporal Fusion Transformer - Google's best", icon: "🌟" },
  { id: "multi_task", name: "Multi-Task", description: "Predict direction + volatility + magnitude", icon: "📊" },
  { id: "gnn", name: "GNN", description: "Graph Neural Network - Asset relationships", icon: "🕸️" },
  { id: "multi_tf_attention", name: "Multi-TF Attn", description: "Separate attention per timeframe", icon: "⏱️" },
  { id: "rl_dqn", name: "RL - DQN", description: "Deep Q-Network - Learns by trading simulation", icon: "🤖" },
  { id: "rl_ppo", name: "RL - PPO", description: "Proximal Policy Optimization - Advanced RL", icon: "🎮" },
];

// Models available for multi-model ensemble
const MULTI_MODEL_OPTIONS = [
  { id: "lstm", name: "LSTM", icon: "🧠", description: "Sequential patterns", color: "from-pink-500/20 to-purple-500/20", borderColor: "border-pink-500" },
  { id: "gru", name: "GRU", icon: "⚡", description: "Fast & efficient", color: "from-yellow-500/20 to-orange-500/20", borderColor: "border-yellow-500" },
  { id: "transformer", name: "Transformer", icon: "🔮", description: "Attention-based", color: "from-violet-500/20 to-indigo-500/20", borderColor: "border-violet-500" },
  { id: "cnn_lstm", name: "CNN+LSTM", icon: "🔗", description: "Hybrid power", color: "from-cyan-500/20 to-blue-500/20", borderColor: "border-cyan-500" },
];

const LR_SCHEDULES = [
  { id: "reduce_plateau", name: "Reduce on Plateau", description: "Reduce LR when loss plateaus" },
  { id: "cosine", name: "Cosine Annealing", description: "Smooth cosine decay" },
  { id: "step", name: "Step Decay", description: "Reduce LR every N epochs" },
];

const Training = () => {
  // Basic config
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  
  // Exchange selection
  const [selectedExchange, setSelectedExchange] = useState("okx");
  const [exchangeStatus, setExchangeStatus] = useState(null);
  
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
  
  // Advanced options
  const [useEarlyStopping, setUseEarlyStopping] = useState(true);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(15);
  const [lrSchedule, setLrSchedule] = useState("reduce_plateau");
  const [useWalkForward, setUseWalkForward] = useState(false);
  const [cvFolds, setCvFolds] = useState(5);
  const [useOptuna, setUseOptuna] = useState(false);
  const [optunTrials, setOptunTrials] = useState(20);
  const [classBalanceMethod, setClassBalanceMethod] = useState("class_weight");
  const [multiTimeframe, setMultiTimeframe] = useState(false);
  const [saveModel, setSaveModel] = useState(true);
  
  // Strategies
  const [selectedStrategies, setSelectedStrategies] = useState([]);
  
  // Multi-Model Config
  const [isMultiModel, setIsMultiModel] = useState(false);
  const [selectedModels, setSelectedModels] = useState(["lstm", "gru", "transformer"]);
  const [ensembleMethod, setEnsembleMethod] = useState("weighted");
  
  // RL Config
  const [rlEpisodes, setRlEpisodes] = useState(100);
  const [rlGamma, setRlGamma] = useState(0.99);
  
  // Status
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [savedModels, setSavedModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [learnedPatterns, setLearnedPatterns] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const intervalRef = useRef(null);
  const timerRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getAdvancedTrainingStatus();
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

  const fetchSavedModels = useCallback(async () => {
    try {
      const result = await getSavedModels();
      setSavedModels(result.models || []);
    } catch (error) {
      console.error("Error fetching saved models:", error);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchStatus();
    fetchSavedModels();
    fetchExchangeStatus();
  }, [fetchHistory, fetchStatus, fetchSavedModels]);

  // Fetch exchange status
  const fetchExchangeStatus = useCallback(async () => {
    try {
      const response = await API.get('/exchange/status');
      setExchangeStatus(response.data);
      if (response.data?.active_exchange) {
        setSelectedExchange(response.data.active_exchange);
      }
    } catch (error) {
      console.error("Error fetching exchange status:", error);
    }
  }, []);

  // Set active exchange
  const handleExchangeChange = async (exchange) => {
    try {
      await API.post(`/exchange/set-active?exchange=${exchange}`);
      setSelectedExchange(exchange);
      toast.success(`Switched to ${exchange.toUpperCase()} for data`);
      fetchExchangeStatus();
    } catch (error) {
      toast.error(`Failed to switch to ${exchange}`);
    }
  };

  // Test exchange connection
  const handleTestExchange = async () => {
    try {
      const response = await API.post('/exchange/test', { exchange: selectedExchange });
      if (response.data.success) {
        toast.success(`${selectedExchange.toUpperCase()} connected! BTC: $${response.data.ticker?.last?.toLocaleString()}`);
      } else {
        toast.error(`Connection failed: ${response.data.error}`);
      }
    } catch (error) {
      toast.error("Connection test failed");
    }
  };

  // Real-time status polling
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
      // Set the exchange before training
      await API.post(`/exchange/set-active?exchange=${selectedExchange}`);
      
      const config = {
        symbol,
        epochs,
        batch_size: batchSize,
        start_date: startDate ? startDate.toISOString() : null,
        end_date: endDate ? endDate.toISOString() : null,
        timeframe,
        mode,
        exchange: selectedExchange,  // Pass exchange to training
        network_type: isMultiModel ? 'multi_model' : networkType,
        strategies: selectedStrategies,
        num_lstm_layers: numLstmLayers,
        lstm_units: lstmUnits.slice(0, numLstmLayers),
        num_dense_layers: numDenseLayers,
        dense_units: denseUnits.slice(0, numDenseLayers),
        dropout_rate: dropoutRate,
        use_attention: useAttention,
        use_batch_norm: useBatchNorm,
        learning_rate: learningRate,
        sequence_length: sequenceLength,
        use_early_stopping: useEarlyStopping,
        early_stopping_patience: earlyStoppingPatience,
        lr_schedule: lrSchedule,
        use_walk_forward: useWalkForward,
        cv_folds: cvFolds,
        use_optuna: useOptuna,
        optuna_trials: optunTrials,
        class_balance_method: classBalanceMethod,
        multi_timeframe: multiTimeframe,
        save_model: saveModel,
        // Multi-Model Config
        is_multi_model: isMultiModel,
        selected_models: selectedModels,
        ensemble_method: ensembleMethod,
        // RL Config
        rl_episodes: rlEpisodes,
        rl_gamma: rlGamma
      };
      
      const result = await startAdvancedTraining(config);
      toast.success(`Training started with ${networkType.toUpperCase()} model!`);
      setTimeout(fetchStatus, 500);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Failed to start training";
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopTraining();
      toast.success("Training stopped");
      fetchStatus();
      fetchHistory();
      fetchSavedModels();
    } catch (error) {
      toast.error("Failed to stop training");
    }
  };

  const handleLoadModel = async (modelPath) => {
    try {
      await loadSavedModel(modelPath);
      toast.success("Model loaded successfully");
      fetchStatus();
    } catch (error) {
      toast.error("Failed to load model");
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

  const isTraining = trainingStatus?.is_training;

  const featureImportanceData = learnedPatterns?.feature_importance
    ? Object.entries(learnedPatterns.feature_importance).slice(0, 8)
        .map(([name, value]) => ({ name: name.replace(/_/g, ' '), importance: value }))
    : [];

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Advanced Model Training</h1>
          <p className="text-muted-foreground mt-1">Deep Learning Training Center with Multiple Architectures</p>
        </div>
        {isTraining && (
          <Badge className="bg-success animate-pulse text-lg px-4 py-1">TRAINING IN PROGRESS</Badge>
        )}
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="config" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="config" className="gap-2"><Settings className="w-4 h-4" /> Configuration</TabsTrigger>
          <TabsTrigger value="advanced" className="gap-2"><Sparkles className="w-4 h-4" /> Advanced Options</TabsTrigger>
          <TabsTrigger value="models" className="gap-2"><Database className="w-4 h-4" /> Saved Models</TabsTrigger>
          <TabsTrigger value="history" className="gap-2"><Clock className="w-4 h-4" /> History</TabsTrigger>
        </TabsList>

        {/* ==================== CONFIGURATION TAB ==================== */}
        <TabsContent value="config" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            
            {/* Training Configuration */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Settings className="w-4 h-4 text-primary" />
                  Training Config
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Exchange Selector */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Data Source
                    {exchangeStatus?.exchanges?.[selectedExchange]?.authenticated && (
                      <Badge className="bg-green-500/20 text-green-400 text-[10px]">Authenticated</Badge>
                    )}
                  </Label>
                  <div className="flex gap-2">
                    <Select value={selectedExchange} onValueChange={handleExchangeChange} disabled={isTraining}>
                      <SelectTrigger className="bg-secondary border-border flex-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="okx">
                          <span className="flex items-center gap-2">🟢 OKX</span>
                        </SelectItem>
                        <SelectItem value="binance">
                          <span className="flex items-center gap-2">🟡 Binance</span>
                        </SelectItem>
                        <SelectItem value="kucoin">
                          <span className="flex items-center gap-2">🟢 KuCoin</span>
                        </SelectItem>
                        <SelectItem value="bybit">
                          <span className="flex items-center gap-2">🟠 Bybit</span>
                        </SelectItem>
                        <SelectItem value="kraken">
                          <span className="flex items-center gap-2">🔵 Kraken</span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={handleTestExchange}
                      disabled={isTraining}
                      className="px-3"
                    >
                      Test
                    </Button>
                  </div>
                </div>

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

                <div className="space-y-2">
                  <Label className="text-xs">Date Range</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isTraining}>
                          <CalendarIcon className="mr-1 h-3 w-3" />
                          {startDate ? format(startDate, "MMM dd") : "Start"}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={startDate} onSelect={setStartDate} /></PopoverContent>
                    </Popover>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs justify-start" disabled={isTraining}>
                          <CalendarIcon className="mr-1 h-3 w-3" />
                          {endDate ? format(endDate, "MMM dd") : "End"}
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

            {/* Network Type Selection */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Network className="w-4 h-4 text-primary" />
                  Network Architecture
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Single/Multi Model Toggle */}
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg border border-border">
                  <div className="flex items-center gap-3">
                    <span className={`text-sm font-mono ${!isMultiModel ? 'text-primary font-semibold' : 'text-muted-foreground'}`}>
                      Single Model
                    </span>
                    <Switch 
                      checked={isMultiModel} 
                      onCheckedChange={setIsMultiModel}
                      disabled={isTraining}
                    />
                    <span className={`text-sm font-mono ${isMultiModel ? 'text-primary font-semibold' : 'text-muted-foreground'}`}>
                      Multi-Model
                    </span>
                  </div>
                  {isMultiModel && (
                    <Badge className="bg-primary/20 text-primary border-primary/30">
                      <Shuffle className="w-3 h-3 mr-1" />
                      Ensemble Mode
                    </Badge>
                  )}
                </div>

                {/* Single Model Selection */}
                {!isMultiModel ? (
                  <>
                    <div className="grid grid-cols-3 gap-2">
                      {NETWORK_TYPES.map((type) => (
                        <div
                          key={type.id}
                          className={`p-2 rounded-lg border cursor-pointer transition-all ${
                            networkType === type.id
                              ? 'border-primary bg-primary/10 shadow-lg shadow-primary/20'
                              : 'border-border bg-secondary/30 hover:border-primary/50'
                          } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                          onClick={() => !isTraining && setNetworkType(type.id)}
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-lg">{type.icon}</span>
                            <div className="flex-1 min-w-0">
                              <p className="font-mono text-xs font-semibold truncate">{type.name}</p>
                              <p className="text-[10px] text-muted-foreground truncate">{type.description}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    {/* Selected Network Info */}
                    <div className="p-2 bg-primary/10 rounded border border-primary/30">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-lg">{NETWORK_TYPES.find(t => t.id === networkType)?.icon}</span>
                        <span className="font-mono text-sm font-semibold">{NETWORK_TYPES.find(t => t.id === networkType)?.name}</span>
                        <Badge className="bg-primary text-[10px]">Selected</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{NETWORK_TYPES.find(t => t.id === networkType)?.description}</p>
                    </div>
                  </>
                ) : (
                  /* Multi-Model Selection */
                  <div className="space-y-5">
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">
                        Select models to train together. Combined predictions yield higher accuracy.
                      </p>
                      <Badge variant="outline" className="text-xs">
                        Min 2 required
                      </Badge>
                    </div>
                    
                    {/* Beautiful Model Cards */}
                    <div className="grid grid-cols-2 gap-3">
                      {MULTI_MODEL_OPTIONS.map((model) => {
                        const isSelected = selectedModels.includes(model.id);
                        const canDeselect = selectedModels.length > 2;
                        
                        return (
                          <div
                            key={model.id}
                            className={`relative group cursor-pointer transition-all duration-300 ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                            onClick={() => {
                              if (isTraining) return;
                              if (isSelected) {
                                if (canDeselect) {
                                  setSelectedModels(prev => prev.filter(m => m !== model.id));
                                }
                              } else {
                                setSelectedModels(prev => [...prev, model.id]);
                              }
                            }}
                          >
                            {/* Card */}
                            <div className={`
                              relative overflow-hidden rounded-xl border-2 p-4 transition-all duration-300
                              ${isSelected 
                                ? `${model.borderColor} bg-gradient-to-br ${model.color} shadow-lg` 
                                : 'border-border/50 bg-secondary/20 hover:border-primary/30 hover:bg-secondary/40'
                              }
                            `}>
                              {/* Selection indicator */}
                              <div className={`
                                absolute top-2 right-2 w-5 h-5 rounded-full flex items-center justify-center transition-all
                                ${isSelected 
                                  ? 'bg-primary text-primary-foreground scale-100' 
                                  : 'bg-muted/50 scale-90 group-hover:scale-100'
                                }
                              `}>
                                {isSelected ? (
                                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                  </svg>
                                ) : (
                                  <span className="w-2 h-2 rounded-full bg-muted-foreground/30"></span>
                                )}
                              </div>
                              
                              {/* Content */}
                              <div className="flex flex-col items-center text-center space-y-2">
                                <div className={`
                                  text-3xl p-3 rounded-xl transition-transform duration-300
                                  ${isSelected ? 'scale-110 bg-background/30' : 'group-hover:scale-105'}
                                `}>
                                  {model.icon}
                                </div>
                                <div>
                                  <h4 className={`font-mono font-bold text-sm ${isSelected ? 'text-foreground' : 'text-muted-foreground group-hover:text-foreground'}`}>
                                    {model.name}
                                  </h4>
                                  <p className="text-[10px] text-muted-foreground mt-0.5">
                                    {model.description}
                                  </p>
                                </div>
                              </div>
                              
                              {/* Glow effect when selected */}
                              {isSelected && (
                                <div className="absolute inset-0 bg-gradient-to-t from-primary/5 to-transparent pointer-events-none" />
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    
                    {/* Ensemble Method Selection */}
                    <div className="space-y-2">
                      <Label className="text-xs font-mono flex items-center gap-2">
                        <Shuffle className="w-3 h-3" />
                        Ensemble Method
                      </Label>
                      <Select value={ensembleMethod} onValueChange={setEnsembleMethod} disabled={isTraining}>
                        <SelectTrigger className="bg-secondary/50 border-border h-10">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="voting">
                            <span className="flex items-center gap-2">
                              <span>🗳️</span> Simple Voting
                            </span>
                          </SelectItem>
                          <SelectItem value="weighted">
                            <span className="flex items-center gap-2">
                              <span>⚖️</span> Weighted Voting (recommended)
                            </span>
                          </SelectItem>
                          <SelectItem value="stacking">
                            <span className="flex items-center gap-2">
                              <span>📚</span> Stacking (XGBoost meta-learner)
                            </span>
                          </SelectItem>
                          <SelectItem value="blending">
                            <span className="flex items-center gap-2">
                              <span>🎨</span> Blending (optimized weights)
                            </span>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {/* Summary Card */}
                    <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-primary/10 via-primary/5 to-transparent border border-primary/30 p-4">
                      <div className="flex items-start justify-between">
                        <div className="space-y-3">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
                              <Shuffle className="w-4 h-4 text-primary" />
                            </div>
                            <div>
                              <span className="font-mono text-lg font-bold text-primary">
                                {selectedModels.length}
                              </span>
                              <span className="text-sm text-muted-foreground ml-1">Models Selected</span>
                            </div>
                          </div>
                          
                          <div className="flex flex-wrap gap-2">
                            {selectedModels.map(m => {
                              const modelInfo = MULTI_MODEL_OPTIONS.find(opt => opt.id === m);
                              return (
                                <Badge 
                                  key={m} 
                                  className={`bg-gradient-to-r ${modelInfo?.color} border ${modelInfo?.borderColor} text-foreground font-mono text-xs px-2 py-1`}
                                >
                                  {modelInfo?.icon} {m.toUpperCase()}
                                </Badge>
                              );
                            })}
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Method</p>
                          <p className="font-mono text-sm text-primary font-semibold capitalize">{ensembleMethod}</p>
                        </div>
                      </div>
                      
                      {/* Decorative element */}
                      <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-primary/5 rounded-full blur-2xl" />
                    </div>
                  </div>
                )}
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
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label className="text-xs">LSTM/GRU Layers</Label>
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

                {/* Architecture Summary */}
                <div className="p-2 bg-primary/10 rounded border border-primary/30">
                  <p className="text-xs font-mono text-primary">
                    {networkType === 'ensemble' 
                      ? 'LSTM + XGBoost + RandomForest Ensemble'
                      : `Input → ${numLstmLayers}x ${networkType.toUpperCase()}(${lstmUnits.slice(0, numLstmLayers).join(',')}) ${useAttention ? '→ Attn' : ''} → ${numDenseLayers}x Dense(${denseUnits.slice(0, numDenseLayers).join(',')}) → Out`
                    }
                  </p>
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
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">Save Model After Training</Label>
                    <Switch checked={saveModel} onCheckedChange={setSaveModel} disabled={isTraining} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Mathematical Strategies */}
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
        </TabsContent>

        {/* ==================== ADVANCED OPTIONS TAB ==================== */}
        <TabsContent value="advanced" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Training Process */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary" />
                  Training Process
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <div>
                    <Label className="text-sm">Early Stopping</Label>
                    <p className="text-xs text-muted-foreground">Stop when validation loss plateaus</p>
                  </div>
                  <Switch checked={useEarlyStopping} onCheckedChange={setUseEarlyStopping} disabled={isTraining} />
                </div>

                {useEarlyStopping && (
                  <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                    <div className="flex justify-between text-xs">
                      <Label>Patience (epochs)</Label>
                      <span className="font-mono text-primary">{earlyStoppingPatience}</span>
                    </div>
                    <Slider value={[earlyStoppingPatience]} onValueChange={([v]) => setEarlyStoppingPatience(v)} min={5} max={30} step={5} disabled={isTraining} />
                  </div>
                )}

                <Separator />

                <div className="space-y-2">
                  <Label className="text-xs">Learning Rate Schedule</Label>
                  <Select value={lrSchedule} onValueChange={setLrSchedule} disabled={isTraining}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LR_SCHEDULES.map((schedule) => (
                        <SelectItem key={schedule.id} value={schedule.id}>
                          {schedule.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {LR_SCHEDULES.find(s => s.id === lrSchedule)?.description}
                  </p>
                </div>

                <Separator />

                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <div>
                    <Label className="text-sm">Multi-Timeframe</Label>
                    <p className="text-xs text-muted-foreground">Use 5m, 15m, 1h, 4h, 1d data</p>
                  </div>
                  <Switch checked={multiTimeframe} onCheckedChange={setMultiTimeframe} disabled={isTraining} />
                </div>
              </CardContent>
            </Card>

            {/* Validation & Search */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Validation & Search
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <div>
                    <Label className="text-sm">Walk-Forward Validation</Label>
                    <p className="text-xs text-muted-foreground">Time-series cross-validation</p>
                  </div>
                  <Switch checked={useWalkForward} onCheckedChange={setUseWalkForward} disabled={isTraining} />
                </div>

                {useWalkForward && (
                  <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                    <div className="flex justify-between text-xs">
                      <Label>CV Folds</Label>
                      <span className="font-mono text-primary">{cvFolds}</span>
                    </div>
                    <Slider value={[cvFolds]} onValueChange={([v]) => setCvFolds(v)} min={3} max={10} step={1} disabled={isTraining} />
                  </div>
                )}

                <Separator />

                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <div>
                    <Label className="text-sm">Optuna Hyperparameter Search</Label>
                    <p className="text-xs text-muted-foreground">Auto-tune model parameters</p>
                  </div>
                  <Switch checked={useOptuna} onCheckedChange={setUseOptuna} disabled={isTraining} />
                </div>

                {useOptuna && (
                  <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                    <div className="flex justify-between text-xs">
                      <Label>Number of Trials</Label>
                      <span className="font-mono text-primary">{optunTrials}</span>
                    </div>
                    <Slider value={[optunTrials]} onValueChange={([v]) => setOptunTrials(v)} min={10} max={50} step={5} disabled={isTraining} />
                    <p className="text-xs text-muted-foreground">More trials = better params but longer time</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Class Balancing */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Shuffle className="w-4 h-4 text-primary" />
                  Class Balancing
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground">
                  Handle imbalanced up/down days in training data
                </p>

                <div className="space-y-3">
                  <div
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      classBalanceMethod === 'class_weight'
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-secondary/30 hover:border-primary/50'
                    }`}
                    onClick={() => !isTraining && setClassBalanceMethod('class_weight')}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${classBalanceMethod === 'class_weight' ? 'bg-primary' : 'bg-muted'}`}></div>
                      <div>
                        <p className="font-mono text-sm">Class Weights</p>
                        <p className="text-xs text-muted-foreground">Adjust loss function weights (recommended)</p>
                      </div>
                    </div>
                  </div>

                  <div
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      classBalanceMethod === 'smote'
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-secondary/30 hover:border-primary/50'
                    }`}
                    onClick={() => !isTraining && setClassBalanceMethod('smote')}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${classBalanceMethod === 'smote' ? 'bg-primary' : 'bg-muted'}`}></div>
                      <div>
                        <p className="font-mono text-sm">SMOTE Oversampling</p>
                        <p className="text-xs text-muted-foreground">Generate synthetic minority samples</p>
                      </div>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Feature Summary */}
                <div className="p-3 bg-secondary/30 rounded-lg">
                  <h4 className="font-mono text-xs text-muted-foreground mb-2">Active Features:</h4>
                  <div className="flex flex-wrap gap-1">
                    {!useEarlyStopping && <Badge variant="outline" className="text-xs">No Early Stop</Badge>}
                    {useWalkForward && <Badge variant="outline" className="text-xs">Walk-Forward CV</Badge>}
                    {useOptuna && <Badge variant="outline" className="text-xs">Optuna Search</Badge>}
                    {multiTimeframe && <Badge variant="outline" className="text-xs">Multi-TF</Badge>}
                    <Badge variant="outline" className="text-xs">{classBalanceMethod === 'smote' ? 'SMOTE' : 'Class Weights'}</Badge>
                    <Badge variant="outline" className="text-xs">{lrSchedule}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* RL Configuration - Only shown when RL model is selected */}
          {networkType.startsWith('rl_') && (
            <Card className="bg-card border-border border-primary/50">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Zap className="w-4 h-4 text-primary" />
                  Reinforcement Learning Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    {networkType === 'rl_dqn' 
                      ? 'Deep Q-Network learns optimal trading actions (BUY/HOLD/SELL) through simulated trading experience.'
                      : 'Proximal Policy Optimization uses advanced policy gradient methods for stable learning.'}
                  </p>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <Label>Training Episodes</Label>
                        <span className="font-mono text-primary">{rlEpisodes}</span>
                      </div>
                      <Slider 
                        value={[rlEpisodes]} 
                        onValueChange={([v]) => setRlEpisodes(v)} 
                        min={50} max={500} step={50} 
                        disabled={isTraining}
                      />
                      <p className="text-xs text-muted-foreground">More episodes = better learning but longer training</p>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <Label>Discount Factor (γ)</Label>
                        <span className="font-mono text-primary">{rlGamma}</span>
                      </div>
                      <Slider 
                        value={[rlGamma * 100]} 
                        onValueChange={([v]) => setRlGamma(v / 100)} 
                        min={90} max={99} step={1} 
                        disabled={isTraining}
                      />
                      <p className="text-xs text-muted-foreground">Higher = values future rewards more</p>
                    </div>
                  </div>
                  
                  <div className="p-3 bg-primary/10 rounded-lg border border-primary/30">
                    <p className="text-xs font-mono">
                      {networkType === 'rl_dqn' ? '🤖 DQN Agent' : '🎮 PPO Agent'}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Actions: BUY (open long), HOLD (wait), SELL (open short)
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Reward: Trading profit/loss with transaction costs
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ==================== SAVED MODELS TAB ==================== */}
        <TabsContent value="models" className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-lg flex items-center gap-2">
                  <Database className="w-5 h-5 text-primary" />
                  Saved Models
                  {savedModels.length > 0 && (
                    <Badge variant="outline" className="ml-2">{savedModels.length} models</Badge>
                  )}
                </CardTitle>
                <Button variant="outline" size="sm" onClick={fetchSavedModels}>
                  <RefreshCw className="w-4 h-4 mr-1" /> Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {savedModels.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {savedModels.map((model, i) => {
                    // Get model icon and color based on network type
                    const getModelStyle = (type) => {
                      const styles = {
                        'lstm': { icon: '🧠', color: 'from-pink-500/20 to-purple-500/20', border: 'border-pink-500/50', badge: 'bg-pink-500/20 text-pink-300' },
                        'gru': { icon: '⚡', color: 'from-yellow-500/20 to-orange-500/20', border: 'border-yellow-500/50', badge: 'bg-yellow-500/20 text-yellow-300' },
                        'transformer': { icon: '🔮', color: 'from-violet-500/20 to-indigo-500/20', border: 'border-violet-500/50', badge: 'bg-violet-500/20 text-violet-300' },
                        'cnn_lstm': { icon: '🔗', color: 'from-cyan-500/20 to-blue-500/20', border: 'border-cyan-500/50', badge: 'bg-cyan-500/20 text-cyan-300' },
                        'ensemble': { icon: '🎯', color: 'from-green-500/20 to-emerald-500/20', border: 'border-green-500/50', badge: 'bg-green-500/20 text-green-300' },
                        'tft': { icon: '🌟', color: 'from-amber-500/20 to-yellow-500/20', border: 'border-amber-500/50', badge: 'bg-amber-500/20 text-amber-300' },
                        'multi_task': { icon: '📊', color: 'from-blue-500/20 to-indigo-500/20', border: 'border-blue-500/50', badge: 'bg-blue-500/20 text-blue-300' },
                        'gnn': { icon: '🕸️', color: 'from-rose-500/20 to-pink-500/20', border: 'border-rose-500/50', badge: 'bg-rose-500/20 text-rose-300' },
                        'rl_dqn': { icon: '🤖', color: 'from-emerald-500/20 to-teal-500/20', border: 'border-emerald-500/50', badge: 'bg-emerald-500/20 text-emerald-300' },
                        'rl_ppo': { icon: '🎮', color: 'from-purple-500/20 to-fuchsia-500/20', border: 'border-purple-500/50', badge: 'bg-purple-500/20 text-purple-300' },
                        'multi_model': { icon: '🔀', color: 'from-teal-500/20 to-cyan-500/20', border: 'border-teal-500/50', badge: 'bg-teal-500/20 text-teal-300' },
                      };
                      return styles[type] || styles['lstm'];
                    };
                    
                    const style = getModelStyle(model.network_type);
                    const accuracy = (model.metrics?.final_accuracy || 0) * 100;
                    
                    return (
                      <div 
                        key={i} 
                        className={`relative overflow-hidden rounded-xl border-2 ${style.border} bg-gradient-to-br ${style.color} hover:shadow-lg hover:shadow-primary/10 transition-all duration-300`}
                      >
                        {/* Model Type Header */}
                        <div className="px-4 pt-4 pb-2">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-2xl">{style.icon}</span>
                              <div>
                                <Badge className={`${style.badge} font-mono text-xs px-2 py-0.5`}>
                                  {model.network_type?.toUpperCase()}
                                </Badge>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="font-mono text-lg font-bold text-primary">{accuracy.toFixed(1)}%</p>
                              <p className="text-[10px] text-muted-foreground">accuracy</p>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2 mb-3">
                            <p className="font-mono text-sm font-semibold">{model.symbol}</p>
                            <span className="text-muted-foreground">•</span>
                            <p className="text-xs text-muted-foreground">{model.config?.timeframe || '1h'}</p>
                          </div>
                        </div>
                        
                        {/* Stats */}
                        <div className="px-4 pb-3 grid grid-cols-3 gap-2">
                          <div className="text-center p-2 bg-background/30 rounded-lg">
                            <p className="font-mono text-sm font-bold">{model.metrics?.epochs_trained || 0}</p>
                            <p className="text-[10px] text-muted-foreground">Epochs</p>
                          </div>
                          <div className="text-center p-2 bg-background/30 rounded-lg">
                            <p className="font-mono text-sm font-bold">{model.config?.sequence_length || 50}</p>
                            <p className="text-[10px] text-muted-foreground">Seq Len</p>
                          </div>
                          <div className="text-center p-2 bg-background/30 rounded-lg">
                            <p className="font-mono text-sm font-bold">{model.config?.batch_size || 32}</p>
                            <p className="text-[10px] text-muted-foreground">Batch</p>
                          </div>
                        </div>
                        
                        {/* Timestamp */}
                        <div className="px-4 pb-2">
                          <p className="text-[10px] text-muted-foreground">
                            Saved: {model.timestamp?.replace(/_/g, ' @ ').replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3')}
                          </p>
                        </div>
                        
                        {/* Actions */}
                        <div className="flex border-t border-border/50">
                          <Button 
                            size="sm" 
                            variant="ghost"
                            className="flex-1 rounded-none rounded-bl-xl gap-1 h-10 hover:bg-primary/20"
                            onClick={() => handleLoadModel(model.path)}
                          >
                            <FolderOpen className="w-4 h-4" />
                            Load Model
                          </Button>
                          <div className="w-px bg-border/50" />
                          <Button 
                            size="sm" 
                            variant="ghost"
                            className="rounded-none rounded-br-xl h-10 px-4 hover:bg-destructive/20 text-destructive"
                            onClick={async () => {
                              if (!confirm("Delete this model?")) return;
                              try {
                                await API.delete(`/models/${encodeURIComponent(model.path)}`);
                                toast.success("Model deleted");
                                fetchSavedModels();
                              } catch {
                                toast.error("Failed to delete");
                              }
                            }}
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Database className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">No saved models yet</p>
                  <p className="text-xs text-muted-foreground mt-1">Train a model with "Save Model" enabled to see it here</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ==================== HISTORY TAB ==================== */}
        <TabsContent value="history" className="space-y-6">
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
                            <Badge className="text-xs" variant="outline">{session.config?.network_type || 'lstm'}</Badge>
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
                <p className="text-center text-muted-foreground py-8">No training history yet</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

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

      {/* ==================== TRAINING PROGRESS ==================== */}
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
              <span className="text-muted-foreground">Network: <Badge variant="outline">{trainingStatus?.network_type || networkType}</Badge></span>
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
              <p className="font-mono text-2xl">{trainingStatus?.data_info?.samples?.toLocaleString() || 0}</p>
            </div>
          </div>

          {/* Live Charts */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-secondary/30 rounded-lg">
              <h4 className="text-sm font-mono text-muted-foreground mb-2">Loss Curve</h4>
              <div className="h-40">
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
                    <Area type="monotone" dataKey="loss" stroke="#FF2E55" fill="url(#lossGrad)" strokeWidth={2} name="Train" />
                    <Area type="monotone" dataKey="val_loss" stroke="#00E5FF" fill="url(#valLossGrad)" strokeWidth={2} name="Val" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="p-4 bg-secondary/30 rounded-lg">
              <h4 className="text-sm font-mono text-muted-foreground mb-2">Accuracy Curve</h4>
              <div className="h-40">
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
                    <Area type="monotone" dataKey="accuracy" stroke="#00FF94" fill="url(#accGrad)" strokeWidth={2} name="Train" />
                    <Area type="monotone" dataKey="val_accuracy" stroke="#B026FF" fill="url(#valAccGrad)" strokeWidth={2} name="Val" />
                  </AreaChart>
                </ResponsiveContainer>
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
                  <div className="h-24">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportanceData} layout="vertical">
                        <XAxis type="number" tick={{ fill: '#A1A1AA', fontSize: 9 }} />
                        <YAxis dataKey="name" type="category" tick={{ fill: '#A1A1AA', fontSize: 9 }} width={70} />
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
    </div>
  );
};

export default Training;
