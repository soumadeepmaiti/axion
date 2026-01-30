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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Activity, Clock, Cpu,
  CalendarIcon, TrendingUp, Zap, Layers, Settings, Calculator,
  GitBranch, Sigma
} from "lucide-react";
import { format } from "date-fns";
import { API } from "@/lib/api";

// Custom API call for advanced training
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
  
  // Network architecture
  const [numLstmLayers, setNumLstmLayers] = useState(2);
  const [lstmUnits, setLstmUnits] = useState([128, 64]);
  const [numDenseLayers, setNumDenseLayers] = useState(2);
  const [denseUnits, setDenseUnits] = useState([64, 32]);
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
  const [mathSignals, setMathSignals] = useState(null);
  const intervalRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      if (status.learned_patterns) setLearnedPatterns(status.learned_patterns);
      if (status.math_signals) setMathSignals(status.math_signals);
    } catch (error) {
      console.error("Error:", error);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const history = await getTrainingHistory();
      setTrainingHistory(history.history || []);
    } catch (error) {
      console.error("Error:", error);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchStatus();
  }, [fetchHistory, fetchStatus]);

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    const pollInterval = trainingStatus?.is_training ? 500 : 5000;
    intervalRef.current = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(intervalRef.current);
  }, [trainingStatus?.is_training, fetchStatus]);

  const handleStartTraining = async () => {
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
      
      await startAdvancedTraining(config);
      toast.success(`Training started in ${mode.replace('_', ' ')} mode!`);
      fetchStatus();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to start training");
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    await stopTraining();
    toast.info("Training stopped");
    fetchStatus();
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

  const featureImportanceData = learnedPatterns?.feature_importance
    ? Object.entries(learnedPatterns.feature_importance).slice(0, 10)
        .map(([name, value]) => ({ name: name.replace(/_/g, ' '), importance: value }))
    : [];

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Model Training</h1>
          <p className="text-muted-foreground mt-1">Pure ML + Mathematical Modeling - Full Control</p>
        </div>
        {trainingStatus?.is_training && (
          <Badge className="bg-success animate-pulse">TRAINING</Badge>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Configuration Panel */}
        <Card className="lg:col-span-1 bg-card border-border">
          <CardHeader className="pb-2">
            <CardTitle className="font-mono text-base flex items-center gap-2">
              <Settings className="w-4 h-4 text-primary" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Mode Selection */}
            <div className="space-y-2">
              <Label className="text-xs">Training Mode</Label>
              <Select value={mode} onValueChange={setMode}>
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

            <Separator />

            {/* Data Selection */}
            <div className="space-y-2">
              <Label className="text-xs">Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger className="bg-secondary border-border h-8">
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
              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger className="bg-secondary border-border h-8">
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

            {/* Date Range */}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-xs">Start</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs">
                      <CalendarIcon className="mr-1 h-3 w-3" />
                      {startDate ? format(startDate, "MM/dd/yy") : "Select"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={startDate} onSelect={setStartDate} /></PopoverContent>
                </Popover>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">End</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full bg-secondary border-border text-xs">
                      <CalendarIcon className="mr-1 h-3 w-3" />
                      {endDate ? format(endDate, "MM/dd/yy") : "Today"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0"><Calendar mode="single" selected={endDate} onSelect={setEndDate} /></PopoverContent>
                </Popover>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-xs">Epochs</Label>
                <Input type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} className="bg-secondary border-border h-8 text-sm" />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Batch Size</Label>
                <Input type="number" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))} className="bg-secondary border-border h-8 text-sm" />
              </div>
            </div>

            <div className="pt-2">
              {trainingStatus?.is_training ? (
                <Button variant="destructive" onClick={handleStopTraining} className="w-full" size="sm">
                  <Square className="w-3 h-3 mr-1" /> Stop
                </Button>
              ) : (
                <Button onClick={handleStartTraining} disabled={loading} className="w-full bg-primary hover:bg-primary/90 glow-primary" size="sm">
                  {loading ? <RefreshCw className="w-3 h-3 mr-1 animate-spin" /> : <Play className="w-3 h-3 mr-1" />}
                  Start Training
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Main Panel */}
        <div className="lg:col-span-3 space-y-6">
          <Tabs defaultValue="network" className="w-full">
            <TabsList className="bg-secondary">
              <TabsTrigger value="network" className="text-xs"><Layers className="w-3 h-3 mr-1" />Network</TabsTrigger>
              <TabsTrigger value="strategies" className="text-xs"><Calculator className="w-3 h-3 mr-1" />Math Strategies</TabsTrigger>
              <TabsTrigger value="progress" className="text-xs"><Activity className="w-3 h-3 mr-1" />Progress</TabsTrigger>
              <TabsTrigger value="learned" className="text-xs"><Brain className="w-3 h-3 mr-1" />Learned</TabsTrigger>
            </TabsList>

            {/* Network Architecture Tab */}
            <TabsContent value="network">
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base">Network Architecture</CardTitle>
                  <p className="text-xs text-muted-foreground">Configure the depth and size of your neural network</p>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-2 gap-6">
                    {/* LSTM Configuration */}
                    <div className="space-y-4">
                      <h4 className="font-mono text-sm text-primary">LSTM Layers</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span>Number of Layers</span>
                          <span className="font-mono text-primary">{numLstmLayers}</span>
                        </div>
                        <Slider value={[numLstmLayers]} onValueChange={([v]) => setNumLstmLayers(v)} min={1} max={4} step={1} />
                      </div>
                      {[...Array(numLstmLayers)].map((_, i) => (
                        <div key={i} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span>Layer {i + 1} Units</span>
                            <span className="font-mono">{lstmUnits[i] || 64}</span>
                          </div>
                          <Slider
                            value={[lstmUnits[i] || 64]}
                            onValueChange={([v]) => {
                              const newUnits = [...lstmUnits];
                              newUnits[i] = v;
                              setLstmUnits(newUnits);
                            }}
                            min={16} max={256} step={16}
                          />
                        </div>
                      ))}
                    </div>

                    {/* Dense Configuration */}
                    <div className="space-y-4">
                      <h4 className="font-mono text-sm text-primary">Dense Layers</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span>Number of Layers</span>
                          <span className="font-mono text-primary">{numDenseLayers}</span>
                        </div>
                        <Slider value={[numDenseLayers]} onValueChange={([v]) => setNumDenseLayers(v)} min={1} max={4} step={1} />
                      </div>
                      {[...Array(numDenseLayers)].map((_, i) => (
                        <div key={i} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span>Layer {i + 1} Units</span>
                            <span className="font-mono">{denseUnits[i] || 32}</span>
                          </div>
                          <Slider
                            value={[denseUnits[i] || 32]}
                            onValueChange={([v]) => {
                              const newUnits = [...denseUnits];
                              newUnits[i] = v;
                              setDenseUnits(newUnits);
                            }}
                            min={8} max={128} step={8}
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  {/* Other Settings */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>Dropout Rate</span>
                        <span className="font-mono text-primary">{dropoutRate}</span>
                      </div>
                      <Slider value={[dropoutRate * 100]} onValueChange={([v]) => setDropoutRate(v / 100)} min={0} max={50} step={5} />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>Learning Rate</span>
                        <span className="font-mono text-primary">{learningRate}</span>
                      </div>
                      <Slider value={[Math.log10(learningRate) + 4]} onValueChange={([v]) => setLearningRate(Math.pow(10, v - 4))} min={1} max={3} step={0.5} />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>Sequence Length</span>
                        <span className="font-mono text-primary">{sequenceLength}</span>
                      </div>
                      <Slider value={[sequenceLength]} onValueChange={([v]) => setSequenceLength(v)} min={10} max={100} step={10} />
                    </div>
                  </div>

                  <div className="flex gap-6">
                    <div className="flex items-center gap-2">
                      <Switch checked={useAttention} onCheckedChange={setUseAttention} />
                      <Label className="text-xs">Attention Mechanism</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch checked={useBatchNorm} onCheckedChange={setUseBatchNorm} />
                      <Label className="text-xs">Batch Normalization</Label>
                    </div>
                  </div>

                  {/* Architecture Summary */}
                  <div className="p-3 bg-secondary/50 rounded-lg">
                    <p className="text-xs font-mono text-muted-foreground">
                      Architecture: Input → {numLstmLayers}x Bi-LSTM({lstmUnits.slice(0, numLstmLayers).join(',')}) 
                      {useAttention && ' → Attention'} → {numDenseLayers}x Dense({denseUnits.slice(0, numDenseLayers).join(',')}) → Sigmoid
                    </p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Mathematical Strategies Tab */}
            <TabsContent value="strategies">
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base flex items-center gap-2">
                    <Sigma className="w-4 h-4" />
                    Mathematical Strategies
                  </CardTitle>
                  <p className="text-xs text-muted-foreground">Renaissance-style quantitative strategies (used in Mathematical/Hybrid modes)</p>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {MATH_STRATEGIES.map((strategy) => (
                      <div
                        key={strategy.id}
                        className={`p-3 rounded-lg border cursor-pointer transition-all ${
                          selectedStrategies.includes(strategy.id)
                            ? 'border-primary bg-primary/10'
                            : 'border-border bg-secondary/30 hover:border-primary/50'
                        }`}
                        onClick={() => toggleStrategy(strategy.id)}
                      >
                        <div className="flex items-center gap-2">
                          <Checkbox checked={selectedStrategies.includes(strategy.id)} />
                          <span className="font-mono text-sm">{strategy.name}</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">{strategy.description}</p>
                      </div>
                    ))}
                  </div>

                  {/* Math Signals Display */}
                  {mathSignals && (
                    <div className="mt-6 space-y-3">
                      <h4 className="font-mono text-sm text-primary">Calculated Signals</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {Object.entries(mathSignals).filter(([k]) => k !== 'aggregate').map(([name, data]) => (
                          <div key={name} className="p-3 bg-secondary/50 rounded-lg">
                            <p className="text-xs text-muted-foreground capitalize">{name.replace(/_/g, ' ')}</p>
                            {data.signal !== undefined && (
                              <Badge className={data.signal > 0 ? 'bg-success mt-1' : data.signal < 0 ? 'bg-destructive mt-1' : 'bg-secondary mt-1'}>
                                {data.signal > 0 ? 'BUY' : data.signal < 0 ? 'SELL' : 'HOLD'}
                              </Badge>
                            )}
                            {data.formula && <p className="text-xs font-mono mt-2 text-muted-foreground">{data.formula}</p>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Progress Tab */}
            <TabsContent value="progress">
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base">Training Progress</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Epoch {trainingStatus?.current_epoch || 0} / {trainingStatus?.total_epochs || 0}</span>
                      <span className="font-mono text-primary">{progressPercent.toFixed(1)}%</span>
                    </div>
                    <Progress value={progressPercent} className="h-2" />
                  </div>

                  <div className="grid grid-cols-4 gap-3">
                    <div className="p-3 bg-secondary rounded-lg text-center">
                      <p className="text-xs text-muted-foreground">Mode</p>
                      <Badge className="mt-1">{trainingStatus?.mode || mode}</Badge>
                    </div>
                    <div className="p-3 bg-secondary rounded-lg text-center">
                      <p className="text-xs text-muted-foreground">Loss</p>
                      <p className="font-mono text-lg">{trainingStatus?.current_loss?.toFixed(4) || '0.0000'}</p>
                    </div>
                    <div className="p-3 bg-secondary rounded-lg text-center">
                      <p className="text-xs text-muted-foreground">Accuracy</p>
                      <p className="font-mono text-lg text-success">{((trainingStatus?.current_accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-3 bg-secondary rounded-lg text-center">
                      <p className="text-xs text-muted-foreground">Samples</p>
                      <p className="font-mono text-lg">{trainingStatus?.data_info?.training_samples?.toLocaleString() || 0}</p>
                    </div>
                  </div>

                  {/* Loss Curve */}
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingStatus?.history || []}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                        <XAxis dataKey="epoch" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                        <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} />
                        <Line type="monotone" dataKey="loss" stroke="#FF2E55" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="val_loss" stroke="#00E5FF" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="accuracy" stroke="#00FF94" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Learned Patterns Tab */}
            <TabsContent value="learned">
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base">Patterns Discovered by Model</CardTitle>
                </CardHeader>
                <CardContent>
                  {learnedPatterns ? (
                    <div className="space-y-6">
                      {/* Equation */}
                      <div className="p-4 bg-secondary/50 rounded-lg">
                        <h4 className="font-mono text-sm text-primary mb-2 flex items-center gap-2">
                          <Zap className="w-4 h-4" /> Model's Learned Equation
                        </h4>
                        <p className="font-mono text-sm">{learnedPatterns.model_equation}</p>
                      </div>

                      {/* Feature Importance */}
                      {featureImportanceData.length > 0 && (
                        <div className="h-48">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={featureImportanceData} layout="vertical">
                              <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                              <XAxis type="number" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                              <YAxis dataKey="name" type="category" tick={{ fill: '#A1A1AA', fontSize: 9 }} width={120} />
                              <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} />
                              <Bar dataKey="importance" fill="#00E5FF" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      )}

                      {/* Weights */}
                      {learnedPatterns.learned_weights && (
                        <div className="grid grid-cols-4 gap-3">
                          {Object.entries(learnedPatterns.learned_weights).map(([layer, stats]) => (
                            <div key={layer} className="p-2 bg-secondary/50 rounded">
                              <p className="text-xs text-muted-foreground">{layer}</p>
                              <p className="font-mono text-xs">μ={stats.mean?.toFixed(3)} σ={stats.std?.toFixed(3)}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">Train the model to see discovered patterns</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Training History */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-2">
          <CardTitle className="font-mono text-base flex items-center gap-2">
            <Clock className="w-4 h-4 text-primary" /> Training History
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trainingHistory.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {trainingHistory.map((session, i) => (
                <div key={i} className="p-3 bg-secondary rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-mono text-sm">{session.symbol}</p>
                      <p className="text-xs text-muted-foreground">{new Date(session.created_at).toLocaleString()}</p>
                      <Badge className="mt-1 text-xs" variant="outline">{session.config?.mode || 'pure_ml'}</Badge>
                    </div>
                    <div className="text-right">
                      <Badge className={session.result?.status === 'completed' ? 'bg-success' : 'bg-warning'}>
                        {((session.result?.best_accuracy || session.result?.final_accuracy || 0) * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-4">No training history yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Training;
