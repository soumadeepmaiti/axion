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
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Activity, Clock, Cpu,
  CalendarIcon, TrendingUp, Zap
} from "lucide-react";
import { format } from "date-fns";
import { startTraining, getTrainingStatus, stopTraining, getTrainingHistory, getModelSummary } from "@/lib/api";

const Training = () => {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [timeframe, setTimeframe] = useState("1h");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [learnedPatterns, setLearnedPatterns] = useState(null);
  const intervalRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      
      if (status.learned_patterns) {
        setLearnedPatterns(status.learned_patterns);
      }
    } catch (error) {
      console.error("Error fetching status:", error);
    }
  }, []);

  const fetchInitialData = useCallback(async () => {
    try {
      const history = await getTrainingHistory();
      setTrainingHistory(history.history || []);
      
      if (history.history && history.history.length > 0) {
        const latest = history.history[0];
        if (latest.learned_patterns) {
          setLearnedPatterns(latest.learned_patterns);
        }
      }
    } catch (error) {
      console.error("Error fetching initial data:", error);
    }
  }, []);

  useEffect(() => {
    fetchInitialData();
    fetchStatus();
  }, [fetchInitialData, fetchStatus]);

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    const pollInterval = trainingStatus?.is_training ? 500 : 5000;
    intervalRef.current = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(intervalRef.current);
  }, [trainingStatus?.is_training, fetchStatus]);

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      const params = {
        symbol,
        epochs,
        batch_size: batchSize,
        timeframe,
        start_date: startDate ? startDate.toISOString() : null,
        end_date: endDate ? endDate.toISOString() : null
      };
      
      const response = await startTraining(
        params.symbol, 
        params.epochs, 
        params.batch_size,
        null, // lookback_days deprecated
        params.start_date,
        params.end_date,
        params.timeframe
      );
      
      toast.success("Training started! Model will discover patterns from your data.");
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
      toast.error("Failed to stop training");
    }
  };

  const progressPercent = trainingStatus?.total_epochs > 0
    ? (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100
    : 0;

  // Prepare feature importance data for chart
  const featureImportanceData = learnedPatterns?.feature_importance
    ? Object.entries(learnedPatterns.feature_importance)
        .slice(0, 10)
        .map(([name, value]) => ({ name: name.replace(/_/g, ' '), importance: value }))
    : [];

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Model Training</h1>
          <p className="text-muted-foreground mt-1">Pure ML - Model discovers patterns from your data</p>
        </div>
        <div className="flex items-center gap-2">
          {trainingStatus?.is_training && (
            <Badge className="bg-success animate-pulse">TRAINING</Badge>
          )}
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
              <Label>Trading Pair</Label>
              <Select value={symbol} onValueChange={setSymbol}>
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
              <Label>Timeframe</Label>
              <Select value={timeframe} onValueChange={setTimeframe}>
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

            {/* Date Range Selection */}
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label>Start Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className="w-full justify-start text-left bg-secondary border-border">
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {startDate ? format(startDate, "MMM d, yyyy") : "Select"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar
                      mode="single"
                      selected={startDate}
                      onSelect={setStartDate}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
              </div>
              <div className="space-y-2">
                <Label>End Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className="w-full justify-start text-left bg-secondary border-border">
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {endDate ? format(endDate, "MMM d, yyyy") : "Today"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar
                      mode="single"
                      selected={endDate}
                      onSelect={setEndDate}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-2">
                <Label>Epochs</Label>
                <Input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="bg-secondary border-border"
                />
              </div>
              <div className="space-y-2">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="bg-secondary border-border"
                />
              </div>
            </div>

            <div className="pt-4">
              {trainingStatus?.is_training ? (
                <Button
                  variant="destructive"
                  onClick={handleStopTraining}
                  className="w-full"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </Button>
              ) : (
                <Button
                  onClick={handleStartTraining}
                  disabled={loading}
                  className="w-full bg-primary text-primary-foreground hover:bg-primary/90 glow-primary"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 mr-2" />
                  )}
                  Start Training
                </Button>
              )}
            </div>

            {trainingStatus?.data_info && (
              <div className="p-3 bg-secondary/50 rounded-lg text-sm">
                <p className="text-muted-foreground">Training Data:</p>
                <p className="font-mono text-xs mt-1">
                  {trainingStatus.data_info.training_samples?.toLocaleString()} samples
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {trainingStatus.data_info.date_range?.start?.split(' ')[0]} → {trainingStatus.data_info.date_range?.end?.split(' ')[0]}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Training Progress */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Training Progress
              {trainingStatus?.is_training && (
                <span className="ml-2 text-xs text-success animate-pulse">● Live</span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  Epoch {trainingStatus?.current_epoch || 0} / {trainingStatus?.total_epochs || 0}
                </span>
                <span className="font-mono text-primary">{progressPercent.toFixed(1)}%</span>
              </div>
              <Progress value={progressPercent} className="h-3" />
            </div>

            <div className="grid grid-cols-4 gap-4">
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase">Status</p>
                <Badge className={trainingStatus?.is_training ? 'bg-success mt-2' : 'bg-secondary mt-2'}>
                  {trainingStatus?.is_training ? 'Training' : 'Idle'}
                </Badge>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase">Loss</p>
                <p className="text-xl font-bold font-mono mt-1">
                  {trainingStatus?.current_loss?.toFixed(4) || '0.0000'}
                </p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase">Accuracy</p>
                <p className="text-xl font-bold font-mono text-success mt-1">
                  {((trainingStatus?.current_accuracy || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase">AUC</p>
                <p className="text-xl font-bold font-mono text-primary mt-1">
                  {trainingStatus?.history?.length > 0 
                    ? (trainingStatus.history[trainingStatus.history.length - 1]?.auc * 100 || 0).toFixed(1) 
                    : '0.0'}%
                </p>
              </div>
            </div>

            {/* Loss Curve */}
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingStatus?.history || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                  <XAxis dataKey="epoch" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} />
                  <Line type="monotone" dataKey="loss" stroke="#FF2E55" strokeWidth={2} dot={false} name="Train Loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="#00E5FF" strokeWidth={2} dot={false} name="Val Loss" />
                  <Line type="monotone" dataKey="accuracy" stroke="#00FF94" strokeWidth={2} dot={false} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Learned Patterns - What the Model Discovered */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="font-mono text-lg flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            Patterns Discovered by Model
            <span className="text-xs text-muted-foreground font-normal ml-2">(No imposed conditions - pure ML learning)</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {learnedPatterns ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Model Equation */}
              <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                <h3 className="font-mono font-semibold text-primary mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Model's Learned Equation
                </h3>
                <div className="bg-background/50 p-4 rounded font-mono text-sm">
                  <p className="text-foreground">{learnedPatterns.model_equation || "P(up) = σ(LSTM(X) · W + b)"}</p>
                </div>
                <p className="text-xs text-muted-foreground mt-3">
                  This equation was discovered by the model from your data, not imposed.
                </p>
              </div>

              {/* Feature Importance Chart */}
              <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                <h3 className="font-mono font-semibold text-primary mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Feature Importance (Model Learned)
                </h3>
                {featureImportanceData.length > 0 ? (
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportanceData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                        <XAxis type="number" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
                        <YAxis dataKey="name" type="category" tick={{ fill: '#A1A1AA', fontSize: 9 }} width={100} />
                        <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F' }} />
                        <Bar dataKey="importance" fill="#00E5FF" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">Train model to see learned feature importance</p>
                )}
              </div>

              {/* Learned Weights */}
              {learnedPatterns.learned_weights && Object.keys(learnedPatterns.learned_weights).length > 0 && (
                <div className="lg:col-span-2 p-4 bg-secondary/50 rounded-lg border border-border">
                  <h3 className="font-mono font-semibold text-primary mb-3">Layer Weights Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(learnedPatterns.learned_weights).map(([layer, stats]) => (
                      <div key={layer} className="p-3 bg-background/50 rounded">
                        <p className="text-xs text-muted-foreground">{layer}</p>
                        <p className="font-mono text-sm">μ: {stats.mean?.toFixed(4)}</p>
                        <p className="font-mono text-sm">σ: {stats.std?.toFixed(4)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">Train the model to discover patterns from your data</p>
              <p className="text-xs text-muted-foreground mt-2">No pre-defined formulas - model learns everything</p>
            </div>
          )}
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
                        {session.data_info.training_samples || 0} samples • {session.data_info.timeframe || '1h'}
                      </p>
                    )}
                  </div>
                  <div className="text-right">
                    <Badge className={session.result?.status === 'completed' ? 'bg-success' : 'bg-warning'}>
                      {session.result?.status || 'unknown'}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      Acc: {((session.result?.final_accuracy || session.result?.best_accuracy || 0) * 100).toFixed(1)}%
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
