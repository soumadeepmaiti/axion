import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import {
  Play, Square, RefreshCw, Brain, Layers, Database,
  Activity, Clock, Target, Cpu
} from "lucide-react";
import { startTraining, getTrainingStatus, stopTraining, getTrainingHistory, getModelSummary } from "@/lib/api";

const Training = () => {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [lookbackDays, setLookbackDays] = useState(30);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [modelSummary, setModelSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const [status, history, summary] = await Promise.all([
        getTrainingStatus(),
        getTrainingHistory(),
        getModelSummary()
      ]);
      setTrainingStatus(status);
      setTrainingHistory(history.history || []);
      setModelSummary(summary);
    } catch (error) {
      console.error("Error fetching training data:", error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      await startTraining(symbol, epochs, batchSize, lookbackDays);
      toast.success("Training started successfully");
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
    : 0;

  return (
    <div data-testid="training-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Model Training</h1>
          <p className="text-muted-foreground mt-1">Train the Hybrid Deep Learning Model</p>
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={fetchStatus}
          className="border-border hover:border-primary/50"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
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

            <div className="space-y-2">
              <Label htmlFor="lookback">Lookback Days</Label>
              <Input
                data-testid="lookback-input"
                id="lookback"
                type="number"
                value={lookbackDays}
                onChange={(e) => setLookbackDays(parseInt(e.target.value))}
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
                  Start Training
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Training Progress */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Training Progress
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
              <Progress value={progressPercent} className="h-2" />
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-secondary rounded-lg">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Status</p>
                <Badge 
                  data-testid="training-status-badge"
                  className={trainingStatus?.is_training ? 'bg-success mt-2' : 'bg-secondary mt-2'}
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

            {/* Loss Curve */}
            {trainingStatus?.history && trainingStatus.history.length > 0 && (
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingStatus.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                    <XAxis 
                      dataKey="epoch" 
                      tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    />
                    <YAxis 
                      tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F', borderRadius: '8px' }}
                    />
                    <Line type="monotone" dataKey="loss" stroke="#FF2E55" strokeWidth={2} dot={false} name="Train Loss" />
                    <Line type="monotone" dataKey="val_loss" stroke="#00E5FF" strokeWidth={2} dot={false} name="Val Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

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
              <p className="text-sm text-muted-foreground mb-3">Bi-LSTM for 1m/5m/10m price data</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input Shape</span>
                  <span className="font-mono">{JSON.stringify(modelSummary?.micro_input_shape || [50, 15])}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">LSTM Units</span>
                  <span className="font-mono">{modelSummary?.lstm_units || 64}</span>
                </div>
              </div>
            </div>

            {/* Branch B */}
            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Database className="w-5 h-5 text-warning" />
                <h3 className="font-mono font-semibold">Branch B: Macro</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">Attention-GRU for 1h/1d trends</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input Shape</span>
                  <span className="font-mono">{JSON.stringify(modelSummary?.macro_input_shape || [24, 10])}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">GRU Units</span>
                  <span className="font-mono">{modelSummary?.gru_units || 32}</span>
                </div>
              </div>
            </div>

            {/* Branch C */}
            <div className="p-4 bg-secondary/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-5 h-5 text-chart-4" />
                <h3 className="font-mono font-semibold">Branch C: Sentiment</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">FinBERT + LLM Enhancement</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Input Shape</span>
                  <span className="font-mono">{JSON.stringify(modelSummary?.sentiment_input_shape || [4])}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dense Units</span>
                  <span className="font-mono">{modelSummary?.dense_units || 64}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Stats */}
          <div className="mt-6 p-4 bg-secondary rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Status</p>
                <Badge className="mt-2">{modelSummary?.status || 'unknown'}</Badge>
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
                <Badge className={modelSummary?.is_trained ? 'bg-success mt-2' : 'bg-secondary mt-2'}>
                  {modelSummary?.is_trained ? 'Yes' : 'No'}
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
