import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar
} from "recharts";
import {
  TrendingUp, TrendingDown, RefreshCw, Brain, Activity,
  Zap, Target, Shield, Clock, DollarSign, BarChart3
} from "lucide-react";
import { getMarketData, makePrediction, getDashboardStats, getAggregateSentiment } from "@/lib/api";

const Dashboard = () => {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [timeframe, setTimeframe] = useState("5m");
  const [marketData, setMarketData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [marketRes, statsRes, sentimentRes] = await Promise.all([
        getMarketData(symbol, timeframe, 100),
        getDashboardStats(),
        getAggregateSentiment(symbol)
      ]);
      
      setMarketData(marketRes.data || []);
      setStats(statsRes);
      setSentiment(sentimentRes);
    } catch (error) {
      console.error("Error fetching data:", error);
      toast.error("Failed to fetch market data");
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const result = await makePrediction(symbol, true);
      setPrediction(result);
      toast.success(`Prediction: ${result.direction_label} with ${(result.confidence * 100).toFixed(1)}% confidence`);
    } catch (error) {
      console.error("Prediction error:", error);
      toast.error("Failed to make prediction");
    } finally {
      setPredicting(false);
    }
  };

  const currentPrice = marketData.length > 0 ? marketData[marketData.length - 1]?.close : 0;
  const priceChange = marketData.length > 1 
    ? ((marketData[marketData.length - 1]?.close - marketData[marketData.length - 2]?.close) / marketData[marketData.length - 2]?.close * 100).toFixed(2)
    : 0;

  return (
    <div data-testid="dashboard-page" className="p-6 space-y-6 noise-overlay">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Trading Dashboard</h1>
          <p className="text-muted-foreground mt-1">Multi-Input Hybrid Deep Learning System</p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger data-testid="symbol-select" className="w-40 bg-card border-border">
              <SelectValue placeholder="Select Symbol" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
              <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
            </SelectContent>
          </Select>
          <Button
            data-testid="refresh-btn"
            variant="outline"
            size="icon"
            onClick={fetchData}
            disabled={loading}
            className="border-border hover:border-primary/50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-card border-border hover:border-primary/30 transition-all">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Current Price</p>
                <p className="text-2xl font-bold font-mono text-foreground mt-1">
                  ${currentPrice?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
                <div className={`flex items-center gap-1 mt-1 ${Number(priceChange) >= 0 ? 'text-success' : 'text-destructive'}`}>
                  {Number(priceChange) >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  <span className="text-sm font-mono">{priceChange}%</span>
                </div>
              </div>
              <DollarSign className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border hover:border-primary/30 transition-all">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Sentiment</p>
                <p className={`text-2xl font-bold font-mono mt-1 ${
                  sentiment?.aggregate_sentiment > 0 ? 'text-success' : 
                  sentiment?.aggregate_sentiment < 0 ? 'text-destructive' : 'text-foreground'
                }`}>
                  {sentiment?.aggregate_sentiment?.toFixed(3) || '0.000'}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {sentiment?.sample_count || 0} sources analyzed
                </p>
              </div>
              <Brain className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border hover:border-primary/30 transition-all">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Model Status</p>
                <p className="text-lg font-semibold text-foreground mt-1">
                  {stats?.is_trained ? 'Trained' : 'Ready'}
                </p>
                <Badge variant={stats?.is_trained ? 'default' : 'secondary'} className="mt-1">
                  {stats?.model_status || 'active'}
                </Badge>
              </div>
              <Activity className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border hover:border-primary/30 transition-all">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Predictions</p>
                <p className="text-2xl font-bold font-mono text-foreground mt-1">
                  {stats?.total_predictions || 0}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {stats?.training_sessions || 0} training sessions
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Price Chart */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="font-mono text-lg">{symbol} Price Chart</CardTitle>
              <Tabs value={timeframe} onValueChange={setTimeframe}>
                <TabsList className="bg-secondary">
                  <TabsTrigger value="1m" data-testid="tf-1m">1m</TabsTrigger>
                  <TabsTrigger value="5m" data-testid="tf-5m">5m</TabsTrigger>
                  <TabsTrigger value="1h" data-testid="tf-1h">1h</TabsTrigger>
                  <TabsTrigger value="1d" data-testid="tf-1d">1d</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={marketData.slice(-50)}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#00E5FF" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                  <XAxis 
                    dataKey="timestamp" 
                    tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tick={{ fill: '#A1A1AA', fontSize: 10 }}
                    tickFormatter={(val) => `$${val.toLocaleString()}`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F', borderRadius: '8px' }}
                    labelStyle={{ color: '#A1A1AA' }}
                    formatter={(value) => [`$${value.toLocaleString()}`, 'Price']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#00E5FF" 
                    strokeWidth={2}
                    fill="url(#priceGradient)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Prediction Panel */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              AI Prediction
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button
              data-testid="predict-btn"
              onClick={handlePredict}
              disabled={predicting}
              className="w-full bg-primary text-primary-foreground hover:bg-primary/90 glow-primary"
            >
              {predicting ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Brain className="w-4 h-4 mr-2" />
              )}
              Generate Prediction
            </Button>

            {prediction && (
              <div className="space-y-4 animate-fade-in">
                {/* Direction */}
                <div className={`p-4 rounded-lg border ${
                  prediction.direction === 1 
                    ? 'bg-success/10 border-success/30' 
                    : 'bg-destructive/10 border-destructive/30'
                }`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Direction</span>
                    <Badge className={prediction.direction === 1 ? 'bg-success' : 'bg-destructive'}>
                      {prediction.direction_label}
                    </Badge>
                  </div>
                  <p className={`text-3xl font-bold font-mono mt-2 ${
                    prediction.direction === 1 ? 'text-success' : 'text-destructive'
                  }`}>
                    {(prediction.probability * 100).toFixed(1)}%
                  </p>
                </div>

                {/* Confidence */}
                <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm">Confidence</span>
                  </div>
                  <span className="font-mono font-semibold text-primary">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Take Profit */}
                <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-success" />
                    <span className="text-sm">Take Profit</span>
                  </div>
                  <span className="font-mono font-semibold text-success">
                    ${prediction.take_profit?.toLocaleString()}
                  </span>
                </div>

                {/* Stop Loss */}
                <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-destructive" />
                    <span className="text-sm">Stop Loss</span>
                  </div>
                  <span className="font-mono font-semibold text-destructive">
                    ${prediction.stop_loss?.toLocaleString()}
                  </span>
                </div>

                {/* Risk/Reward */}
                <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm">Risk/Reward</span>
                  </div>
                  <span className="font-mono font-semibold text-warning">
                    1:{prediction.risk_reward?.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Volume Chart */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="font-mono text-lg">Volume Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={marketData.slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                <XAxis 
                  dataKey="timestamp" 
                  tick={{ fill: '#A1A1AA', fontSize: 10 }}
                  tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                />
                <YAxis 
                  tick={{ fill: '#A1A1AA', fontSize: 10 }}
                  tickFormatter={(val) => `${(val / 1000000).toFixed(1)}M`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #1F1F1F', borderRadius: '8px' }}
                  formatter={(value) => [`${value.toLocaleString()}`, 'Volume']}
                />
                <Bar dataKey="volume" fill="#B026FF" opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
