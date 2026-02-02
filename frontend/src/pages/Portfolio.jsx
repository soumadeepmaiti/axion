import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter, Legend
} from "recharts";
import {
  Wallet, TrendingUp, PieChart as PieChartIcon, Settings, Brain,
  DollarSign, Target, Percent, RefreshCw, Play, ChevronRight,
  BarChart3, Layers, Zap, Shield, GitCompare
} from "lucide-react";
import { API } from "@/lib/api";

// Default assets
const DEFAULT_ASSETS = [
  "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
  "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
  "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
  "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "NEAR/USDT"
];

// Colors for pie chart
const COLORS = [
  '#00d4aa', '#00a8cc', '#0088aa', '#006688', '#004466',
  '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4',
  '#84cc16', '#f97316', '#6366f1', '#14b8a6', '#a855f7',
  '#22c55e', '#eab308', '#3b82f6', '#e11d48', '#64748b'
];

const Portfolio = () => {
  // Asset selection
  const [selectedAssets, setSelectedAssets] = useState(DEFAULT_ASSETS.slice(0, 10));
  const [assetStats, setAssetStats] = useState({});
  
  // Investment settings
  const [investmentAmount, setInvestmentAmount] = useState(1000);
  const [horizon, setHorizon] = useState("7d");
  const [objective, setObjective] = useState("max_sharpe");
  const [strategy, setStrategy] = useState("traditional_ml");
  const [compareAll, setCompareAll] = useState(true);
  
  // Constraints
  const [maxWeight, setMaxWeight] = useState(30);
  const [minAssets, setMinAssets] = useState(5);
  
  // Results
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [efficientFrontier, setEfficientFrontier] = useState(null);
  
  // Status
  const [loading, setLoading] = useState(false);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [dlTraining, setDlTraining] = useState(false);
  const [rlTraining, setRlTraining] = useState(false);

  // Fetch initial data
  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await API.get('/portfolio/model-info');
      setModelInfo(response.data);
    } catch (error) {
      console.error("Error fetching model info:", error);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await API.get('/portfolio/training-status');
      setTrainingStatus(response.data);
      return response.data;
    } catch (error) {
      console.error("Error fetching training status:", error);
      return null;
    }
  };

  // Train Deep Learning Model
  const handleTrainDeepLearning = async () => {
    if (!dataLoaded) {
      toast.error("Fetch data first");
      return;
    }
    
    setDlTraining(true);
    try {
      const response = await API.post('/portfolio/train-model', {
        model_type: 'deep_learning',
        epochs: 100
      });
      
      if (response.data.status === 'started') {
        toast.success("Deep Learning training started");
        
        // Poll for completion
        const pollInterval = setInterval(async () => {
          await fetchModelInfo();
          if (modelInfo?.deep_learning_trained) {
            clearInterval(pollInterval);
            setDlTraining(false);
            toast.success("Deep Learning model trained!");
          }
        }, 3000);
        
        // Timeout after 3 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          setDlTraining(false);
          fetchModelInfo();
        }, 180000);
      }
    } catch (error) {
      toast.error("Training failed");
      setDlTraining(false);
    }
  };

  // Train RL Agent
  const handleTrainRLAgent = async () => {
    if (!dataLoaded) {
      toast.error("Fetch data first");
      return;
    }
    
    setRlTraining(true);
    try {
      const response = await API.post('/portfolio/train-model', {
        model_type: 'rl_agent',
        n_episodes: 50
      });
      
      if (response.data.status === 'started') {
        toast.success("RL Agent training started");
        
        // Poll for completion
        const pollInterval = setInterval(async () => {
          await fetchModelInfo();
          if (modelInfo?.rl_agent_trained) {
            clearInterval(pollInterval);
            setRlTraining(false);
            toast.success("RL Agent trained!");
          }
        }, 3000);
        
        // Timeout after 3 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          setRlTraining(false);
          fetchModelInfo();
        }, 180000);
      }
    } catch (error) {
      toast.error("Training failed");
      setRlTraining(false);
    }
  };

  // Fetch data for selected assets
  const handleFetchData = async () => {
    if (selectedAssets.length < 2) {
      toast.error("Select at least 2 assets");
      return;
    }
    
    setLoading(true);
    try {
      const response = await API.post('/portfolio/fetch-data', {
        assets: selectedAssets,
        timeframe: '1d'
      });
      
      if (response.data.status === 'success') {
        setAssetStats(response.data.statistics || {});
        setCorrelationData(response.data.correlation_data);
        setDataLoaded(true);
        toast.success(`Loaded data for ${response.data.assets_fetched} assets`);
      } else {
        toast.error(response.data.message || "Failed to fetch data");
      }
    } catch (error) {
      toast.error("Failed to fetch data");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // Train models
  const handleTrainModels = async () => {
    setLoading(true);
    try {
      const response = await API.post('/portfolio/train', {
        assets: selectedAssets,
        timeframe: '1d',
        epochs: 50
      });
      
      if (response.data.status === 'started') {
        toast.success("Training started for all assets");
        
        // Poll for training status
        const pollInterval = setInterval(async () => {
          const status = await fetchTrainingStatus();
          if (status && !status.is_training) {
            clearInterval(pollInterval);
            setLoading(false);
            fetchModelInfo();
            toast.success("Training completed!");
          }
        }, 2000);
      } else {
        toast.error(response.data.message || "Failed to start training");
        setLoading(false);
      }
    } catch (error) {
      toast.error("Failed to start training");
      setLoading(false);
    }
  };

  // Optimize portfolio
  const handleOptimize = async () => {
    if (!dataLoaded) {
      toast.error("Please fetch data first");
      return;
    }
    
    setLoading(true);
    try {
      const response = await API.post('/portfolio/optimize', {
        assets: selectedAssets,
        investment_amount: investmentAmount,
        strategy: strategy,
        objective: objective,
        horizon: horizon,
        compare_all: compareAll,
        constraints: {
          max_weight: maxWeight,
          min_assets: minAssets
        }
      });
      
      setOptimizationResult(response.data);
      
      if (response.data.status === 'success') {
        toast.success("Portfolio optimized!");
        
        // Fetch efficient frontier
        const frontierResponse = await API.get(`/portfolio/efficient-frontier?assets=${selectedAssets.join(',')}`);
        if (frontierResponse.data.status === 'success') {
          setEfficientFrontier(frontierResponse.data.frontier);
        }
      } else {
        toast.error(response.data.message || "Optimization failed");
      }
    } catch (error) {
      toast.error("Optimization failed");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // Toggle asset selection
  const toggleAsset = (asset) => {
    setSelectedAssets(prev => 
      prev.includes(asset)
        ? prev.filter(a => a !== asset)
        : [...prev, asset]
    );
  };

  // Select/deselect all
  const selectAll = () => setSelectedAssets(DEFAULT_ASSETS);
  const deselectAll = () => setSelectedAssets([]);

  // Get allocation data for pie chart
  const getAllocationData = () => {
    if (!optimizationResult?.allocations && !optimizationResult?.strategies?.traditional_ml?.allocations) {
      return [];
    }
    
    const allocations = optimizationResult.allocations || 
                       optimizationResult.strategies?.traditional_ml?.allocations || [];
    
    return allocations.map((a, i) => ({
      name: a.symbol.replace('/USDT', ''),
      value: a.weight,
      amount: a.amount,
      color: COLORS[i % COLORS.length]
    }));
  };

  // Get comparison table data
  const getComparisonData = () => {
    if (!optimizationResult?.strategies) return [];
    
    const strategies = optimizationResult.strategies;
    const assets = new Set();
    
    // Collect all assets
    Object.values(strategies).forEach(s => {
      if (s.allocations) {
        s.allocations.forEach(a => assets.add(a.symbol));
      }
    });
    
    return Array.from(assets).map(asset => {
      const row = { asset: asset.replace('/USDT', '') };
      Object.entries(strategies).forEach(([key, strategy]) => {
        if (strategy.allocations) {
          const alloc = strategy.allocations.find(a => a.symbol === asset);
          row[key] = alloc ? alloc.weight : 0;
        } else {
          row[key] = '-';
        }
      });
      return row;
    }).filter(row => Object.values(row).some(v => v > 0 && v !== '-'));
  };

  const allocationData = getAllocationData();
  const comparisonData = getComparisonData();
  const metrics = optimizationResult?.metrics || 
                 optimizationResult?.strategies?.traditional_ml?.metrics || {};

  return (
    <div data-testid="portfolio-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Portfolio Optimizer</h1>
          <p className="text-muted-foreground mt-1">
            Multi-Asset Allocation with ML Predictions
          </p>
        </div>
        <div className="flex items-center gap-2">
          {modelInfo?.is_trained && (
            <Badge className="bg-green-500/20 text-green-400">
              {modelInfo.num_models} Models Trained
            </Badge>
          )}
          {dataLoaded && (
            <Badge className="bg-primary/20 text-primary">
              Data Loaded
            </Badge>
          )}
        </div>
      </div>

      <Tabs defaultValue="config" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="config" className="gap-2">
            <Settings className="w-4 h-4" /> Configuration
          </TabsTrigger>
          <TabsTrigger value="results" className="gap-2">
            <PieChartIcon className="w-4 h-4" /> Allocation
          </TabsTrigger>
          <TabsTrigger value="compare" className="gap-2">
            <GitCompare className="w-4 h-4" /> Compare
          </TabsTrigger>
          <TabsTrigger value="analysis" className="gap-2">
            <BarChart3 className="w-4 h-4" /> Analysis
          </TabsTrigger>
        </TabsList>

        {/* ============ CONFIGURATION TAB ============ */}
        <TabsContent value="config" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Asset Selection */}
            <Card className="bg-card border-border lg:col-span-2">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="font-mono text-base flex items-center gap-2">
                    <Layers className="w-4 h-4 text-primary" />
                    Select Assets ({selectedAssets.length}/20)
                  </CardTitle>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={selectAll}>All</Button>
                    <Button size="sm" variant="outline" onClick={deselectAll}>None</Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 md:grid-cols-5 gap-2">
                  {DEFAULT_ASSETS.map((asset) => {
                    const isSelected = selectedAssets.includes(asset);
                    const symbol = asset.replace('/USDT', '');
                    const stats = assetStats[asset];
                    
                    return (
                      <div
                        key={asset}
                        className={`p-3 rounded-lg border cursor-pointer transition-all ${
                          isSelected
                            ? 'border-primary bg-primary/10'
                            : 'border-border bg-secondary/30 hover:border-primary/50'
                        }`}
                        onClick={() => toggleAsset(asset)}
                      >
                        <div className="flex items-center gap-2">
                          <Checkbox checked={isSelected} />
                          <span className="font-mono text-sm font-semibold">{symbol}</span>
                        </div>
                        {stats && (
                          <div className="mt-1 text-[10px] text-muted-foreground">
                            <span className={stats.expected_return > 0 ? 'text-green-400' : 'text-red-400'}>
                              {stats.expected_return > 0 ? '+' : ''}{stats.expected_return}%
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                
                {/* Data Actions */}
                <div className="flex gap-2 mt-4">
                  <Button 
                    onClick={handleFetchData} 
                    disabled={loading || selectedAssets.length < 2}
                    className="flex-1"
                  >
                    <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                    Fetch Market Data
                  </Button>
                  <Button 
                    onClick={handleTrainModels} 
                    disabled={loading || !dataLoaded}
                    variant="outline"
                    className="flex-1"
                  >
                    <Brain className="w-4 h-4 mr-2" />
                    Train ML Models
                  </Button>
                </div>
                
                {trainingStatus?.is_training && (
                  <div className="mt-4 p-3 bg-primary/10 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm">Training: {trainingStatus.current_asset}</span>
                      <span className="text-sm font-mono">{trainingStatus.progress}%</span>
                    </div>
                    <Progress value={trainingStatus.progress} />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Investment Settings */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <DollarSign className="w-4 h-4 text-primary" />
                  Investment Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Investment Amount ($)</Label>
                  <Input
                    type="number"
                    value={investmentAmount}
                    onChange={(e) => setInvestmentAmount(parseFloat(e.target.value) || 0)}
                    className="bg-secondary border-border font-mono"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Prediction Horizon</Label>
                  <Select value={horizon} onValueChange={setHorizon}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="24h">24 Hours</SelectItem>
                      <SelectItem value="7d">7 Days</SelectItem>
                      <SelectItem value="30d">30 Days</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Optimization Objective</Label>
                  <Select value={objective} onValueChange={setObjective}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="max_sharpe">Max Sharpe Ratio</SelectItem>
                      <SelectItem value="max_return">Max Return (Aggressive)</SelectItem>
                      <SelectItem value="min_risk">Min Risk (Conservative)</SelectItem>
                      <SelectItem value="risk_parity">Risk Parity</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Strategy</Label>
                  <Select value={strategy} onValueChange={setStrategy}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="traditional_ml">Traditional + ML</SelectItem>
                      <SelectItem value="deep_learning">Deep Learning (Phase 2)</SelectItem>
                      <SelectItem value="rl_agent">RL Agent (Phase 2)</SelectItem>
                      <SelectItem value="hybrid">Hybrid Ensemble</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                <div className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <Label>Max Weight per Asset</Label>
                      <span className="font-mono text-primary">{maxWeight}%</span>
                    </div>
                    <Slider
                      value={[maxWeight]}
                      onValueChange={([v]) => setMaxWeight(v)}
                      min={10}
                      max={100}
                      step={5}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <Label>Min Assets in Portfolio</Label>
                      <span className="font-mono text-primary">{minAssets}</span>
                    </div>
                    <Slider
                      value={[minAssets]}
                      onValueChange={([v]) => setMinAssets(v)}
                      min={1}
                      max={15}
                      step={1}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg">
                  <Label className="text-xs">Compare All Strategies</Label>
                  <Checkbox 
                    checked={compareAll} 
                    onCheckedChange={setCompareAll}
                  />
                </div>

                <Button 
                  onClick={handleOptimize} 
                  disabled={loading || !dataLoaded}
                  className="w-full bg-primary hover:bg-primary/90"
                  size="lg"
                >
                  <Target className="w-4 h-4 mr-2" />
                  Optimize Portfolio
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ============ ALLOCATION TAB ============ */}
        <TabsContent value="results" className="space-y-6">
          {optimizationResult ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Pie Chart */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base flex items-center gap-2">
                    <PieChartIcon className="w-4 h-4 text-primary" />
                    Recommended Allocation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {allocationData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={allocationData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}%`}
                        >
                          {allocationData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          formatter={(value, name, props) => [
                            `${value}% ($${props.payload.amount})`,
                            props.payload.name
                          ]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                      Run optimization to see allocation
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Allocation Table */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-base flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-primary" />
                    Investment Breakdown
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {allocationData.map((item, i) => (
                      <div 
                        key={item.name}
                        className="flex items-center justify-between p-2 bg-secondary/30 rounded"
                      >
                        <div className="flex items-center gap-3">
                          <div 
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: item.color }}
                          />
                          <span className="font-mono font-semibold">{item.name}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-muted-foreground">{item.value}%</span>
                          <span className="font-mono text-primary font-semibold">
                            ${item.amount.toLocaleString()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Metrics */}
                  <Separator className="my-4" />
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-3 bg-secondary/30 rounded-lg">
                      <p className="text-2xl font-mono font-bold text-green-400">
                        {metrics.expected_return > 0 ? '+' : ''}{metrics.expected_return || 0}%
                      </p>
                      <p className="text-xs text-muted-foreground">Expected Return</p>
                    </div>
                    <div className="text-center p-3 bg-secondary/30 rounded-lg">
                      <p className="text-2xl font-mono font-bold text-yellow-400">
                        {metrics.volatility || 0}%
                      </p>
                      <p className="text-xs text-muted-foreground">Volatility</p>
                    </div>
                    <div className="text-center p-3 bg-secondary/30 rounded-lg">
                      <p className="text-2xl font-mono font-bold text-primary">
                        {metrics.sharpe_ratio || 0}
                      </p>
                      <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card className="bg-card border-border">
              <CardContent className="py-12 text-center">
                <PieChartIcon className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Allocation Yet</h3>
                <p className="text-muted-foreground mb-4">
                  Configure your settings and click "Optimize Portfolio" to see recommendations
                </p>
                <Button onClick={() => document.querySelector('[value="config"]')?.click()}>
                  Go to Configuration
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ============ COMPARE TAB ============ */}
        <TabsContent value="compare" className="space-y-6">
          {optimizationResult?.strategies ? (
            <div className="space-y-6">
              {/* Strategy Comparison Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(optimizationResult.strategies).map(([key, strategy]) => (
                  <Card 
                    key={key}
                    className={`bg-card border-border ${
                      optimizationResult.recommended === key 
                        ? 'border-primary ring-2 ring-primary/20' 
                        : ''
                    }`}
                  >
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="font-mono text-sm">
                          {key === 'traditional_ml' && '🎯 Traditional+ML'}
                          {key === 'deep_learning' && '🧠 Deep Learning'}
                          {key === 'rl_agent' && '🤖 RL Agent'}
                          {key === 'hybrid' && '🔀 Hybrid'}
                        </CardTitle>
                        {optimizationResult.recommended === key && (
                          <Badge className="bg-primary text-[10px]">Recommended</Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      {strategy.status === 'success' ? (
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Return</span>
                            <span className={`font-mono ${
                              strategy.metrics?.expected_return > 0 
                                ? 'text-green-400' 
                                : 'text-red-400'
                            }`}>
                              {strategy.metrics?.expected_return > 0 ? '+' : ''}
                              {strategy.metrics?.expected_return}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Risk</span>
                            <span className="font-mono text-yellow-400">
                              {strategy.metrics?.volatility}%
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Sharpe</span>
                            <span className="font-mono text-primary">
                              {strategy.metrics?.sharpe_ratio}
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Assets</span>
                            <span className="font-mono">
                              {strategy.metrics?.num_assets}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-4">
                          <Badge variant="outline" className="text-yellow-400">
                            {strategy.status === 'pending' ? 'Coming in Phase 2' : 'Error'}
                          </Badge>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Comparison Table */}
              {comparisonData.length > 0 && (
                <Card className="bg-card border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="font-mono text-base flex items-center gap-2">
                      <GitCompare className="w-4 h-4 text-primary" />
                      Allocation Comparison by Asset
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left p-2 font-mono text-xs">Asset</th>
                            <th className="text-right p-2 font-mono text-xs">Traditional+ML</th>
                            <th className="text-right p-2 font-mono text-xs">Deep Learning</th>
                            <th className="text-right p-2 font-mono text-xs">RL Agent</th>
                            <th className="text-right p-2 font-mono text-xs">Hybrid</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparisonData.slice(0, 10).map((row, i) => (
                            <tr key={row.asset} className="border-b border-border/50">
                              <td className="p-2 font-mono font-semibold">{row.asset}</td>
                              <td className="text-right p-2 font-mono">
                                {typeof row.traditional_ml === 'number' ? `${row.traditional_ml}%` : '-'}
                              </td>
                              <td className="text-right p-2 font-mono text-muted-foreground">
                                {typeof row.deep_learning === 'number' ? `${row.deep_learning}%` : '-'}
                              </td>
                              <td className="text-right p-2 font-mono text-muted-foreground">
                                {typeof row.rl_agent === 'number' ? `${row.rl_agent}%` : '-'}
                              </td>
                              <td className="text-right p-2 font-mono">
                                {typeof row.hybrid === 'number' ? `${row.hybrid}%` : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          ) : (
            <Card className="bg-card border-border">
              <CardContent className="py-12 text-center">
                <GitCompare className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Comparison Data</h3>
                <p className="text-muted-foreground">
                  Enable "Compare All Strategies" and run optimization
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ============ ANALYSIS TAB ============ */}
        <TabsContent value="analysis" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Correlation Heatmap */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Asset Correlations
                </CardTitle>
              </CardHeader>
              <CardContent>
                {correlationData?.matrix ? (
                  <div className="overflow-x-auto">
                    <div className="min-w-[400px]">
                      <div className="grid" style={{ 
                        gridTemplateColumns: `60px repeat(${correlationData.assets.length}, 1fr)` 
                      }}>
                        {/* Header row */}
                        <div></div>
                        {correlationData.assets.map(asset => (
                          <div key={asset} className="text-[10px] font-mono text-center p-1 truncate">
                            {asset}
                          </div>
                        ))}
                        
                        {/* Data rows */}
                        {correlationData.matrix.map((row, i) => (
                          <>
                            <div key={`label-${i}`} className="text-[10px] font-mono p-1 flex items-center">
                              {correlationData.assets[i]}
                            </div>
                            {row.map((val, j) => {
                              const intensity = Math.abs(val);
                              const color = val > 0 
                                ? `rgba(0, 212, 170, ${intensity})` 
                                : `rgba(239, 68, 68, ${intensity})`;
                              return (
                                <div 
                                  key={`${i}-${j}`}
                                  className="aspect-square flex items-center justify-center text-[8px] font-mono"
                                  style={{ backgroundColor: color }}
                                  title={`${correlationData.assets[i]} vs ${correlationData.assets[j]}: ${val}`}
                                >
                                  {i === j ? '' : val.toFixed(1)}
                                </div>
                              );
                            })}
                          </>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                    Fetch data to see correlations
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Efficient Frontier */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  Efficient Frontier
                </CardTitle>
              </CardHeader>
              <CardContent>
                {efficientFrontier && efficientFrontier.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis 
                        dataKey="volatility" 
                        name="Risk (Volatility)" 
                        unit="%"
                        stroke="#666"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'Risk (%)', position: 'bottom', fill: '#666' }}
                      />
                      <YAxis 
                        dataKey="return" 
                        name="Return" 
                        unit="%"
                        stroke="#666"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'Return (%)', angle: -90, position: 'left', fill: '#666' }}
                      />
                      <Tooltip 
                        formatter={(value, name) => [`${value}%`, name]}
                        contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                      />
                      <Scatter 
                        data={efficientFrontier} 
                        fill="#00d4aa"
                        line={{ stroke: '#00d4aa', strokeWidth: 2 }}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                    Run optimization to see efficient frontier
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Asset Statistics */}
            <Card className="bg-card border-border lg:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Asset Performance Overview
                </CardTitle>
              </CardHeader>
              <CardContent>
                {Object.keys(assetStats).length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={Object.values(assetStats).slice(0, 15)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis 
                        dataKey="symbol" 
                        stroke="#666" 
                        tick={{ fontSize: 10 }}
                        tickFormatter={(v) => v.replace('/USDT', '')}
                      />
                      <YAxis stroke="#666" tick={{ fontSize: 10 }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                        formatter={(value, name) => [
                          `${value}%`,
                          name === 'expected_return' ? 'Expected Return' : 'Volatility'
                        ]}
                      />
                      <Legend />
                      <Bar dataKey="expected_return" name="Expected Return" fill="#00d4aa" />
                      <Bar dataKey="volatility" name="Volatility" fill="#f59e0b" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                    Fetch data to see asset statistics
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Portfolio;
