import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { 
  Settings as SettingsIcon, Save, Brain, Cpu, Database, RefreshCw,
  Key, Shield, Bell, BarChart3, Palette, FolderOpen, Trash2,
  Eye, EyeOff, CheckCircle, AlertTriangle, Mail, DollarSign,
  TrendingUp, Clock, Globe, Moon, Sun, Monitor
} from "lucide-react";
import { API } from "@/lib/api";

// API functions
const getSettings = async () => {
  try {
    const response = await API.get('/settings');
    return response.data;
  } catch {
    return null;
  }
};

const saveSettings = async (settings) => {
  const response = await API.post('/settings', settings);
  return response.data;
};

const getSavedModels = async () => {
  const response = await API.get('/models/saved');
  return response.data;
};

const deleteModel = async (modelPath) => {
  const response = await API.delete(`/models/${encodeURIComponent(modelPath)}`);
  return response.data;
};

const loadModel = async (modelPath) => {
  const response = await API.post('/models/load', null, { params: { model_path: modelPath } });
  return response.data;
};

const healthCheck = async () => {
  const response = await API.get('/health');
  return response.data;
};

const Settings = () => {
  // API Keys
  const [apiKeys, setApiKeys] = useState({
    cryptopanic: "",
    twitter_api_key: "",
    twitter_api_secret: "",
    reddit_client_id: "",
    reddit_client_secret: "",
    glassnode: "",
    alphavantage: ""
  });
  const [showKeys, setShowKeys] = useState({});

  // Trading Settings
  const [tradingSettings, setTradingSettings] = useState({
    default_stop_loss: 2,
    default_take_profit: 4,
    default_position_size: 10,
    max_daily_trades: 10,
    max_open_positions: 3,
    risk_per_trade: 2,
    use_trailing_stop: false,
    trailing_stop_pct: 1
  });

  // Notification Settings
  const [notifications, setNotifications] = useState({
    email_enabled: false,
    email_address: "",
    alert_on_prediction: true,
    alert_on_trade: true,
    alert_on_stop_loss: true,
    alert_on_take_profit: true,
    price_alerts: []
  });
  const [newPriceAlert, setNewPriceAlert] = useState({ symbol: "BTC/USDT", price: "", direction: "above" });

  // Data Source Settings
  const [dataSettings, setDataSettings] = useState({
    exchange: "binanceus",
    default_symbol: "BTC/USDT",
    default_timeframe: "1h",
    trading_pairs: ["BTC/USDT", "ETH/USDT"],
    data_lookback_days: 30
  });

  // Theme Settings
  const [themeSettings, setThemeSettings] = useState({
    theme: "dark",
    chart_style: "area",
    show_volume: true,
    show_indicators: true,
    animation_enabled: true
  });

  // Model Management
  const [savedModels, setSavedModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  
  // System Status
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  // Fetch initial data
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [settingsRes, modelsRes, healthRes] = await Promise.all([
        getSettings(),
        getSavedModels(),
        healthCheck()
      ]);

      if (settingsRes) {
        if (settingsRes.api_keys) setApiKeys(prev => ({ ...prev, ...settingsRes.api_keys }));
        if (settingsRes.trading) setTradingSettings(prev => ({ ...prev, ...settingsRes.trading }));
        if (settingsRes.notifications) setNotifications(prev => ({ ...prev, ...settingsRes.notifications }));
        if (settingsRes.data_source) setDataSettings(prev => ({ ...prev, ...settingsRes.data_source }));
        if (settingsRes.theme) setThemeSettings(prev => ({ ...prev, ...settingsRes.theme }));
        if (settingsRes.active_model) setActiveModel(settingsRes.active_model);
      }

      setSavedModels(modelsRes?.models || []);
      setHealth(healthRes);
    } catch (error) {
      console.error("Error fetching settings:", error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Save all settings
  const handleSaveSettings = async () => {
    setSaving(true);
    try {
      await saveSettings({
        api_keys: apiKeys,
        trading: tradingSettings,
        notifications: notifications,
        data_source: dataSettings,
        theme: themeSettings,
        active_model: activeModel
      });
      toast.success("Settings saved successfully!");
    } catch (error) {
      toast.error("Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  // Toggle key visibility
  const toggleKeyVisibility = (key) => {
    setShowKeys(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Delete model
  const handleDeleteModel = async (modelPath) => {
    if (!confirm("Are you sure you want to delete this model?")) return;
    try {
      await deleteModel(modelPath);
      toast.success("Model deleted");
      fetchData();
    } catch (error) {
      toast.error("Failed to delete model");
    }
  };

  // Load model
  const handleLoadModel = async (model) => {
    try {
      await loadModel(model.path);
      setActiveModel(model.path);
      toast.success(`Model loaded: ${model.network_type}`);
    } catch (error) {
      toast.error("Failed to load model");
    }
  };

  // Add price alert
  const addPriceAlert = () => {
    if (!newPriceAlert.price) {
      toast.error("Please enter a price");
      return;
    }
    setNotifications(prev => ({
      ...prev,
      price_alerts: [...prev.price_alerts, { ...newPriceAlert, id: Date.now() }]
    }));
    setNewPriceAlert({ symbol: "BTC/USDT", price: "", direction: "above" });
    toast.success("Price alert added");
  };

  // Remove price alert
  const removePriceAlert = (id) => {
    setNotifications(prev => ({
      ...prev,
      price_alerts: prev.price_alerts.filter(a => a.id !== id)
    }));
  };

  return (
    <div data-testid="settings-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Settings</h1>
          <p className="text-muted-foreground mt-1">Configure your trading system</p>
        </div>
        <Button 
          data-testid="save-all-settings-btn"
          onClick={handleSaveSettings} 
          disabled={saving}
          className="gap-2"
        >
          {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save All Settings
        </Button>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="api-keys" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="api-keys" className="gap-1 text-xs"><Key className="w-3 h-3" /> API Keys</TabsTrigger>
          <TabsTrigger value="trading" className="gap-1 text-xs"><Shield className="w-3 h-3" /> Trading</TabsTrigger>
          <TabsTrigger value="notifications" className="gap-1 text-xs"><Bell className="w-3 h-3" /> Alerts</TabsTrigger>
          <TabsTrigger value="data" className="gap-1 text-xs"><Database className="w-3 h-3" /> Data</TabsTrigger>
          <TabsTrigger value="theme" className="gap-1 text-xs"><Palette className="w-3 h-3" /> Display</TabsTrigger>
          <TabsTrigger value="models" className="gap-1 text-xs"><Brain className="w-3 h-3" /> Models</TabsTrigger>
        </TabsList>

        {/* ==================== API KEYS TAB ==================== */}
        <TabsContent value="api-keys" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Sentiment APIs */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  Sentiment Data APIs
                </CardTitle>
                <CardDescription>Configure news and social sentiment sources</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* CryptoPanic */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    CryptoPanic API Key
                    {apiKeys.cryptopanic && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      data-testid="cryptopanic-key-input"
                      type={showKeys.cryptopanic ? "text" : "password"}
                      placeholder="Enter CryptoPanic API key"
                      value={apiKeys.cryptopanic}
                      onChange={(e) => setApiKeys({ ...apiKeys, cryptopanic: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('cryptopanic')}>
                      {showKeys.cryptopanic ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from cryptopanic.com/developers/api</p>
                </div>

                <Separator />

                {/* Twitter */}
                <div className="space-y-2">
                  <Label className="text-xs">Twitter API Key</Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.twitter ? "text" : "password"}
                      placeholder="API Key"
                      value={apiKeys.twitter_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, twitter_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('twitter')}>
                      {showKeys.twitter ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <Input
                    type={showKeys.twitter ? "text" : "password"}
                    placeholder="API Secret"
                    value={apiKeys.twitter_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, twitter_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>

                <Separator />

                {/* Reddit */}
                <div className="space-y-2">
                  <Label className="text-xs">Reddit API</Label>
                  <Input
                    type={showKeys.reddit ? "text" : "password"}
                    placeholder="Client ID"
                    value={apiKeys.reddit_client_id}
                    onChange={(e) => setApiKeys({ ...apiKeys, reddit_client_id: e.target.value })}
                    className="bg-secondary border-border"
                  />
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.reddit ? "text" : "password"}
                      placeholder="Client Secret"
                      value={apiKeys.reddit_client_secret}
                      onChange={(e) => setApiKeys({ ...apiKeys, reddit_client_secret: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('reddit')}>
                      {showKeys.reddit ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Market Data APIs */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Market Data APIs
                </CardTitle>
                <CardDescription>Configure on-chain and market data sources</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Glassnode */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Glassnode API Key
                    <Badge variant="outline" className="text-xs">On-Chain Data</Badge>
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.glassnode ? "text" : "password"}
                      placeholder="Enter Glassnode API key"
                      value={apiKeys.glassnode}
                      onChange={(e) => setApiKeys({ ...apiKeys, glassnode: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('glassnode')}>
                      {showKeys.glassnode ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">For whale tracking, exchange flows</p>
                </div>

                <Separator />

                {/* Alpha Vantage */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Alpha Vantage API Key
                    <Badge variant="outline" className="text-xs">Stock Data</Badge>
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.alphavantage ? "text" : "password"}
                      placeholder="Enter Alpha Vantage key"
                      value={apiKeys.alphavantage}
                      onChange={(e) => setApiKeys({ ...apiKeys, alphavantage: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('alphavantage')}>
                      {showKeys.alphavantage ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">For S&P 500, DXY correlation</p>
                </div>

                {/* Status Summary */}
                <div className="p-3 bg-secondary/50 rounded-lg mt-4">
                  <h4 className="text-xs font-mono text-muted-foreground mb-2">API Status</h4>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2">
                      {apiKeys.cryptopanic ? <CheckCircle className="w-3 h-3 text-success" /> : <AlertTriangle className="w-3 h-3 text-warning" />}
                      <span className="text-xs">CryptoPanic</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {apiKeys.twitter_api_key ? <CheckCircle className="w-3 h-3 text-success" /> : <AlertTriangle className="w-3 h-3 text-warning" />}
                      <span className="text-xs">Twitter</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {apiKeys.reddit_client_id ? <CheckCircle className="w-3 h-3 text-success" /> : <AlertTriangle className="w-3 h-3 text-warning" />}
                      <span className="text-xs">Reddit</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {apiKeys.glassnode ? <CheckCircle className="w-3 h-3 text-success" /> : <AlertTriangle className="w-3 h-3 text-warning" />}
                      <span className="text-xs">Glassnode</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== TRADING TAB ==================== */}
        <TabsContent value="trading" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Risk Management */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Shield className="w-4 h-4 text-primary" />
                  Risk Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Default Stop Loss</Label>
                    <span className="font-mono text-destructive">{tradingSettings.default_stop_loss}%</span>
                  </div>
                  <Slider 
                    value={[tradingSettings.default_stop_loss]} 
                    onValueChange={([v]) => setTradingSettings({ ...tradingSettings, default_stop_loss: v })}
                    min={0.5} max={10} step={0.5}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Default Take Profit</Label>
                    <span className="font-mono text-success">{tradingSettings.default_take_profit}%</span>
                  </div>
                  <Slider 
                    value={[tradingSettings.default_take_profit]} 
                    onValueChange={([v]) => setTradingSettings({ ...tradingSettings, default_take_profit: v })}
                    min={1} max={20} step={0.5}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Risk Per Trade</Label>
                    <span className="font-mono text-primary">{tradingSettings.risk_per_trade}%</span>
                  </div>
                  <Slider 
                    value={[tradingSettings.risk_per_trade]} 
                    onValueChange={([v]) => setTradingSettings({ ...tradingSettings, risk_per_trade: v })}
                    min={0.5} max={5} step={0.5}
                  />
                  <p className="text-xs text-muted-foreground">Max % of capital risked per trade</p>
                </div>

                <Separator />

                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded">
                  <div>
                    <Label className="text-sm">Trailing Stop Loss</Label>
                    <p className="text-xs text-muted-foreground">Lock in profits as price moves</p>
                  </div>
                  <Switch 
                    checked={tradingSettings.use_trailing_stop} 
                    onCheckedChange={(v) => setTradingSettings({ ...tradingSettings, use_trailing_stop: v })}
                  />
                </div>

                {tradingSettings.use_trailing_stop && (
                  <div className="space-y-2 pl-4 border-l-2 border-primary/30">
                    <div className="flex justify-between text-xs">
                      <Label>Trailing Distance</Label>
                      <span className="font-mono">{tradingSettings.trailing_stop_pct}%</span>
                    </div>
                    <Slider 
                      value={[tradingSettings.trailing_stop_pct]} 
                      onValueChange={([v]) => setTradingSettings({ ...tradingSettings, trailing_stop_pct: v })}
                      min={0.5} max={5} step={0.5}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Position Settings */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <DollarSign className="w-4 h-4 text-primary" />
                  Position Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Default Position Size</Label>
                    <span className="font-mono text-primary">{tradingSettings.default_position_size}%</span>
                  </div>
                  <Slider 
                    value={[tradingSettings.default_position_size]} 
                    onValueChange={([v]) => setTradingSettings({ ...tradingSettings, default_position_size: v })}
                    min={5} max={50} step={5}
                  />
                  <p className="text-xs text-muted-foreground">% of capital per trade</p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label className="text-xs">Max Daily Trades</Label>
                  <Input
                    type="number"
                    value={tradingSettings.max_daily_trades}
                    onChange={(e) => setTradingSettings({ ...tradingSettings, max_daily_trades: parseInt(e.target.value) || 10 })}
                    className="bg-secondary border-border"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Max Open Positions</Label>
                  <Input
                    type="number"
                    value={tradingSettings.max_open_positions}
                    onChange={(e) => setTradingSettings({ ...tradingSettings, max_open_positions: parseInt(e.target.value) || 3 })}
                    className="bg-secondary border-border"
                  />
                </div>

                {/* Risk/Reward Summary */}
                <div className="p-3 bg-primary/10 rounded border border-primary/30">
                  <h4 className="text-xs font-mono text-primary mb-2">Risk/Reward Summary</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>R/R Ratio: <span className="font-mono">{(tradingSettings.default_take_profit / tradingSettings.default_stop_loss).toFixed(1)}:1</span></div>
                    <div>Max Risk: <span className="font-mono">{tradingSettings.risk_per_trade * tradingSettings.max_open_positions}%</span></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== NOTIFICATIONS TAB ==================== */}
        <TabsContent value="notifications" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Email Alerts */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Mail className="w-4 h-4 text-primary" />
                  Email Notifications
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-secondary/30 rounded">
                  <div>
                    <Label className="text-sm">Enable Email Alerts</Label>
                    <p className="text-xs text-muted-foreground">Receive email notifications</p>
                  </div>
                  <Switch 
                    checked={notifications.email_enabled} 
                    onCheckedChange={(v) => setNotifications({ ...notifications, email_enabled: v })}
                  />
                </div>

                {notifications.email_enabled && (
                  <div className="space-y-2">
                    <Label className="text-xs">Email Address</Label>
                    <Input
                      type="email"
                      placeholder="your@email.com"
                      value={notifications.email_address}
                      onChange={(e) => setNotifications({ ...notifications, email_address: e.target.value })}
                      className="bg-secondary border-border"
                    />
                  </div>
                )}

                <Separator />

                <div className="space-y-3">
                  <Label className="text-xs text-muted-foreground">Alert Types</Label>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">New Prediction</span>
                    <Switch 
                      checked={notifications.alert_on_prediction} 
                      onCheckedChange={(v) => setNotifications({ ...notifications, alert_on_prediction: v })}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Trade Executed</span>
                    <Switch 
                      checked={notifications.alert_on_trade} 
                      onCheckedChange={(v) => setNotifications({ ...notifications, alert_on_trade: v })}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Stop Loss Hit</span>
                    <Switch 
                      checked={notifications.alert_on_stop_loss} 
                      onCheckedChange={(v) => setNotifications({ ...notifications, alert_on_stop_loss: v })}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Take Profit Hit</span>
                    <Switch 
                      checked={notifications.alert_on_take_profit} 
                      onCheckedChange={(v) => setNotifications({ ...notifications, alert_on_take_profit: v })}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Price Alerts */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  Price Alerts
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-3 gap-2">
                  <Select value={newPriceAlert.symbol} onValueChange={(v) => setNewPriceAlert({ ...newPriceAlert, symbol: v })}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                      <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={newPriceAlert.direction} onValueChange={(v) => setNewPriceAlert({ ...newPriceAlert, direction: v })}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="above">Above</SelectItem>
                      <SelectItem value="below">Below</SelectItem>
                    </SelectContent>
                  </Select>
                  <Input
                    type="number"
                    placeholder="Price"
                    value={newPriceAlert.price}
                    onChange={(e) => setNewPriceAlert({ ...newPriceAlert, price: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
                <Button onClick={addPriceAlert} variant="outline" className="w-full">
                  Add Price Alert
                </Button>

                <Separator />

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {notifications.price_alerts.length > 0 ? (
                    notifications.price_alerts.map((alert) => (
                      <div key={alert.id} className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{alert.symbol}</Badge>
                          <span className="text-xs">{alert.direction}</span>
                          <span className="font-mono text-sm">${alert.price}</span>
                        </div>
                        <Button variant="ghost" size="icon" onClick={() => removePriceAlert(alert.id)}>
                          <Trash2 className="w-4 h-4 text-destructive" />
                        </Button>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-muted-foreground text-center py-4">No price alerts set</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== DATA SOURCE TAB ==================== */}
        <TabsContent value="data" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Exchange Settings */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Globe className="w-4 h-4 text-primary" />
                  Exchange Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Exchange</Label>
                  <Select value={dataSettings.exchange} onValueChange={(v) => setDataSettings({ ...dataSettings, exchange: v })}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="binanceus">Binance US</SelectItem>
                      <SelectItem value="binance">Binance Global</SelectItem>
                      <SelectItem value="coinbase">Coinbase</SelectItem>
                      <SelectItem value="kraken">Kraken</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Default Trading Pair</Label>
                  <Select value={dataSettings.default_symbol} onValueChange={(v) => setDataSettings({ ...dataSettings, default_symbol: v })}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                      <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                      <SelectItem value="SOL/USDT">SOL/USDT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Default Timeframe</Label>
                  <Select value={dataSettings.default_timeframe} onValueChange={(v) => setDataSettings({ ...dataSettings, default_timeframe: v })}>
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
              </CardContent>
            </Card>

            {/* Data Range */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Clock className="w-4 h-4 text-primary" />
                  Data Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <Label>Data Lookback</Label>
                    <span className="font-mono text-primary">{dataSettings.data_lookback_days} days</span>
                  </div>
                  <Slider 
                    value={[dataSettings.data_lookback_days]} 
                    onValueChange={([v]) => setDataSettings({ ...dataSettings, data_lookback_days: v })}
                    min={7} max={365} step={7}
                  />
                  <p className="text-xs text-muted-foreground">Historical data for training/analysis</p>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label className="text-xs">Active Trading Pairs</Label>
                  <div className="flex flex-wrap gap-2">
                    {["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"].map((pair) => (
                      <Badge 
                        key={pair}
                        variant={dataSettings.trading_pairs.includes(pair) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => {
                          if (dataSettings.trading_pairs.includes(pair)) {
                            setDataSettings({ 
                              ...dataSettings, 
                              trading_pairs: dataSettings.trading_pairs.filter(p => p !== pair) 
                            });
                          } else {
                            setDataSettings({ 
                              ...dataSettings, 
                              trading_pairs: [...dataSettings.trading_pairs, pair] 
                            });
                          }
                        }}
                      >
                        {pair}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== THEME TAB ==================== */}
        <TabsContent value="theme" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Appearance */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Palette className="w-4 h-4 text-primary" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Theme</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: "dark", icon: Moon, label: "Dark" },
                      { id: "light", icon: Sun, label: "Light" },
                      { id: "system", icon: Monitor, label: "System" }
                    ].map((theme) => (
                      <div
                        key={theme.id}
                        className={`p-3 rounded-lg border cursor-pointer transition-all text-center ${
                          themeSettings.theme === theme.id
                            ? 'border-primary bg-primary/10'
                            : 'border-border bg-secondary/30 hover:border-primary/50'
                        }`}
                        onClick={() => setThemeSettings({ ...themeSettings, theme: theme.id })}
                      >
                        <theme.icon className="w-5 h-5 mx-auto mb-1" />
                        <span className="text-xs">{theme.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-sm">Animations</Label>
                    <p className="text-xs text-muted-foreground">Enable UI animations</p>
                  </div>
                  <Switch 
                    checked={themeSettings.animation_enabled} 
                    onCheckedChange={(v) => setThemeSettings({ ...themeSettings, animation_enabled: v })}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Chart Settings */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Chart Preferences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Chart Style</Label>
                  <Select value={themeSettings.chart_style} onValueChange={(v) => setThemeSettings({ ...themeSettings, chart_style: v })}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="candlestick">Candlestick</SelectItem>
                      <SelectItem value="area">Area</SelectItem>
                      <SelectItem value="line">Line</SelectItem>
                      <SelectItem value="bar">Bar</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Show Volume</span>
                    <Switch 
                      checked={themeSettings.show_volume} 
                      onCheckedChange={(v) => setThemeSettings({ ...themeSettings, show_volume: v })}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Show Indicators</span>
                    <Switch 
                      checked={themeSettings.show_indicators} 
                      onCheckedChange={(v) => setThemeSettings({ ...themeSettings, show_indicators: v })}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== MODELS TAB ==================== */}
        <TabsContent value="models" className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-lg flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  Saved Models
                </CardTitle>
                <Button variant="outline" size="sm" onClick={fetchData}>
                  <RefreshCw className="w-4 h-4 mr-1" /> Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {savedModels.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {savedModels.map((model, i) => (
                    <div 
                      key={i} 
                      className={`p-4 rounded-lg border transition-all ${
                        activeModel === model.path 
                          ? 'border-primary bg-primary/10' 
                          : 'border-border bg-secondary hover:border-primary/50'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <p className="font-mono text-sm font-semibold">{model.symbol}</p>
                          <p className="text-xs text-muted-foreground">{model.timestamp}</p>
                        </div>
                        <div className="flex items-center gap-1">
                          {activeModel === model.path && (
                            <Badge className="bg-success text-xs">Active</Badge>
                          )}
                          <Badge className="bg-primary text-xs">{model.network_type?.toUpperCase()}</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1 mb-3 text-xs">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Accuracy</span>
                          <span className="font-mono text-success">{((model.metrics?.final_accuracy || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Epochs</span>
                          <span className="font-mono">{model.metrics?.epochs_trained || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Timeframe</span>
                          <span className="font-mono">{model.config?.timeframe || '-'}</span>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <Button 
                          size="sm" 
                          className="flex-1 gap-1"
                          variant={activeModel === model.path ? "outline" : "default"}
                          onClick={() => handleLoadModel(model)}
                          disabled={activeModel === model.path}
                        >
                          <FolderOpen className="w-3 h-3" />
                          {activeModel === model.path ? "Loaded" : "Load"}
                        </Button>
                        <Button 
                          size="sm" 
                          variant="destructive"
                          onClick={() => handleDeleteModel(model.path)}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Brain className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">No saved models</p>
                  <p className="text-xs text-muted-foreground mt-1">Train a model to see it here</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* System Status */}
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="font-mono text-base flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {health?.services && Object.entries(health.services).map(([service, status]) => (
                  <div key={service} className="p-4 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">{service.replace(/_/g, ' ')}</p>
                    <Badge className={status === 'active' || status === 'connected' ? 'bg-success mt-2' : 'bg-warning mt-2'}>
                      {status}
                    </Badge>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-secondary/50 rounded-lg">
                <p className="text-xs text-muted-foreground">Last updated: {health?.timestamp ? new Date(health.timestamp).toLocaleString() : 'N/A'}</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Settings;
