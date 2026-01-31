import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { 
  Settings as SettingsIcon, Save, Brain, Database, RefreshCw,
  Key, Shield, Bell, BarChart3, Palette,
  Eye, EyeOff, CheckCircle, AlertTriangle, Mail, DollarSign,
  TrendingUp, Moon, Sun, Monitor, Trash2
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

const healthCheck = async () => {
  const response = await API.get('/health');
  return response.data;
};

const Settings = () => {
  // API Keys
  const [apiKeys, setApiKeys] = useState({
    // Sentiment APIs
    cryptopanic: "",
    twitter_api_key: "",
    twitter_api_secret: "",
    reddit_client_id: "",
    reddit_client_secret: "",
    // Market Data APIs
    glassnode: "",
    alphavantage: "",
    // LLM APIs
    openai: "",
    claude: "",
    gemini: "",
    deepseek: "",
    grok: "",
    kimi: "",
    // Exchange APIs
    binance_api_key: "",
    binance_api_secret: "",
    coinbase_api_key: "",
    coinbase_api_secret: "",
    kucoin_api_key: "",
    kucoin_api_secret: "",
    kucoin_passphrase: "",
    kraken_api_key: "",
    kraken_api_secret: "",
    bybit_api_key: "",
    bybit_api_secret: "",
    okx_api_key: "",
    okx_api_secret: "",
    okx_passphrase: "",
    // Custom APIs
    custom_api_1_name: "",
    custom_api_1_key: "",
    custom_api_1_url: "",
    custom_api_2_name: "",
    custom_api_2_key: "",
    custom_api_2_url: "",
    custom_api_3_name: "",
    custom_api_3_key: "",
    custom_api_3_url: ""
  });
  const [showKeys, setShowKeys] = useState({});
  const [llmProviders, setLlmProviders] = useState([]);

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

  // Theme Settings
  const [themeSettings, setThemeSettings] = useState({
    theme: "dark",
    chart_style: "area",
    show_volume: true,
    show_indicators: true,
    animation_enabled: true
  });
  
  // System Status
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  // Fetch initial data
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [settingsRes, healthRes, llmRes] = await Promise.all([
        getSettings(),
        healthCheck(),
        API.get('/llm/providers').then(r => r.data).catch(() => ({ providers: [] }))
      ]);

      if (settingsRes) {
        if (settingsRes.api_keys) setApiKeys(prev => ({ ...prev, ...settingsRes.api_keys }));
        if (settingsRes.trading) setTradingSettings(prev => ({ ...prev, ...settingsRes.trading }));
        if (settingsRes.notifications) setNotifications(prev => ({ ...prev, ...settingsRes.notifications }));
        if (settingsRes.theme) setThemeSettings(prev => ({ ...prev, ...settingsRes.theme }));
      }

      setHealth(healthRes);
      setLlmProviders(llmRes.providers || []);
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
        theme: themeSettings
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
          <p className="text-muted-foreground mt-1">Configure API keys, trading defaults, and preferences</p>
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

      {/* Main Tabs - 6 tabs */}
      <Tabs defaultValue="exchanges" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="exchanges" className="gap-1 text-xs"><Database className="w-3 h-3" /> Exchanges</TabsTrigger>
          <TabsTrigger value="llm" className="gap-1 text-xs"><Brain className="w-3 h-3" /> LLM Models</TabsTrigger>
          <TabsTrigger value="api-keys" className="gap-1 text-xs"><Key className="w-3 h-3" /> Data APIs</TabsTrigger>
          <TabsTrigger value="trading" className="gap-1 text-xs"><Shield className="w-3 h-3" /> Trading</TabsTrigger>
          <TabsTrigger value="notifications" className="gap-1 text-xs"><Bell className="w-3 h-3" /> Alerts</TabsTrigger>
          <TabsTrigger value="theme" className="gap-1 text-xs"><Palette className="w-3 h-3" /> Display</TabsTrigger>
        </TabsList>

        {/* ==================== EXCHANGES TAB ==================== */}
        <TabsContent value="exchanges" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Binance & Coinbase */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  Binance
                </CardTitle>
                <CardDescription>World&apos;s largest crypto exchange</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.binance_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.binance ? "text" : "password"}
                      placeholder="Enter Binance API key"
                      value={apiKeys.binance_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, binance_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('binance')}>
                      {showKeys.binance ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.binance ? "text" : "password"}
                    placeholder="Enter Binance API secret"
                    value={apiKeys.binance_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, binance_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
                <p className="text-xs text-muted-foreground">Get keys from binance.com/en/my/settings/api-management</p>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  Coinbase
                </CardTitle>
                <CardDescription>US-based regulated exchange</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.coinbase_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.coinbase ? "text" : "password"}
                      placeholder="Enter Coinbase API key"
                      value={apiKeys.coinbase_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, coinbase_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('coinbase')}>
                      {showKeys.coinbase ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.coinbase ? "text" : "password"}
                    placeholder="Enter Coinbase API secret"
                    value={apiKeys.coinbase_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, coinbase_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
                <p className="text-xs text-muted-foreground">Get keys from coinbase.com/settings/api</p>
              </CardContent>
            </Card>

            {/* KuCoin & Kraken */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  KuCoin
                </CardTitle>
                <CardDescription>Popular altcoin exchange</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.kucoin_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.kucoin ? "text" : "password"}
                      placeholder="Enter KuCoin API key"
                      value={apiKeys.kucoin_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, kucoin_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('kucoin')}>
                      {showKeys.kucoin ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.kucoin ? "text" : "password"}
                    placeholder="Enter KuCoin API secret"
                    value={apiKeys.kucoin_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, kucoin_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Passphrase</Label>
                  <Input
                    type={showKeys.kucoin ? "text" : "password"}
                    placeholder="Enter KuCoin passphrase"
                    value={apiKeys.kucoin_passphrase}
                    onChange={(e) => setApiKeys({ ...apiKeys, kucoin_passphrase: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  Kraken
                </CardTitle>
                <CardDescription>Established secure exchange</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.kraken_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.kraken ? "text" : "password"}
                      placeholder="Enter Kraken API key"
                      value={apiKeys.kraken_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, kraken_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('kraken')}>
                      {showKeys.kraken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.kraken ? "text" : "password"}
                    placeholder="Enter Kraken API secret"
                    value={apiKeys.kraken_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, kraken_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Bybit & OKX */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  Bybit
                </CardTitle>
                <CardDescription>Derivatives & spot trading</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.bybit_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.bybit ? "text" : "password"}
                      placeholder="Enter Bybit API key"
                      value={apiKeys.bybit_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, bybit_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('bybit')}>
                      {showKeys.bybit ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.bybit ? "text" : "password"}
                    placeholder="Enter Bybit API secret"
                    value={apiKeys.bybit_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, bybit_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  OKX
                </CardTitle>
                <CardDescription>Global crypto exchange</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    API Key
                    {apiKeys.okx_api_key && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.okx ? "text" : "password"}
                      placeholder="Enter OKX API key"
                      value={apiKeys.okx_api_key}
                      onChange={(e) => setApiKeys({ ...apiKeys, okx_api_key: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('okx')}>
                      {showKeys.okx ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">API Secret</Label>
                  <Input
                    type={showKeys.okx ? "text" : "password"}
                    placeholder="Enter OKX API secret"
                    value={apiKeys.okx_api_secret}
                    onChange={(e) => setApiKeys({ ...apiKeys, okx_api_secret: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Passphrase</Label>
                  <Input
                    type={showKeys.okx ? "text" : "password"}
                    placeholder="Enter OKX passphrase"
                    value={apiKeys.okx_passphrase}
                    onChange={(e) => setApiKeys({ ...apiKeys, okx_passphrase: e.target.value })}
                    className="bg-secondary border-border"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Custom APIs */}
            <Card className="bg-card border-border lg:col-span-2">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Key className="w-4 h-4 text-primary" />
                  Custom APIs
                </CardTitle>
                <CardDescription>Add your own custom API integrations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[1, 2, 3].map((num) => (
                    <div key={num} className="p-4 bg-secondary/30 rounded-lg border border-border space-y-3">
                      <div className="space-y-2">
                        <Label className="text-xs">Custom API {num} - Name</Label>
                        <Input
                          placeholder="API Name"
                          value={apiKeys[`custom_api_${num}_name`]}
                          onChange={(e) => setApiKeys({ ...apiKeys, [`custom_api_${num}_name`]: e.target.value })}
                          className="bg-secondary border-border"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Base URL</Label>
                        <Input
                          placeholder="https://api.example.com"
                          value={apiKeys[`custom_api_${num}_url`]}
                          onChange={(e) => setApiKeys({ ...apiKeys, [`custom_api_${num}_url`]: e.target.value })}
                          className="bg-secondary border-border"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">API Key</Label>
                        <Input
                          type="password"
                          placeholder="Enter API key"
                          value={apiKeys[`custom_api_${num}_key`]}
                          onChange={(e) => setApiKeys({ ...apiKeys, [`custom_api_${num}_key`]: e.target.value })}
                          className="bg-secondary border-border"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Exchange Status Summary */}
            <Card className="bg-card border-border lg:col-span-2">
              <CardHeader>
                <CardTitle className="font-mono text-base">Exchange Connection Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                  {[
                    { name: "Binance", key: "binance_api_key" },
                    { name: "Coinbase", key: "coinbase_api_key" },
                    { name: "KuCoin", key: "kucoin_api_key" },
                    { name: "Kraken", key: "kraken_api_key" },
                    { name: "Bybit", key: "bybit_api_key" },
                    { name: "OKX", key: "okx_api_key" }
                  ].map((exchange) => (
                    <div key={exchange.name} className={`p-3 rounded-lg text-center ${apiKeys[exchange.key] ? 'bg-success/20 border border-success/50' : 'bg-secondary/30 border border-border'}`}>
                      <p className="font-mono text-sm">{exchange.name}</p>
                      {apiKeys[exchange.key] ? (
                        <CheckCircle className="w-4 h-4 mx-auto mt-1 text-success" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 mx-auto mt-1 text-muted-foreground" />
                      )}
                      <p className="text-xs text-muted-foreground mt-1">
                        {apiKeys[exchange.key] ? "Configured" : "Not set"}
                      </p>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-4 text-center">
                  Configure exchange APIs to enable real-time data and trading features
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ==================== LLM MODELS TAB ==================== */}
        <TabsContent value="llm" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* OpenAI & Claude */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  OpenAI & Claude
                </CardTitle>
                <CardDescription>Premium LLM providers for trading analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* OpenAI */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    OpenAI API Key (GPT-4, GPT-5)
                    {llmProviders.includes("openai") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      data-testid="openai-key-input"
                      type={showKeys.openai ? "text" : "password"}
                      placeholder="sk-..."
                      value={apiKeys.openai}
                      onChange={(e) => setApiKeys({ ...apiKeys, openai: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('openai')}>
                      {showKeys.openai ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from platform.openai.com</p>
                </div>

                <Separator />

                {/* Claude */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Claude API Key (Anthropic)
                    {llmProviders.includes("claude") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.claude ? "text" : "password"}
                      placeholder="sk-ant-..."
                      value={apiKeys.claude}
                      onChange={(e) => setApiKeys({ ...apiKeys, claude: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('claude')}>
                      {showKeys.claude ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from console.anthropic.com</p>
                </div>
              </CardContent>
            </Card>

            {/* Gemini & DeepSeek */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  Gemini & DeepSeek
                </CardTitle>
                <CardDescription>Alternative LLM providers</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Gemini */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Google Gemini API Key
                    {llmProviders.includes("gemini") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.gemini ? "text" : "password"}
                      placeholder="AI..."
                      value={apiKeys.gemini}
                      onChange={(e) => setApiKeys({ ...apiKeys, gemini: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('gemini')}>
                      {showKeys.gemini ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from aistudio.google.com</p>
                </div>

                <Separator />

                {/* DeepSeek */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    DeepSeek API Key
                    {llmProviders.includes("deepseek") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.deepseek ? "text" : "password"}
                      placeholder="sk-..."
                      value={apiKeys.deepseek}
                      onChange={(e) => setApiKeys({ ...apiKeys, deepseek: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('deepseek')}>
                      {showKeys.deepseek ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from platform.deepseek.com</p>
                </div>
              </CardContent>
            </Card>

            {/* Grok & Kimi */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  Grok & Kimi
                </CardTitle>
                <CardDescription>Emerging LLM providers</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Grok */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    xAI Grok API Key
                    {llmProviders.includes("grok") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.grok ? "text" : "password"}
                      placeholder="xai-..."
                      value={apiKeys.grok}
                      onChange={(e) => setApiKeys({ ...apiKeys, grok: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('grok')}>
                      {showKeys.grok ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from x.ai</p>
                </div>

                <Separator />

                {/* Kimi */}
                <div className="space-y-2">
                  <Label className="text-xs flex items-center gap-2">
                    Moonshot Kimi API Key
                    {llmProviders.includes("kimi") && <CheckCircle className="w-3 h-3 text-success" />}
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      type={showKeys.kimi ? "text" : "password"}
                      placeholder="..."
                      value={apiKeys.kimi}
                      onChange={(e) => setApiKeys({ ...apiKeys, kimi: e.target.value })}
                      className="bg-secondary border-border"
                    />
                    <Button variant="outline" size="icon" onClick={() => toggleKeyVisibility('kimi')}>
                      {showKeys.kimi ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">Get key from moonshot.cn</p>
                </div>
              </CardContent>
            </Card>

            {/* LLM Status */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  LLM Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    {["openai", "claude", "gemini", "deepseek", "grok", "kimi"].map((provider) => (
                      <div key={provider} className={`p-3 rounded-lg border ${llmProviders.includes(provider) ? 'border-success bg-success/10' : 'border-border bg-secondary/30'}`}>
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-sm capitalize">{provider}</span>
                          {llmProviders.includes(provider) ? (
                            <CheckCircle className="w-4 h-4 text-success" />
                          ) : (
                            <AlertTriangle className="w-4 h-4 text-muted-foreground" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {llmProviders.includes(provider) ? "Ready" : "Not configured"}
                        </p>
                      </div>
                    ))}
                  </div>
                  
                  <div className="p-3 bg-primary/10 rounded-lg border border-primary/30">
                    <h4 className="text-xs font-mono text-primary mb-2">Active Providers: {llmProviders.length}/6</h4>
                    <p className="text-xs text-muted-foreground">
                      {llmProviders.length > 0 
                        ? `Using: ${llmProviders.join(", ")}`
                        : "Add API keys and save to enable LLM features"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

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

                {/* System Status */}
                <div className="p-3 bg-primary/10 rounded-lg border border-primary/30">
                  <h4 className="text-xs font-mono text-primary mb-2">System Status</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {health?.services && Object.entries(health.services).map(([service, status]) => (
                      <div key={service} className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${status === 'active' || status === 'connected' ? 'bg-success' : 'bg-warning'}`}></div>
                        <span className="text-xs capitalize">{service.replace(/_/g, ' ')}</span>
                      </div>
                    ))}
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
      </Tabs>
    </div>
  );
};

export default Settings;
