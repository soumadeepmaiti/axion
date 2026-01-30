import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Settings as SettingsIcon, Save, Brain, Cpu, Database, RefreshCw } from "lucide-react";
import { getModelSummary, buildModel, analyzeSentiment, healthCheck } from "@/lib/api";

const Settings = () => {
  const [modelConfig, setModelConfig] = useState({
    lstm_units: 64,
    gru_units: 32,
    dense_units: 64,
    dropout_rate: 0.3
  });
  const [sentimentText, setSentimentText] = useState("");
  const [sentimentResult, setSentimentResult] = useState(null);
  const [useLlm, setUseLlm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [summary, healthRes] = await Promise.all([
          getModelSummary(),
          healthCheck()
        ]);
        if (summary) {
          setModelConfig({
            lstm_units: summary.lstm_units || 64,
            gru_units: summary.gru_units || 32,
            dense_units: summary.dense_units || 64,
            dropout_rate: summary.dropout_rate || 0.3
          });
        }
        setHealth(healthRes);
      } catch (error) {
        console.error("Error fetching settings:", error);
      }
    };
    fetchData();
  }, []);

  const handleSaveModel = async () => {
    setLoading(true);
    try {
      await buildModel(modelConfig);
      toast.success("Model configuration updated");
    } catch (error) {
      console.error("Error saving model:", error);
      toast.error("Failed to update model");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeSentiment = async () => {
    if (!sentimentText.trim()) {
      toast.error("Please enter text to analyze");
      return;
    }
    setLoading(true);
    try {
      const result = await analyzeSentiment(sentimentText, useLlm);
      setSentimentResult(result.result);
      toast.success("Sentiment analyzed");
    } catch (error) {
      console.error("Error analyzing sentiment:", error);
      toast.error("Failed to analyze sentiment");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div data-testid="settings-page" className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground font-mono">Settings</h1>
        <p className="text-muted-foreground mt-1">Configure model and system parameters</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Configuration */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              Model Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="lstm">LSTM Units</Label>
                <Input
                  data-testid="lstm-units-input"
                  id="lstm"
                  type="number"
                  value={modelConfig.lstm_units}
                  onChange={(e) => setModelConfig({ ...modelConfig, lstm_units: parseInt(e.target.value) })}
                  className="bg-secondary border-border"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="gru">GRU Units</Label>
                <Input
                  data-testid="gru-units-input"
                  id="gru"
                  type="number"
                  value={modelConfig.gru_units}
                  onChange={(e) => setModelConfig({ ...modelConfig, gru_units: parseInt(e.target.value) })}
                  className="bg-secondary border-border"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dense">Dense Units</Label>
                <Input
                  data-testid="dense-units-input"
                  id="dense"
                  type="number"
                  value={modelConfig.dense_units}
                  onChange={(e) => setModelConfig({ ...modelConfig, dense_units: parseInt(e.target.value) })}
                  className="bg-secondary border-border"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dropout">Dropout Rate</Label>
                <Input
                  data-testid="dropout-rate-input"
                  id="dropout"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={modelConfig.dropout_rate}
                  onChange={(e) => setModelConfig({ ...modelConfig, dropout_rate: parseFloat(e.target.value) })}
                  className="bg-secondary border-border"
                />
              </div>
            </div>
            <Button
              data-testid="save-model-btn"
              onClick={handleSaveModel}
              disabled={loading}
              className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {loading ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
              Rebuild Model
            </Button>
          </CardContent>
        </Card>

        {/* Sentiment Test */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              Sentiment Analysis Test
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="sentiment-text">Text to Analyze</Label>
              <Textarea
                data-testid="sentiment-text-input"
                id="sentiment-text"
                placeholder="Enter financial text to analyze..."
                value={sentimentText}
                onChange={(e) => setSentimentText(e.target.value)}
                className="bg-secondary border-border min-h-24"
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Switch
                  data-testid="use-llm-switch"
                  checked={useLlm}
                  onCheckedChange={setUseLlm}
                />
                <Label>Use LLM Enhancement</Label>
              </div>
              <Button
                data-testid="analyze-sentiment-btn"
                onClick={handleAnalyzeSentiment}
                disabled={loading}
                variant="outline"
              >
                Analyze
              </Button>
            </div>

            {sentimentResult && (
              <div className="p-4 bg-secondary rounded-lg space-y-3 animate-fade-in">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Sentiment Score</span>
                  <span className={`font-mono font-bold ${
                    sentimentResult.sentiment > 0 ? 'text-success' : 
                    sentimentResult.sentiment < 0 ? 'text-destructive' : 'text-foreground'
                  }`}>
                    {sentimentResult.sentiment?.toFixed(4)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Confidence</span>
                  <span className="font-mono text-primary">{(sentimentResult.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex gap-2">
                  <Badge className="bg-success/20 text-success">+{(sentimentResult.positive * 100).toFixed(0)}%</Badge>
                  <Badge className="bg-destructive/20 text-destructive">-{(sentimentResult.negative * 100).toFixed(0)}%</Badge>
                  <Badge className="bg-secondary">{(sentimentResult.neutral * 100).toFixed(0)}% neutral</Badge>
                </div>
                {sentimentResult.llm_enhanced && (
                  <p className="text-xs text-muted-foreground">{sentimentResult.llm_reasoning}</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* System Status */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="font-mono text-lg flex items-center gap-2">
              <Database className="w-5 h-5 text-primary" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {health?.services && Object.entries(health.services).map(([service, status]) => (
                <div key={service} className="p-4 bg-secondary rounded-lg">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">{service}</p>
                  <Badge className={status === 'active' || status === 'connected' ? 'bg-success mt-2' : 'bg-warning mt-2'}>
                    {status}
                  </Badge>
                </div>
              ))}
            </div>
            <div className="mt-4 p-4 bg-secondary rounded-lg">
              <p className="text-xs text-muted-foreground">Last updated: {health?.timestamp ? new Date(health.timestamp).toLocaleString() : 'N/A'}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Settings;
