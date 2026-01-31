import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import {
  Send, Bot, User, RefreshCw, Brain, TrendingUp, TrendingDown,
  AlertTriangle, Loader2, MessageSquare, Vote, BarChart3,
  ThumbsUp, ThumbsDown, Minus, Clock, Zap
} from "lucide-react";
import { API } from "@/lib/api";

const Advisor = () => {
  // Chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("openai");
  const [availableProviders, setAvailableProviders] = useState([]);
  const [useMultiChat, setUseMultiChat] = useState(false);
  const messagesEndRef = useRef(null);

  // Ensemble signal state
  const [signalLoading, setSignalLoading] = useState(false);
  const [ensembleSignal, setEnsembleSignal] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState("BTC/USDT");

  // Market context
  const [marketContext, setMarketContext] = useState(null);

  // Fetch providers on mount
  useEffect(() => {
    fetchProviders();
    fetchMarketContext();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchProviders = async () => {
    try {
      const response = await API.get('/llm/providers');
      setAvailableProviders(response.data.providers || []);
      if (response.data.providers?.length > 0) {
        setSelectedProvider(response.data.providers[0]);
      }
    } catch (error) {
      console.error("Error fetching providers:", error);
    }
  };

  const fetchMarketContext = async () => {
    try {
      const response = await API.get('/market/latest/BTC/USDT');
      setMarketContext(response.data);
    } catch (error) {
      console.error("Error fetching market context:", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    if (availableProviders.length === 0) {
      toast.error("No LLM providers configured. Add API keys in Settings.");
      return;
    }

    const userMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      let response;
      if (useMultiChat) {
        response = await API.post('/llm/multi-chat', {
          message: input,
          context: marketContext ? `Current BTC: $${marketContext.price?.toLocaleString()}` : null
        });
        
        const results = response.data.results || [];
        const multiResponse = {
          role: "assistant",
          isMulti: true,
          responses: results.map(r => ({
            provider: r.provider,
            content: r.content || r.error || "No response",
            latency: r.latency_ms
          }))
        };
        setMessages(prev => [...prev, multiResponse]);
      } else {
        response = await API.post('/llm/chat', {
          message: input,
          provider: selectedProvider,
          context: marketContext ? `Current BTC: $${marketContext.price?.toLocaleString()}` : null
        });
        
        const assistantMessage = {
          role: "assistant",
          provider: response.data.provider,
          content: response.data.content || response.data.error || "No response",
          latency: response.data.latency_ms
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      const errorMessage = {
        role: "assistant",
        content: `Error: ${error.response?.data?.detail || error.message}`,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      toast.error("Failed to get response");
    } finally {
      setLoading(false);
    }
  };

  const handleGetSignal = async () => {
    if (availableProviders.length === 0) {
      toast.error("No LLM providers configured. Add API keys in Settings.");
      return;
    }

    setSignalLoading(true);
    try {
      const response = await API.post('/llm/signal', { symbol: selectedSymbol });
      setEnsembleSignal(response.data);
      toast.success(`Ensemble signal: ${response.data.signal}`);
    } catch (error) {
      toast.error("Failed to get ensemble signal");
    } finally {
      setSignalLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatLatency = (ms) => {
    if (!ms) return '';
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const getSignalColor = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'BUY': return 'text-success bg-success/20';
      case 'SELL': return 'text-destructive bg-destructive/20';
      default: return 'text-warning bg-warning/20';
    }
  };

  const getSignalIcon = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'BUY': return <TrendingUp className="w-6 h-6" />;
      case 'SELL': return <TrendingDown className="w-6 h-6" />;
      default: return <Minus className="w-6 h-6" />;
    }
  };

  return (
    <div data-testid="advisor-page" className="p-6 h-[calc(100vh-100px)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">AI Trading Advisor</h1>
          <p className="text-muted-foreground mt-1">Chat with AI models for market insights</p>
        </div>
        <div className="flex items-center gap-4">
          {marketContext && (
            <div className="text-right">
              <p className="text-xs text-muted-foreground">BTC/USDT</p>
              <p className={`font-mono text-lg ${marketContext.change_percent >= 0 ? 'text-success' : 'text-destructive'}`}>
                ${marketContext.price?.toLocaleString()}
              </p>
            </div>
          )}
          <Badge className={availableProviders.length > 0 ? 'bg-success' : 'bg-warning'}>
            {availableProviders.length} LLM{availableProviders.length !== 1 ? 's' : ''} Active
          </Badge>
        </div>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="chat" className="flex-1 flex flex-col">
        <TabsList className="grid w-full grid-cols-2 mb-4">
          <TabsTrigger value="chat" className="gap-2"><MessageSquare className="w-4 h-4" /> Chat Advisor</TabsTrigger>
          <TabsTrigger value="ensemble" className="gap-2"><Vote className="w-4 h-4" /> Ensemble Signal</TabsTrigger>
        </TabsList>

        {/* ==================== CHAT TAB ==================== */}
        <TabsContent value="chat" className="flex-1 flex flex-col">
          <div className="flex gap-4 mb-4">
            <div className="flex items-center gap-2">
              <Label className="text-xs whitespace-nowrap">Provider:</Label>
              <Select value={selectedProvider} onValueChange={setSelectedProvider} disabled={useMultiChat}>
                <SelectTrigger className="w-32 bg-secondary border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableProviders.map(p => (
                    <SelectItem key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</SelectItem>
                  ))}
                  {availableProviders.length === 0 && (
                    <SelectItem value="none" disabled>No providers</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={useMultiChat} onCheckedChange={setUseMultiChat} />
              <Label className="text-xs">Multi-LLM (compare responses)</Label>
            </div>
          </div>

          {/* Chat Messages */}
          <Card className="flex-1 bg-card border-border overflow-hidden flex flex-col">
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center">
                    <Brain className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                    <h3 className="text-lg font-semibold mb-2">Start a conversation</h3>
                    <p className="text-muted-foreground text-sm max-w-md">
                      Ask about market analysis, trading strategies, technical indicators, or any crypto-related questions.
                    </p>
                    <div className="flex flex-wrap gap-2 justify-center mt-4">
                      {["What's your BTC outlook?", "Explain RSI divergence", "Best entry for ETH?"].map((q) => (
                        <Button
                          key={q}
                          variant="outline"
                          size="sm"
                          onClick={() => setInput(q)}
                          className="text-xs"
                        >
                          {q}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.isMulti ? (
                    // Multi-response layout
                    <div className="w-full space-y-2">
                      <p className="text-xs text-muted-foreground text-center">Responses from {msg.responses.length} providers</p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {msg.responses.map((r, j) => (
                          <div key={j} className="p-3 rounded-lg bg-secondary/50 border border-border">
                            <div className="flex items-center justify-between mb-2">
                              <Badge variant="outline" className="capitalize">{r.provider}</Badge>
                              {r.latency && <span className="text-xs text-muted-foreground">{formatLatency(r.latency)}</span>}
                            </div>
                            <p className="text-sm whitespace-pre-wrap">{r.content}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    // Single message
                    <div className={`max-w-[80%] p-3 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-primary text-primary-foreground' 
                        : msg.isError 
                          ? 'bg-destructive/20 border border-destructive' 
                          : 'bg-secondary border border-border'
                    }`}>
                      <div className="flex items-center gap-2 mb-1">
                        {msg.role === 'user' ? (
                          <User className="w-4 h-4" />
                        ) : (
                          <Bot className="w-4 h-4" />
                        )}
                        {msg.provider && <Badge variant="outline" className="text-xs capitalize">{msg.provider}</Badge>}
                        {msg.latency && <span className="text-xs text-muted-foreground">{formatLatency(msg.latency)}</span>}
                      </div>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  )}
                </div>
              ))}
              
              {loading && (
                <div className="flex justify-start">
                  <div className="p-3 rounded-lg bg-secondary border border-border">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </CardContent>

            {/* Input */}
            <div className="p-4 border-t border-border">
              <div className="flex gap-2">
                <Input
                  data-testid="chat-input"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about crypto markets, trading strategies..."
                  className="bg-secondary border-border"
                  disabled={loading || availableProviders.length === 0}
                />
                <Button 
                  data-testid="send-btn"
                  onClick={handleSend} 
                  disabled={loading || !input.trim() || availableProviders.length === 0}
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </Button>
              </div>
              {availableProviders.length === 0 && (
                <p className="text-xs text-destructive mt-2">
                  <AlertTriangle className="w-3 h-3 inline mr-1" />
                  No LLM providers configured. Go to Settings → LLM Models to add API keys.
                </p>
              )}
            </div>
          </Card>
        </TabsContent>

        {/* ==================== ENSEMBLE TAB ==================== */}
        <TabsContent value="ensemble" className="flex-1">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
            {/* Signal Request */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <Vote className="w-4 h-4 text-primary" />
                  Ensemble Voting
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Get trading signals from all configured LLMs and see the consensus vote.
                </p>

                <div className="space-y-2">
                  <Label className="text-xs">Symbol</Label>
                  <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                    <SelectTrigger className="bg-secondary border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
                      <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button 
                  data-testid="get-signal-btn"
                  onClick={handleGetSignal} 
                  disabled={signalLoading || availableProviders.length === 0}
                  className="w-full gap-2"
                >
                  {signalLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Getting Signals...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Get Ensemble Signal
                    </>
                  )}
                </Button>

                {availableProviders.length === 0 && (
                  <p className="text-xs text-destructive">
                    <AlertTriangle className="w-3 h-3 inline mr-1" />
                    Configure LLM API keys in Settings first
                  </p>
                )}

                <Separator />

                <div className="text-xs text-muted-foreground">
                  <p className="font-semibold mb-1">How it works:</p>
                  <ol className="list-decimal list-inside space-y-1">
                    <li>Each LLM analyzes current market data</li>
                    <li>They vote: BUY, SELL, or HOLD</li>
                    <li>Majority vote determines the signal</li>
                    <li>Consensus strength shows agreement level</li>
                  </ol>
                </div>
              </CardContent>
            </Card>

            {/* Signal Result */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle className="font-mono text-base flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Signal Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                {ensembleSignal ? (
                  <div className="space-y-4">
                    {/* Main Signal */}
                    <div className={`p-6 rounded-lg text-center ${getSignalColor(ensembleSignal.signal)}`}>
                      {getSignalIcon(ensembleSignal.signal)}
                      <p className="text-3xl font-mono font-bold mt-2">{ensembleSignal.signal}</p>
                      <p className="text-sm mt-1">
                        Consensus: {((ensembleSignal.consensus || 0) * 100).toFixed(0)}%
                      </p>
                    </div>

                    {/* Votes */}
                    {ensembleSignal.votes && (
                      <div className="grid grid-cols-3 gap-2">
                        <div className="p-3 rounded bg-success/20 text-center">
                          <ThumbsUp className="w-4 h-4 mx-auto text-success" />
                          <p className="font-mono text-lg mt-1">{ensembleSignal.votes.BUY || 0}</p>
                          <p className="text-xs text-muted-foreground">BUY</p>
                        </div>
                        <div className="p-3 rounded bg-warning/20 text-center">
                          <Minus className="w-4 h-4 mx-auto text-warning" />
                          <p className="font-mono text-lg mt-1">{ensembleSignal.votes.HOLD || 0}</p>
                          <p className="text-xs text-muted-foreground">HOLD</p>
                        </div>
                        <div className="p-3 rounded bg-destructive/20 text-center">
                          <ThumbsDown className="w-4 h-4 mx-auto text-destructive" />
                          <p className="font-mono text-lg mt-1">{ensembleSignal.votes.SELL || 0}</p>
                          <p className="text-xs text-muted-foreground">SELL</p>
                        </div>
                      </div>
                    )}

                    {/* Individual Results */}
                    {ensembleSignal.results && (
                      <div className="space-y-2">
                        <p className="text-xs text-muted-foreground">Individual Responses:</p>
                        {ensembleSignal.results.map((r, i) => (
                          <div key={i} className="p-2 rounded bg-secondary/30 border border-border">
                            <div className="flex items-center justify-between">
                              <Badge variant="outline" className="capitalize">{r.provider}</Badge>
                              <div className="flex items-center gap-2">
                                <Badge className={getSignalColor(r.trading_signal)}>
                                  {r.trading_signal || "N/A"}
                                </Badge>
                                {r.latency_ms && (
                                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {formatLatency(r.latency_ms)}
                                  </span>
                                )}
                              </div>
                            </div>
                            {r.reasoning && (
                              <p className="text-xs text-muted-foreground mt-1">{r.reasoning}</p>
                            )}
                            {r.error && (
                              <p className="text-xs text-destructive mt-1">{r.error}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    <p className="text-xs text-center text-muted-foreground mt-4">
                      ⚠️ This is not financial advice. Always do your own research.
                    </p>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center">
                    <div className="text-center">
                      <Vote className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">No signal yet</p>
                      <p className="text-xs text-muted-foreground mt-1">Click "Get Ensemble Signal" to start</p>
                    </div>
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

export default Advisor;
