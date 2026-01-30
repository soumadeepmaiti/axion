import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "sonner";
import { RefreshCw, TrendingUp, TrendingDown, History, Target, Shield } from "lucide-react";
import { getPredictionHistory, makePrediction } from "@/lib/api";

const Predictions = () => {
  const [symbol, setSymbol] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getPredictionHistory(symbol || null, 50);
      setPredictions(result.predictions || []);
    } catch (error) {
      console.error("Error fetching predictions:", error);
      toast.error("Failed to fetch predictions");
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  const stats = {
    total: predictions.length,
    long: predictions.filter(p => p.direction === 1).length,
    short: predictions.filter(p => p.direction === 0).length,
    avgConfidence: predictions.length > 0 
      ? predictions.reduce((acc, p) => acc + (p.confidence || 0), 0) / predictions.length 
      : 0
  };

  return (
    <div data-testid="predictions-page" className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-mono">Prediction History</h1>
          <p className="text-muted-foreground mt-1">View and analyze past predictions</p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger data-testid="filter-symbol-select" className="w-40 bg-card border-border">
              <SelectValue placeholder="All Symbols" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Symbols</SelectItem>
              <SelectItem value="BTC/USDT">BTC/USDT</SelectItem>
              <SelectItem value="ETH/USDT">ETH/USDT</SelectItem>
            </SelectContent>
          </Select>
          <Button
            data-testid="refresh-predictions-btn"
            variant="outline"
            size="icon"
            onClick={fetchPredictions}
            disabled={loading}
            className="border-border hover:border-primary/50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Total</p>
                <p className="text-2xl font-bold font-mono text-foreground mt-1">{stats.total}</p>
              </div>
              <History className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Long</p>
                <p className="text-2xl font-bold font-mono text-success mt-1">{stats.long}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-success/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Short</p>
                <p className="text-2xl font-bold font-mono text-destructive mt-1">{stats.short}</p>
              </div>
              <TrendingDown className="w-8 h-8 text-destructive/50" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Avg Confidence</p>
                <p className="text-2xl font-bold font-mono text-primary mt-1">
                  {(stats.avgConfidence * 100).toFixed(1)}%
                </p>
              </div>
              <Target className="w-8 h-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Predictions Table */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="font-mono text-lg">Recent Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead>Time</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Probability</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Entry Price</TableHead>
                <TableHead>Take Profit</TableHead>
                <TableHead>Stop Loss</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.length > 0 ? (
                predictions.map((pred, index) => (
                  <TableRow key={index} className="border-border hover:bg-secondary/50">
                    <TableCell className="font-mono text-sm">
                      {new Date(pred.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-semibold">{pred.symbol}</TableCell>
                    <TableCell>
                      <Badge className={pred.direction === 1 ? 'bg-success' : 'bg-destructive'}>
                        {pred.direction_label}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono">
                      {(pred.probability * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="font-mono text-primary">
                      {(pred.confidence * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="font-mono">
                      ${pred.current_price?.toLocaleString()}
                    </TableCell>
                    <TableCell className="font-mono text-success">
                      ${pred.take_profit?.toLocaleString()}
                    </TableCell>
                    <TableCell className="font-mono text-destructive">
                      ${pred.stop_loss?.toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-muted-foreground py-8">
                    No predictions yet. Go to Dashboard to generate predictions.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default Predictions;
