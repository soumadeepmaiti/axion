import "@/App.css";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import Dashboard from "@/pages/Dashboard";
import Training from "@/pages/Training";
import Predictions from "@/pages/Predictions";
import Backtesting from "@/pages/Backtesting";
import Settings from "@/pages/Settings";
import { 
  LayoutDashboard, 
  Brain, 
  TrendingUp, 
  Settings as SettingsIcon,
  Activity,
  Cpu,
  TestTube2
} from "lucide-react";

const Sidebar = () => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", icon: LayoutDashboard, label: "Dashboard" },
    { path: "/training", icon: Brain, label: "Training" },
    { path: "/backtesting", icon: TestTube2, label: "Backtesting" },
    { path: "/predictions", icon: TrendingUp, label: "Predictions" },
    { path: "/settings", icon: SettingsIcon, label: "Settings" },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-card border-r border-border flex flex-col z-50">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center glow-primary">
            <Cpu className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-foreground font-mono tracking-tight">
              OMNI<span className="text-primary">CRYPTO</span>
            </h1>
            <p className="text-xs text-muted-foreground">Hybrid AI Trading</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          
          return (
            <NavLink
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase()}`}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive 
                  ? "bg-primary/10 text-primary border border-primary/30" 
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
              {isActive && (
                <Activity className="w-3 h-3 ml-auto animate-pulse" />
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Status */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
          <span className="text-muted-foreground">System Active</span>
        </div>
      </div>
    </aside>
  );
};

function App() {
  return (
    <div className="min-h-screen bg-background">
      <BrowserRouter>
        <Sidebar />
        <main className="ml-64 min-h-screen">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
        <Toaster position="top-right" theme="dark" />
      </BrowserRouter>
    </div>
  );
}

export default App;
