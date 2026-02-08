import { KpiStrip } from "../components/dashboard/KpiStrip";
import { RiskLeaderboard } from "../components/dashboard/RiskLeaderboard";
import { PrimaryChart } from "../components/dashboard/PrimaryChart";
import { ExplanationPanel } from "../components/dashboard/ExplanationPanel";
import { ScenarioControls } from "../components/dashboard/ScenarioControls";
import { ShieldAlert } from "lucide-react";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-[#070A12] text-white font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="h-14 border-b border-white/5 flex items-center justify-between px-6 sticky top-0 bg-[#070A12]/80 backdrop-blur-md z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 flex items-center justify-center border border-blue-500/20">
            <ShieldAlert className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-white/90">TariffShock</h1>
            <p className="text-[10px] text-white/40 uppercase tracking-widest font-medium">Risk Simulation Engine</p>
          </div>
        </div>

        <ScenarioControls />
      </header>

      {/* Main Content */}
      <main className="p-6 max-w-[1600px] mx-auto space-y-6">

        <KpiStrip />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Column: Analytics (Wider) */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            <PrimaryChart />
            <RiskLeaderboard />
          </div>

          {/* Right Column: Explanation (Narrower) */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            <ExplanationPanel />

            {/* Additional context or smaller widgets could go here */}
            <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-5">
              <h3 className="text-xs font-semibold text-white/50 uppercase tracking-widest mb-3">Simulation Parameters</h3>
              <div className="space-y-2 text-sm text-white/60">
                <div className="flex justify-between">
                  <span>Base Year</span>
                  <span className="text-white/80 font-mono">2024</span>
                </div>
                <div className="flex justify-between">
                  <span>Model</span>
                  <span className="text-white/80 font-mono">V2.1 (Fixed Weights)</span>
                </div>
                <div className="flex justify-between">
                  <span>Refresh Rate</span>
                  <span className="text-white/80 font-mono">Real-time</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
