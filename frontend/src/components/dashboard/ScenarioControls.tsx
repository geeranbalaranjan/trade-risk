import { Play, RotateCcw } from "lucide-react";

export const ScenarioControls = () => {
    return (
        <div className="flex items-center gap-4 bg-white/[0.02] border border-white/5 rounded-lg p-1.5 pr-2">
            {/* Run Simulation Button */}
            <button className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-md transition-colors shadow-lg shadow-blue-900/20">
                <Play className="w-3 h-3 fill-current" />
                RUN SIMULATION
            </button>

            <div className="h-4 w-px bg-white/10" />

            {/* Preset Selector */}
            <select className="bg-transparent text-xs text-white/80 font-medium focus:outline-none cursor-pointer hover:text-white transition-colors">
                <option>Preset: US Steel Shock 2025</option>
                <option>Preset: EU Border Adjustment</option>
                <option>Preset: Custom Scenario</option>
            </select>

            <div className="h-4 w-px bg-white/10" />

            {/* Params Display (Collapsed) */}
            <div className="flex items-center gap-3 px-2">
                <div className="flex flex-col">
                    <span className="text-[10px] uppercase tracking-wider text-white/40">Tariff</span>
                    <span className="text-xs font-mono text-white/90">10%</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-[10px] uppercase tracking-wider text-white/40">Target</span>
                    <span className="text-xs font-mono text-white/90">US Only</span>
                </div>
            </div>

            <div className="h-4 w-px bg-white/10" />

            <button className="p-1.5 text-white/40 hover:text-white transition-colors" title="Reset">
                <RotateCcw className="w-3.5 h-3.5" />
            </button>
        </div>
    );
};
