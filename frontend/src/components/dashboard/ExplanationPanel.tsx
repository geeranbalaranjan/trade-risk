import { Info, Sparkles } from "lucide-react";

export const ExplanationPanel = () => {
    return (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl flex flex-col h-full min-h-[400px]">
            <div className="px-5 py-4 border-b border-white/5 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <h3 className="text-sm font-semibold text-white/90">AI Risk Analysis</h3>
            </div>

            <div className="p-5 flex-1 flex flex-col gap-6">
                {/* Main Insight */}
                <div className="space-y-2">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20">
                            High Impact
                        </span>
                        <span className="text-xs text-white/40">Automotive Mfg.</span>
                    </div>
                    <p className="text-sm text-white/80 leading-relaxed">
                        Risk increased primarily due to heavy reliance on US exports (82% of sector volume) and existing low margins.
                        The 10% tariff shock directly impacts the sector's profitability threshold, raising the composite risk score by 12 points.
                    </p>
                </div>

                {/* Drivers Breakdown */}
                <div className="space-y-3">
                    <h4 className="text-xs font-medium text-white/50 uppercase tracking-widest">Key Drivers</h4>

                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-white/70 mb-1">
                            <span>Export Exposure</span>
                            <span className="font-mono">High</span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div className="h-full bg-rose-500/60 w-[85%] rounded-full" />
                        </div>
                    </div>

                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-white/70 mb-1">
                            <span>Supply Chain Concentration</span>
                            <span className="font-mono">Med</span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div className="h-full bg-amber-500/60 w-[60%] rounded-full" />
                        </div>
                    </div>
                </div>

                {/* Simulation Context */}
                <div className="mt-auto pt-4 border-t border-white/5">
                    <div className="flex items-start gap-2 text-xs text-white/50">
                        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <p>Analysis based on 2024 trade data and current simulation parameters. Confidence: High.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};
