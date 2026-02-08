import { TrendingUp, DollarSign, Activity } from "lucide-react";
import { cn } from "../../lib/utils";

interface KpiCardProps {
    label: string;
    value: string | number;
    change?: string;
    positive?: boolean; // true = good, false = bad (for color)
    icon?: React.ReactNode;
}

const KpiCard = ({ label, value, change, positive, icon }: KpiCardProps) => {
    return (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-4 flex flex-col justify-between h-24 relative overflow-hidden group hover:bg-white/[0.04] transition-colors">
            <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                {icon}
            </div>
            <span className="text-xs uppercase tracking-widest text-white/50 font-medium z-10">
                {label}
            </span>
            <div className="flex items-baseline gap-2 z-10">
                <span className="text-2xl font-semibold text-white/90">{value}</span>
                {change && (
                    <span
                        className={cn(
                            "text-xs font-medium",
                            positive ? "text-emerald-400" : "text-rose-400"
                        )}
                    >
                        {change}
                    </span>
                )}
            </div>
        </div>
    );
};

export const KpiStrip = () => {
    // Mock data - would come from props/context
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <KpiCard
                label="Most Exposed Sector"
                value="Automotive"
                icon={<Activity className="w-8 h-8" />}
            />
            <KpiCard
                label="Largest Risk Δ"
                value="+12.4%"
                change="vs Baseline"
                positive={false}
                icon={<TrendingUp className="w-8 h-8" />}
            />
            <KpiCard
                label="Affected Export Value"
                value="$4.2B"
                icon={<DollarSign className="w-8 h-8" />}
            />
            <KpiCard
                label="Active Scenario"
                value="US / 10%"
                change="Steel Focus"
                positive={true} // Neutral
                icon={<Activity className="w-8 h-8" />}
            />
        </div>
    );
};
