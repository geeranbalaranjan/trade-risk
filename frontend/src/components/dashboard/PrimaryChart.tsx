import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";

const data = [
    { name: "Auto", baseline: 60, shocked: 85 },
    { name: "Steel", baseline: 55, shocked: 78 },
    { name: "Lumber", baseline: 65, shocked: 62 },
    { name: "Oil", baseline: 40, shocked: 45 },
    { name: "Agri", baseline: 30, shocked: 30 },
    { name: "Tech", baseline: 20, shocked: 22 },
];

export const PrimaryChart = () => {
    return (
        <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-5 flex flex-col h-[300px]">
            <h3 className="text-sm font-semibold text-white/90 mb-4">Risk Distribution Delta</h3>

            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} barGap={4}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                        <XAxis
                            dataKey="name"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
                            dy={10}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
                        />
                        <Tooltip
                            cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                            contentStyle={{
                                backgroundColor: '#0A0A0A',
                                borderColor: 'rgba(255,255,255,0.1)',
                                borderRadius: '8px',
                                fontSize: '12px',
                                color: '#fff'
                            }}
                        />
                        <Bar dataKey="baseline" fill="rgba(255,255,255,0.1)" radius={[4, 4, 0, 0]} name="Baseline Risk" />
                        <Bar dataKey="shocked" fill="rgba(244, 63, 94, 0.8)" radius={[4, 4, 0, 0]} name="Shocked Risk" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
