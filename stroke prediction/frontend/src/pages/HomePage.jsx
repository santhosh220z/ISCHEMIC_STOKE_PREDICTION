import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Brain, Users, Clock, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

const StatCard = ({ icon: Icon, value, label, color }) => (
    <motion.div
        whileHover={{ y: -5 }}
        className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 flex flex-col items-center text-center"
    >
        <div className={`p-3 rounded-xl ${color} bg-opacity-10 mb-4`}>
            <Icon className={color.replace('bg-', 'text-')} size={32} />
        </div>
        <h3 className={`text-4xl font-bold ${color.replace('bg-', 'text-')} mb-2`}>{value}</h3>
        <p className="text-textSecondary">{label}</p>
    </motion.div>
);

const HomePage = () => {
    const navigate = useNavigate();



    return (
        <div className="space-y-12">
            {/* Hero Section */}
            <section className="text-center py-12 max-w-4xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-full text-textSecondary text-sm font-medium mb-8"
                >
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    Every Second Counts
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="text-5xl font-bold text-dark mb-6 leading-tight"
                >
                    Your Life is <span className="text-primary">Precious</span><br />
                    Protect Your <span className="text-secondary">Health</span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="text-xl text-textSecondary mb-10 max-w-2xl mx-auto"
                >
                    Advanced AI-powered stroke risk assessment and early detection system.
                    Get accurate predictions in seconds.
                </motion.p>

                <motion.button
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => navigate('/clinical')}
                    className="bg-gradient-to-r from-primary to-secondary text-white px-8 py-4 rounded-xl font-bold text-lg shadow-xl shadow-primary/30 flex items-center gap-2 mx-auto"
                >
                    Get Started <ArrowRight size={20} />
                </motion.button>
            </section>

            {/* Stats Grid */}
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard icon={Activity} value="1" label="Stroke occurs every 40 seconds" color="bg-red-500" />
                <StatCard icon={Brain} value="15M" label="People suffer stroke worldwide/year" color="bg-blue-500" />
                <StatCard icon={Users} value="10%" label="Of strokes occur in people <50" color="bg-amber-500" />
                <StatCard icon={Clock} value="80%" label="Strokes can be prevented" color="bg-green-500" />
            </section>

            {/* FAST Warning Signs */}
            <section className="bg-white rounded-3xl p-8 border border-gray-100 shadow-sm">
                <div className="text-center mb-10">
                    <h2 className="text-2xl font-bold text-dark">Know the Signs - Act FAST</h2>
                    <p className="text-textSecondary mt-2">Recognizing these signs can save a life</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                    {[
                        { letter: 'F', title: 'Face', desc: 'Does one side droop?', color: 'text-red-500' },
                        { letter: 'A', title: 'Arms', desc: 'Can both be raised?', color: 'text-red-500' },
                        { letter: 'S', title: 'Speech', desc: 'Is speech slurred?', color: 'text-red-500' },
                        { letter: 'T', title: 'Time', desc: 'Call emergency now!', color: 'text-red-500' },
                    ].map((item, index) => (
                        <motion.div
                            key={item.letter}
                            whileHover={{ y: -5 }}
                            className="text-center p-6 rounded-2xl bg-gray-50 border border-gray-100"
                        >
                            <div className={`text-5xl font-bold ${item.color} mb-4`}>{item.letter}</div>
                            <h3 className="text-lg font-bold text-dark mb-2">{item.title}</h3>
                            <p className="text-textSecondary text-sm">{item.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </section>
        </div>
    );
};

// Fix Utensils import error by using Activity instead as placeholder or importing Utensils
// I see I defined stats array but used StatCard below directly. I'll correct the component.
// I'll replace Utensils with Activity in the array definition if used, but I used explicit StatCard calls.
// Actually I noticed `icon: Utensils` in `stats` array but I didn't import `Utensils`. 
// I used `Activity` in the explicit render. I should clean up the unused `stats` array or use it.
// I will just remove the stats array and use the explicit render which is correct.

export default HomePage;
