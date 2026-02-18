import React from 'react';
import { motion } from 'framer-motion';

const RiskGauge = ({ percentage, color, label }) => {
    // Calculate rotation: 0% -> -90deg, 100% -> 90deg
    const rotation = (percentage / 100) * 180 - 90;

    const getColorHex = (colorClass) => {
        if (colorClass.includes('green')) return '#10B981';
        if (colorClass.includes('yellow')) return '#F59E0B';
        if (colorClass.includes('red')) return '#EF4444';
        return '#2CA58D';
    };

    return (
        <div className="relative w-64 h-32 overflow-hidden mx-auto mb-8">
            {/* Gauge Background */}
            <div className="absolute top-0 left-0 w-full h-64 rounded-full border-[20px] border-gray-100 box-border"></div>

            {/* Gauge Fill */}
            <motion.div
                initial={{ rotate: -90 }}
                animate={{ rotate: rotation }}
                transition={{ duration: 1.5, type: "spring" }}
                className="absolute top-0 left-0 w-full h-64 rounded-full border-[20px] border-transparent box-border origin-center"
                style={{
                    borderTopColor: getColorHex(color),
                    borderRightColor: getColorHex(color),
                    transform: `rotate(${rotation}deg)`
                }}
            ></motion.div>

            {/* Center Text */}
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-center">
                <h2 className={`text-4xl font-bold ${color.replace('bg-', 'text-')}`}>
                    {percentage.toFixed(1)}%
                </h2>
                <p className="text-textSecondary text-sm font-medium uppercase tracking-wider">{label}</p>
            </div>
        </div>
    );
};

export default RiskGauge;
