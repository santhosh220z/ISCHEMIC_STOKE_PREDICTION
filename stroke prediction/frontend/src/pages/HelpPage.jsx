import React from 'react';
import { motion } from 'framer-motion';
import { HelpCircle, AlertTriangle, Activity, Brain, Shield } from 'lucide-react';

const HelpPage = () => {
    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div className="flex items-center gap-3 mb-8">
                <div className="p-3 bg-primary/10 rounded-xl">
                    <HelpCircle className="text-primary" size={32} />
                </div>
                <div>
                    <h1 className="text-3xl font-bold text-dark">Help & Information</h1>
                    <p className="text-textSecondary">Guide to using StrokeSense</p>
                </div>
            </div>

            {/* About Section */}
            <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100"
            >
                <h2 className="text-xl font-bold text-dark mb-4 flex items-center gap-2">
                    <Shield className="text-primary" size={24} />
                    About StrokeSense
                </h2>
                <p className="text-textSecondary leading-relaxed mb-6">
                    StrokeSense is an AI-powered healthcare application designed to help assess stroke risk
                    and detect potential stroke patterns in brain MRI images. It combines a Random Forest
                    classifier for clinical data and a Convolutional Neural Network (CNN) for image analysis.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                        <h3 className="font-bold text-dark mb-2 flex items-center gap-2">
                            <Activity size={20} className="text-blue-500" /> Clinical Assessment
                        </h3>
                        <p className="text-sm text-textSecondary">
                            Predicts stroke risk based on patient demographics and health metrics like BMI, glucose levels, and heart disease history.
                        </p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                        <h3 className="font-bold text-dark mb-2 flex items-center gap-2">
                            <Brain size={20} className="text-purple-500" /> MRI Analysis
                        </h3>
                        <p className="text-sm text-textSecondary">
                            Analyzes brain MRI scans to detect stroke patterns and segment lesion areas using deep learning models.
                        </p>
                    </div>
                </div>
            </motion.section>

            {/* FAST Signs */}
            <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-red-50 p-8 rounded-3xl border border-red-100"
            >
                <div className="flex items-start gap-4 mb-6">
                    <AlertTriangle className="text-red-500 flex-shrink-0" size={32} />
                    <div>
                        <h2 className="text-xl font-bold text-red-900">Emergency Signs (FAST)</h2>
                        <p className="text-red-700 mt-1">If you observe these signs, call emergency services immediately.</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-red-100 text-center">
                        <div className="text-4xl font-bold text-red-500 mb-2">F</div>
                        <div className="font-bold text-dark">Face</div>
                        <p className="text-xs text-textSecondary">Does one side droop?</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-red-100 text-center">
                        <div className="text-4xl font-bold text-red-500 mb-2">A</div>
                        <div className="font-bold text-dark">Arms</div>
                        <p className="text-xs text-textSecondary">Does one arm drift?</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-red-100 text-center">
                        <div className="text-4xl font-bold text-red-500 mb-2">S</div>
                        <div className="font-bold text-dark">Speech</div>
                        <p className="text-xs text-textSecondary">Is speech slurred?</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-red-100 text-center">
                        <div className="text-4xl font-bold text-red-500 mb-2">T</div>
                        <div className="font-bold text-dark">Time</div>
                        <p className="text-xs text-textSecondary">Call emergency now!</p>
                    </div>
                </div>
            </motion.section>

            {/* Disclaimer */}
            <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-center text-sm text-textSecondary max-w-2xl mx-auto"
            >
                <p className="font-medium text-dark mb-2">Important Disclaimer</p>
                <p>
                    This tool is for educational and informational purposes only. It is not a replacement for professional medical diagnosis.
                    Always consult healthcare professionals for medical decisions.
                </p>
            </motion.section>
        </div>
    );
};

export default HelpPage;
