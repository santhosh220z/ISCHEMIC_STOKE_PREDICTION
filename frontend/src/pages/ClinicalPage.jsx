import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Heart, User, Briefcase, Home, Cigarette, Save, CheckCircle } from 'lucide-react';
import axios from 'axios';
import RiskGauge from '../components/RiskGauge';
import { useAuth } from '../context/AuthContext';

const ClinicalPage = () => {
    const { user } = useAuth();
    const [formData, setFormData] = useState({
        age: 50,
        gender: 'Male',
        hypertension: false,
        heartDisease: false,
        everMarried: 'Yes',
        workType: 'Private',
        residenceType: 'Urban',
        avgGlucose: 100.0,
        bmi: 25.0,
        smokingStatus: 'never smoked'
    });

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [savedToRecords, setSavedToRecords] = useState(false);
    const [saving, setSaving] = useState(false);

    const saveToRecords = async () => {
        if (!user?.email || !result) return;
        setSaving(true);
        try {
            await axios.post('/api/records/save', {
                email: user.email,
                type: 'clinical',
                result: result.result
            });
            setSavedToRecords(true);
        } catch (err) {
            console.error('Error saving record:', err);
        } finally {
            setSaving(false);
        }
    };

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        const payload = {
            ...formData,
            age: Number(formData.age),
            avgGlucose: Number(formData.avgGlucose),
            bmi: Number(formData.bmi)
        };

        try {
            const response = await axios.post('/api/predict/clinical', payload);
            if (response.data.success) {
                setResult(response.data);
            }
        } catch (err) {
            setError(err.response?.data?.message || 'Prediction failed. Please check inputs.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Form Section */}
            <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100"
            >
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 bg-primary/10 rounded-xl">
                        <Activity className="text-primary" size={24} />
                    </div>
                    <h2 className="text-2xl font-bold text-dark">Patient Information</h2>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Demographics */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">Age</label>
                            <input
                                type="number"
                                name="age"
                                value={formData.age}
                                onChange={handleChange}
                                className="w-full rounded-xl border-gray-200 focus:border-primary focus:ring-primary"
                                min="0" max="100"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">Gender</label>
                            <select
                                name="gender"
                                value={formData.gender}
                                onChange={handleChange}
                                className="w-full rounded-xl border-gray-200 focus:border-primary focus:ring-primary"
                            >
                                <option>Male</option>
                                <option>Female</option>
                                <option>Other</option>
                            </select>
                        </div>
                    </div>

                    {/* Health Metrics */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">Glucose Lvl</label>
                            <input
                                type="number"
                                name="avgGlucose"
                                value={formData.avgGlucose}
                                onChange={handleChange}
                                step="0.1"
                                className="w-full rounded-xl border-gray-200 focus:border-primary focus:ring-primary"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">BMI</label>
                            <input
                                type="number"
                                name="bmi"
                                value={formData.bmi}
                                onChange={handleChange}
                                step="0.1"
                                className="w-full rounded-xl border-gray-200 focus:border-primary focus:ring-primary"
                            />
                        </div>
                    </div>

                    {/* Conditions as Toggle Cards */}
                    <div className="grid grid-cols-2 gap-4">
                        <div
                            onClick={() => setFormData({ ...formData, hypertension: !formData.hypertension })}
                            className={`p-4 rounded-xl border cursor-pointer transition-all ${formData.hypertension ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-transparent'}`}
                        >
                            <div className="flex items-center gap-2">
                                <input type="checkbox" checked={formData.hypertension} readOnly className="rounded text-red-500 focus:ring-red-500" />
                                <span className={`font-medium ${formData.hypertension ? 'text-red-700' : 'text-gray-600'}`}>Hypertension</span>
                            </div>
                        </div>
                        <div
                            onClick={() => setFormData({ ...formData, heartDisease: !formData.heartDisease })}
                            className={`p-4 rounded-xl border cursor-pointer transition-all ${formData.heartDisease ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-transparent'}`}
                        >
                            <div className="flex items-center gap-2">
                                <input type="checkbox" checked={formData.heartDisease} readOnly className="rounded text-red-500 focus:ring-red-500" />
                                <span className={`font-medium ${formData.heartDisease ? 'text-red-700' : 'text-gray-600'}`}>Heart Disease</span>
                            </div>
                        </div>
                    </div>

                    {/* Social Factors */}
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">Work Type</label>
                            <select name="workType" value={formData.workType} onChange={handleChange} className="w-full rounded-xl border-gray-200">
                                <option value="Private">Private</option>
                                <option value="Self-employed">Self-employed</option>
                                <option value="Govt_job">Government Job</option>
                                <option value="children">Children</option>
                                <option value="Never_worked">Never Worked</option>
                            </select>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-textSecondary mb-1">Residence</label>
                                <select name="residenceType" value={formData.residenceType} onChange={handleChange} className="w-full rounded-xl border-gray-200">
                                    <option value="Urban">Urban</option>
                                    <option value="Rural">Rural</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-textSecondary mb-1">Smoking</label>
                                <select name="smokingStatus" value={formData.smokingStatus} onChange={handleChange} className="w-full rounded-xl border-gray-200">
                                    <option value="never smoked">Never smoked</option>
                                    <option value="formerly smoked">Formerly smoked</option>
                                    <option value="smokes">Smokes</option>
                                    <option value="Unknown">Unknown</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-textSecondary mb-1">Marital Status</label>
                            <select name="everMarried" value={formData.everMarried} onChange={handleChange} className="w-full rounded-xl border-gray-200">
                                <option value="Yes">Married</option>
                                <option value="No">Single</option>
                            </select>
                        </div>
                    </div>

                    {error && <div className="text-red-500 text-sm font-medium">{error}</div>}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-4 bg-primary text-white rounded-xl font-bold shadow-lg shadow-primary/30 hover:shadow-xl hover:bg-secondary transition-all disabled:opacity-50"
                    >
                        {loading ? 'Analyzing...' : 'Analyze Risk'}
                    </button>
                </form>
            </motion.div>

            {/* Results Section */}
            <AnimatePresence>
                {result ? (
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 flex flex-col h-full"
                    >
                        <div className="text-center mb-8">
                            <h2 className="text-2xl font-bold text-dark mb-6">Analysis Results</h2>
                            <RiskGauge
                                percentage={result.result.percentage}
                                color={result.result.color.replace('#', 'bg-')} // Mapping logic needs adjustment or explicit mapping
                                label={result.result.category}
                            />
                            <p className="text-textSecondary mt-4 max-w-md mx-auto">
                                Based on provided clinical data, the patient has a
                                <strong className="text-dark"> {result.result.category}</strong> probability of stroke relative to the dataset baseline.
                            </p>
                        </div>

                        <div className="space-y-4 overflow-y-auto flex-1 pr-2">
                            <h3 className="font-bold text-dark text-lg border-b pb-2">Recommendations</h3>

                            {result.recommendations.lifestyle.map((rec, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="flex items-start gap-3 p-3 rounded-lg bg-gray-50 border border-gray-100"
                                >
                                    <div className="mt-1 w-2 h-2 rounded-full bg-primary flex-shrink-0" />
                                    <p className="text-sm text-textSecondary">{rec.replace('- ', '')}</p>
                                </motion.div>
                            ))}

                            <h3 className="font-bold text-dark text-lg border-b pb-2 pt-4">Medical Advice</h3>
                            {result.recommendations.medical.map((rec, i) => (
                                <motion.div
                                    key={`med-${i}`}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 + i * 0.1 }}
                                    className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 border border-blue-100"
                                >
                                    <div className="mt-1 w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />
                                    <p className="text-sm text-textSecondary">{rec.replace('- ', '')}</p>
                                </motion.div>
                            ))}

                            {/* Save to Records Button */}
                            <button
                                onClick={saveToRecords}
                                disabled={savedToRecords || saving}
                                className={`w-full mt-6 py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
                                    savedToRecords 
                                        ? 'bg-green-100 text-green-600 cursor-default' 
                                        : 'bg-primary text-white hover:bg-secondary shadow-lg shadow-primary/30'
                                }`}
                            >
                                {savedToRecords ? (
                                    <><CheckCircle size={20} /> Saved to Records</>
                                ) : saving ? (
                                    <>Saving...</>
                                ) : (
                                    <><Save size={20} /> Save to Records</>
                                )}
                            </button>
                        </div>
                    </motion.div>
                ) : (
                    <div className="flex items-center justify-center p-8 bg-gray-50 rounded-3xl border border-dashed border-gray-200">
                        <div className="text-center text-textSecondary">
                            <Activity size={48} className="mx-auto mb-4 opacity-20" />
                            <p>Fill in patient details and click Analyze Risk to see results</p>
                        </div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default ClinicalPage;
