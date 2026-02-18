import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileImage, X, AlertTriangle, CheckCircle, Eye, EyeOff, Save } from 'lucide-react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

const MRIPage = () => {
    const { user } = useAuth();
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [showOverlay, setShowOverlay] = useState(true);
    const [savedToRecords, setSavedToRecords] = useState(false);
    const [saving, setSaving] = useState(false);

    const saveToRecords = async () => {
        if (!user?.email || !result) return;
        setSaving(true);
        try {
            await axios.post('/api/records/save', {
                email: user.email,
                type: 'mri',
                result: result
            });
            setSavedToRecords(true);
        } catch (err) {
            console.error('Error saving record:', err);
        } finally {
            setSaving(false);
        }
    };

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
            setResult(null);
            setError('');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const selected = e.dataTransfer.files[0];
        if (selected && selected.type.startsWith('image/')) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
            setResult(null);
            setError('');
        }
    };

    const handleSubmit = async () => {
        if (!file) {
            console.log("No file selected");
            return;
        }

        console.log("Submitting file:", file.name);
        setLoading(true);
        const formData = new FormData();
        formData.append('image', file);

        try {
            console.log("Sending request to /api/predict/mri");
            const response = await axios.post('/api/predict/mri', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log("Response received:", response.data);
            if (response.data.success) {
                setResult(response.data.result);
                setShowOverlay(true);
            } else {
                console.error("Prediction failed:", response.data.message);
                setError(response.data.message || 'Analysis failed.');
            }
        } catch (err) {
            console.error("Error submitting MRI:", err);
            setError('Analysis failed. Please try another image.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[calc(100vh-8rem)]">
            {/* Upload Section */}
            <div className="flex flex-col space-y-6">
                <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 flex-1 flex flex-col">
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-2xl font-bold text-dark">Upload MRI Scan</h2>
                        {result && result.lesionOverlay && (
                            <button
                                onClick={() => setShowOverlay(!showOverlay)}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${showOverlay
                                        ? 'bg-red-100 text-red-700 hover:bg-red-200'
                                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                            >
                                {showOverlay ? <Eye size={18} /> : <EyeOff size={18} />}
                                {showOverlay ? 'Hide Lesion' : 'Show Lesion'}
                            </button>
                        )}
                    </div>

                    <div
                        className={`flex-1 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 transition-colors ${file ? 'border-primary bg-primary/5' : 'border-gray-200 hover:border-primary hover:bg-gray-50'}`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {preview ? (
                            <div className="relative w-full h-full flex items-center justify-center">
                                <div className="relative">
                                    <img src={preview} alt="MRI Preview" className="max-h-full max-w-full object-contain rounded-lg" style={{ maxHeight: '400px' }} />
                                    {/* Lesion Overlay */}
                                    {result && result.lesionOverlay && showOverlay && (
                                        <motion.img
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ duration: 0.5 }}
                                            src={`data:image/png;base64,${result.lesionOverlay}`}
                                            alt="Lesion Overlay"
                                            className="absolute inset-0 w-full h-full object-contain rounded-lg pointer-events-none"
                                            style={{ mixBlendMode: 'normal' }}
                                        />
                                    )}
                                </div>
                                <button
                                    onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                                    className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-lg hover:bg-red-50 text-red-500"
                                >
                                    <X size={20} />
                                </button>
                                {/* Lesion Legend */}
                                {result && result.lesionOverlay && (
                                    <div className="absolute bottom-2 left-2 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-2 shadow-md">
                                        <div className="flex items-center gap-2 text-sm">
                                            <div className={`w-4 h-4 rounded ${result.hasStroke ? 'bg-red-500/60' : 'bg-orange-400/60'}`}></div>
                                            <span className="font-medium text-gray-700">
                                                {result.hasStroke ? 'Lesion Area' : 'Area of Interest'}
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                                    <Upload size={32} />
                                </div>
                                <h3 className="text-lg font-bold text-dark">Drag & Drop MRI image</h3>
                                <p className="text-textSecondary mt-2 mb-6">or click to browse from your computer</p>

                                <label className="px-6 py-3 bg-white border border-gray-200 rounded-xl font-bold text-dark hover:bg-gray-50 cursor-pointer transition-colors shadow-sm">
                                    Browse Files
                                    <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                                </label>
                                <p className="mt-4 text-xs text-textSecondary uppercase tracking-wide">Supports PNG, JPG, BMP, TIFF</p>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={!file || loading}
                        className="w-full mt-6 py-4 bg-primary text-white rounded-xl font-bold shadow-lg shadow-primary/30 hover:shadow-xl hover:bg-secondary transition-all disabled:opacity-50 disabled:shadow-none"
                    >
                        {loading ? 'Processing Scan...' : 'Analyze MRI'}
                    </button>
                </div>
            </div>

            {/* Results Section */}
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 flex flex-col">
                <h2 className="text-2xl font-bold text-dark mb-6">Analysis Results</h2>

                {error && (
                    <div className="mb-4 p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 flex items-center gap-2">
                        <AlertTriangle size={20} />
                        <span className="font-medium">{error}</span>
                    </div>
                )}

                {result ? (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex-1 flex flex-col items-center justify-center text-center space-y-8"
                    >
                        <div className={`w-24 h-24 rounded-full flex items-center justify-center ${result.hasStroke ? 'bg-red-100 text-red-500' : 'bg-green-100 text-green-500'}`}>
                            {result.hasStroke ? <AlertTriangle size={48} /> : <CheckCircle size={48} />}
                        </div>

                        <div>
                            <h3 className={`text-3xl font-bold ${result.hasStroke ? 'text-red-600' : 'text-green-600'}`}>
                                {result.prediction}
                            </h3>
                            <p className="text-textSecondary mt-2">
                                Confidence: <span className="font-bold text-dark">{(result.confidence * 100).toFixed(1)}%</span>
                            </p>
                        </div>

                        {result.hasStroke && (
                            <div className="w-full max-w-sm bg-red-50 rounded-2xl p-6 border border-red-100">
                                <h4 className="font-bold text-red-800 mb-4">Detailed Findings</h4>

                                {/* Lesion Prediction Info */}
                                {result.lesionOverlay && (
                                    <div className="mb-4 p-3 bg-white/60 rounded-xl border border-red-200">
                                        <div className="flex items-center gap-2 mb-2">
                                            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                                            <span className="font-semibold text-red-800">Lesion Detected</span>
                                        </div>
                                        <p className="text-xs text-red-700">
                                            Red highlighted area on MRI shows predicted lesion location
                                        </p>
                                    </div>
                                )}

                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-red-700">Affected Area</span>
                                    <span className="font-bold text-red-800">{result.affectedArea}%</span>
                                </div>
                                <div className="w-full bg-red-200 rounded-full h-2">
                                    <div
                                        className="bg-red-500 h-2 rounded-full transition-all duration-1000"
                                        style={{ width: `${Math.min(result.affectedArea, 100)}%` }}
                                    />
                                </div>
                                <p className="text-xs text-red-600 mt-4">
                                    Please consult a neurologist immediately for detailed diagnosis.
                                </p>
                            </div>
                        )}

                        {!result.hasStroke && (
                            <div className="w-full max-w-sm bg-green-50 rounded-2xl p-6 border border-green-100">
                                <p className="text-green-700 font-medium">
                                    No stroke patterns detected in this scan. Recommend routine monitoring.
                                </p>
                            </div>
                        )}

                        {/* Save to Records Button */}
                        <button
                            onClick={saveToRecords}
                            disabled={savedToRecords || saving}
                            className={`mt-6 px-6 py-3 rounded-xl font-bold flex items-center gap-2 transition-all ${savedToRecords
                                    ? 'bg-green-100 text-green-600 cursor-default'
                                    : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-600/30'
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
                    </motion.div>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-textSecondary opacity-40">
                        <div className="text-center">
                            <FileImage size={64} className="mx-auto mb-4" />
                            <p>Upload a scan to view analysis</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MRIPage;
