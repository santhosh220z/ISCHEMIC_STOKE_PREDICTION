import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { ClipboardList, Trash2, Calendar, FileText, Activity, X } from 'lucide-react';

const RecordsPage = () => {
  const { user } = useAuth();
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // 'all', 'mri', 'clinical'
  const [selectedRecord, setSelectedRecord] = useState(null);

  useEffect(() => {
    if (user?.email) {
      fetchRecords();
    }
  }, [user]);

  const fetchRecords = async () => {
    try {
      const response = await axios.get(`/api/records/${user.email}`);
      if (response.data.success) {
        setRecords(response.data.records);
      }
    } catch (error) {
      console.error('Error fetching records:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteRecord = async (recordId) => {
    if (!window.confirm('Are you sure you want to delete this record?')) return;
    
    try {
      const response = await axios.delete(`/api/records/${user.email}/${recordId}`);
      if (response.data.success) {
        setRecords(records.filter(r => r.id !== recordId));
        if (selectedRecord?.id === recordId) setSelectedRecord(null);
      }
    } catch (error) {
      console.error('Error deleting record:', error);
    }
  };

  const filteredRecords = filter === 'all' 
    ? records 
    : records.filter(r => r.type === filter);

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
        <h2 className="text-2xl font-bold text-dark mb-4">Please log in to view your records</h2>
        <a href="/login" className="text-primary hover:text-secondary underline font-medium">Go to Login</a>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row justify-between items-center gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold text-dark flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl text-primary">
              <ClipboardList size={32} />
            </div>
            My Medical Records
          </h1>
          <p className="text-textSecondary mt-2 ml-14">
            View and manage your past analyses and assessments
          </p>
        </div>

        {/* Filters */}
        <div className="flex bg-gray-100 p-1 rounded-xl">
          {['all', 'mri', 'clinical'].map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                filter === type
                  ? 'bg-white text-primary shadow-sm'
                  : 'text-textSecondary hover:text-dark'
              }`}
            >
              {type === 'all' ? 'All Records' : type === 'mri' ? 'MRI Scans' : 'Clinical'}
            </button>
          ))}
        </div>
      </motion.div>

      {loading ? (
        <div className="text-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mx-auto mb-4"></div>
          <p className="text-textSecondary">Loading records...</p>
        </div>
      ) : filteredRecords.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-20 bg-gray-50 rounded-3xl border border-dashed border-gray-200"
        >
          <div className="p-4 bg-gray-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
            <FileText className="text-gray-400" size={40} />
          </div>
          <h3 className="text-xl font-bold text-dark mb-2">No records found</h3>
          <p className="text-textSecondary">
            Complete an <a href="/mri" className="text-primary hover:underline">MRI analysis</a> or <a href="/clinical" className="text-primary hover:underline">Clinical assessment</a> to see history here.
          </p>
        </motion.div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence>
            {filteredRecords.map((record) => (
              <motion.div
                key={record.id}
                layout
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-all cursor-pointer group"
                onClick={() => setSelectedRecord(record)}
              >
                {/* Type Badge & Date */}
                <div className="flex justify-between items-start mb-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                    record.type === 'mri' 
                      ? 'bg-purple-50 text-purple-600' 
                      : 'bg-blue-50 text-blue-600'
                  }`}>
                    {record.type === 'mri' ? 'MRI Scan' : 'Clinical'}
                  </span>
                  <span className="text-textSecondary text-xs flex items-center gap-1">
                    <Calendar size={12} />
                    {formatDate(record.date)}
                  </span>
                </div>

                {/* Main Result */}
                <div className="mb-4">
                    {record.type === 'mri' ? (
                      <div>
                        <h3 className={`text-lg font-bold mb-1 ${
                          record.result.hasStroke ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {record.result.hasStroke ? 'Stroke Detected' : 'No Stroke Detected'}
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-textSecondary">
                          <Activity size={14} />
                          Confidence: {(record.result.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    ) : (
                      <div>
                        <h3 className={`text-lg font-bold mb-1 ${
                          record.result.risk === 'High' ? 'text-red-600' : 
                          record.result.risk === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                        }`}>
                          {record.result.risk} Risk Level
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-textSecondary">
                          <Activity size={14} />
                          Probability: {record.result.probability}%
                        </div>
                      </div>
                    )}
                </div>

                {/* Footer Action */}
                <div className="pt-4 border-t border-gray-50 flex justify-between items-center">
                    <span className="text-sm text-primary font-medium group-hover:underline">View Details</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteRecord(record.id); }}
                      className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete Record"
                    >
                      <Trash2 size={16} />
                    </button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedRecord && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedRecord(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              className="bg-white rounded-3xl p-8 max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-2xl font-bold text-dark flex items-center gap-3">
                  {selectedRecord.type === 'mri' ? (
                    <div className="p-2 bg-purple-50 rounded-xl text-purple-600">
                        <Activity size={24} />
                    </div>
                  ) : (
                    <div className="p-2 bg-blue-50 rounded-xl text-blue-600">
                        <FileText size={24} />
                    </div>
                  )}
                  {selectedRecord.type === 'mri' ? 'MRI Analysis' : 'Clinical Assessment'}
                </h2>
                <button
                  onClick={() => setSelectedRecord(null)}
                  className="p-2 text-gray-400 hover:text-dark hover:bg-gray-100 rounded-full transition-colors"
                >
                  <X size={24} />
                </button>
              </div>

              <div className="space-y-6">
                {/* Date Badge */}
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-full text-sm text-textSecondary">
                    <Calendar size={14} />
                    {formatDate(selectedRecord.date)}
                </div>

                {/* Main Result Card */}
                <div className={`p-6 rounded-2xl border ${
                    selectedRecord.type === 'mri' 
                        ? (selectedRecord.result.hasStroke ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100')
                        : (selectedRecord.result.risk === 'High' ? 'bg-red-50 border-red-100' : 
                           selectedRecord.result.risk === 'Medium' ? 'bg-yellow-50 border-yellow-100' : 'bg-green-50 border-green-100')
                }`}>
                    <h3 className="text-sm font-bold uppercase tracking-wide mb-1 opacity-70">Primary Result</h3>
                    <p className={`text-2xl font-bold ${
                        selectedRecord.type === 'mri' 
                        ? (selectedRecord.result.hasStroke ? 'text-red-700' : 'text-green-700')
                        : (selectedRecord.result.risk === 'High' ? 'text-red-700' : 
                           selectedRecord.result.risk === 'Medium' ? 'text-yellow-700' : 'text-green-700')
                    }`}>
                        {selectedRecord.type === 'mri' 
                            ? (selectedRecord.result.hasStroke ? 'Stroke Detected' : 'No Stroke Detected')
                            : `${selectedRecord.result.risk} Risk Level`
                        }
                    </p>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                        <span className="text-sm text-textSecondary block mb-1">Confidence</span>
                        <span className="text-lg font-bold text-dark">
                            {selectedRecord.type === 'mri' 
                                ? `${(selectedRecord.result.confidence * 100).toFixed(1)}%`
                                : `${selectedRecord.result.probability}%`
                            }
                        </span>
                    </div>
                    
                    {selectedRecord.type === 'mri' && selectedRecord.result.affectedArea && (
                        <div className="p-4 bg-gray-50 rounded-xl border border-gray-100">
                            <span className="text-sm text-textSecondary block mb-1">Affected Area</span>
                            <span className="text-lg font-bold text-dark">
                                {selectedRecord.result.affectedArea}%
                            </span>
                        </div>
                    )}
                </div>

                <div className="pt-6 border-t border-gray-100">
                    <button
                    onClick={() => { deleteRecord(selectedRecord.id); }}
                    className="w-full py-4 bg-white border border-red-200 text-red-600 rounded-xl hover:bg-red-50 transition-colors font-bold flex items-center justify-center gap-2"
                    >
                    <Trash2 size={20} /> Delete This Record
                    </button>
                </div>
              </div>

            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default RecordsPage;
