import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Home, FileText, Image as ImageIcon, HelpCircle, LogOut, ClipboardList } from 'lucide-react';

import { useAuth } from '../context/AuthContext';

const Sidebar = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { name: 'Home', icon: Home, path: '/' },
    { name: 'Clinical Data', icon: FileText, path: '/clinical' },
    { name: 'MRI Images', icon: ImageIcon, path: '/mri' },
    { name: 'My Records', icon: ClipboardList, path: '/records' },
    { name: 'Help', icon: HelpCircle, path: '/help' },
  ];

  return (
    <div className="w-64 bg-light border-r border-gray-200 h-screen fixed left-0 top-0 flex flex-col">
      {/* Brand */}
      <div className="p-6 border-b border-gray-200 flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg">
          SS
        </div>
        <div>
          <h1 className="text-lg font-bold text-dark leading-tight">StrokeSense</h1>
          <p className="text-xs text-textSecondary">AI Prediction</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${isActive
                ? 'bg-primary/10 text-primary font-medium shadow-sm'
                : 'text-textSecondary hover:bg-gray-100 hover:text-dark'
              }`
            }
          >
            <item.icon size={20} />
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>

      {/* User & Logout */}
      <div className="p-4 border-t border-gray-200">
        {user ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3 px-2">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">
                {user.fullName ? user.fullName[0].toUpperCase() : 'U'}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-dark truncate">{user.fullName}</p>
                <p className="text-xs text-textSecondary truncate">{user.email}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center gap-3 px-4 py-3 text-textSecondary hover:bg-red-50 hover:text-red-600 rounded-xl transition-colors duration-200"
            >
              <LogOut size={20} />
              <span>Logout</span>
            </button>
          </div>
        ) : (
          <button
            onClick={() => navigate('/login')}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-bold rounded-xl shadow-lg shadow-primary/30 hover:shadow-xl hover:bg-secondary transition-all"
          >
            <span>Sign In</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
