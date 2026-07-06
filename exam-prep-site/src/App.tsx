import { BrowserRouter, Routes, Route, NavLink, Navigate, useParams } from 'react-router-dom';
import { chapters, labs } from './data/chapters';
import ChapterPage from './components/ChapterPage';
import LabPage from './components/LabPage';
import { useState } from 'react';

function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-950 flex">
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`
        fixed lg:sticky top-0 left-0 z-50 h-screen w-72
        bg-gray-900 border-r border-gray-800 overflow-y-auto
        transform transition-transform duration-200
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        flex flex-col
      `}>
        <div className="p-5 border-b border-gray-800">
          <h1 className="text-lg font-bold text-white">Deep Learning & CV</h1>
          <p className="text-xs text-gray-500 mt-1">Final Exam Prep Wiki</p>
        </div>

        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-2 mb-2 mt-2">Chapters</p>
          {chapters.map((ch) => (
            <NavLink
              key={ch.id}
              to={`/chapter/${ch.id}`}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive ? 'bg-blue-600/20 text-blue-300 font-medium' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                }`
              }
            >
              <span className="text-xs font-mono text-gray-500 w-6">{ch.id}</span>
              <span className="truncate">{ch.title}</span>
            </NavLink>
          ))}

          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-2 mb-2 mt-6">Labs</p>
          {labs.map((lab) => (
            <NavLink
              key={lab.id}
              to={`/lab/${lab.id}`}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive ? 'bg-green-600/20 text-green-300 font-medium' : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                }`
              }
            >
              <span className="text-xs font-mono text-gray-500 w-6">L{lab.id}</span>
              <span className="truncate">{lab.name}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800 text-xs text-gray-600">SCUT · 2026 Final Exam</div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-30 bg-gray-950/80 backdrop-blur border-b border-gray-800 px-4 py-3 flex items-center gap-3">
          <button className="lg:hidden text-gray-400 hover:text-white p-1" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
          </button>
          <div className="flex-1" />
          <span className="text-xs text-gray-500">Deep Learning & Computer Vision</span>
        </header>

        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/chapter/1" replace />} />
            <Route path="/chapter/:id" element={<ChapterRoute />} />
            <Route path="/lab/:id" element={<LabRoute />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function ChapterRoute() {
  const { id } = useParams<{ id: string }>();
  const chapter = chapters.find((c) => c.id === parseInt(id || '1'));
  if (!chapter) return <Navigate to="/chapter/1" replace />;
  return <ChapterPage chapter={chapter} />;
}

function LabRoute() {
  const { id } = useParams<{ id: string }>();
  const lab = labs.find((l) => l.id === parseInt(id || '1'));
  if (!lab) return <Navigate to="/lab/1" replace />;
  return <LabPage name={lab.name} file={lab.file} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}
