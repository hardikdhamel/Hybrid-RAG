import { useState, useRef, useEffect } from 'react';
import { FiUpload, FiSend, FiFile, FiTrash2, FiInfo, FiDatabase, FiCpu, FiSearch, FiChevronDown, FiChevronUp } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';
import { uploadDocument, queryDocuments, getStats, resetSystem, healthCheck } from './api';
import './App.css';

function formatMath(text) {
  if (!text) return text;
  // Convert \[ ... \] to $$...$$ (display math)
  let result = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, p1) => `$$${p1}$$`);
  // Convert \( ... \) to $...$ (inline math)
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, (_, p1) => `$${p1}$`);
  // Convert standalone [ formula ] on its own line to $$formula$$ (display math)
  result = result.replace(/^\s*\[\s*(.+?)\s*\]\s*$/gm, (_, p1) => `$$${p1.trim()}$$`);
  return result;
}

function AssistantMessage({ msg }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  return (
    <div className="message-bubble assistant-bubble">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
      >{formatMath(msg.content)}</ReactMarkdown>
      {msg.sources && msg.sources.length > 0 && (
        <div className="sources-section">
          <button className="sources-toggle" onClick={() => setSourcesOpen(!sourcesOpen)}>
            {sourcesOpen ? <FiChevronUp /> : <FiChevronDown />}
            <span>Sources ({msg.sources.length})</span>
            <div className="retrieval-badges-inline">
              <span className="badge badge-vector">Vec: {msg.retrieval_info?.vector_results}</span>
              <span className="badge badge-bm25">BM25: {msg.retrieval_info?.bm25_results}</span>
              <span className="badge badge-fused">Fused: {msg.retrieval_info?.fused_results}</span>
            </div>
          </button>
          {sourcesOpen && (
            <div className="sources-content">
              {msg.sources.map((src, j) => (
                <div key={j} className="source-card">
                  <div className="source-header">
                    <span className="source-index">[{src.index}]</span>
                    <span className="source-file">{src.source_file}</span>
                    <span className="source-score">RRF: {src.rrf_score}</span>
                  </div>
                  <div className="source-scores">
                    <span className="score-item score-rrf">RRF: {src.rrf_score}</span>
                    <span className="score-item score-vec">Vec: {src.vector_score ?? '—'}</span>
                    <span className="score-item score-bm25">BM25: {src.bm25_score ?? '—'}</span>
                  </div>
                  <div className="source-retrieval">
                    {src.retrieval_sources.map((rs, k) => (
                      <span key={k} className={`badge badge-${rs}`}>{rs}</span>
                    ))}
                  </div>
                  <p className="source-preview">{src.text_preview}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stats, setStats] = useState({ vector_count: 0, bm25_count: 0, ingested_files: [] });
  const [serverOnline, setServerOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const fileInputRef = useRef(null);

  useEffect(() => {
    checkHealth();
    fetchStats();
  }, []);

  const checkHealth = async () => {
    try {
      await healthCheck();
      setServerOnline(true);
    } catch {
      setServerOnline(false);
    }
  };

  const fetchStats = async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch {
      // server might be offline
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadDocument(file, (progress) => {
        setUploadProgress(progress);
      });

      fetchStats();
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setMessages((prev) => [
        ...prev,
        { type: 'error', content: `Upload failed: ${detail}` },
      ]);
    } finally {
      setUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userQuery = query.trim();
    setQuery('');
    setMessages((prev) => [...prev, { type: 'user', content: userQuery }]);
    setLoading(true);

    try {
      const result = await queryDocuments(userQuery);

      setMessages((prev) => [
        ...prev,
        {
          type: 'assistant',
          content: result.answer,
          sources: result.sources,
          retrieval_info: result.retrieval_info,
        },
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      setMessages((prev) => [
        ...prev,
        { type: 'error', content: `Query failed: ${detail}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to clear all indexed data?')) return;
    try {
      await resetSystem();
      setMessages([{ type: 'system', content: 'All data has been cleared.' }]);
      fetchStats();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { type: 'error', content: 'Reset failed.' },
      ]);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2><FiCpu /> Hybrid RAG</h2>
          <span className={`status-dot ${serverOnline ? 'online' : 'offline'}`}></span>
        </div>

        <div className="sidebar-section">
          <h3><FiDatabase /> Knowledge Base</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-value">{stats.vector_count}</span>
              <span className="stat-label">Vector Chunks</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{stats.bm25_count}</span>
              <span className="stat-label">BM25 Chunks</span>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <h3><FiFile /> Uploaded Files</h3>
          <div className="file-list">
            {stats.ingested_files.length === 0 ? (
              <p className="no-files">No files uploaded yet</p>
            ) : (
              stats.ingested_files.map((file, i) => (
                <div key={i} className="file-item">
                  <FiFile /> <span>{file}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-section">
          <h3><FiInfo /> Models</h3>
          <div className="model-info">
            <div className="model-tag">
              <span className="model-label">LLM</span>
              <span className="model-name">gpt-oss:120b-cloud</span>
            </div>
            <div className="model-tag">
              <span className="model-label">Embed</span>
              <span className="model-name">nomic-embed-text</span>
            </div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            <FiUpload /> {uploading ? `Uploading ${uploadProgress}%` : 'Upload Document'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
          <button className="btn btn-danger" onClick={handleReset}>
            <FiTrash2 /> Clear All Data
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-container">
        <div className="chat-header">
          <button className="toggle-sidebar" onClick={() => setSidebarOpen(!sidebarOpen)}>
            ☰
          </button>
          <h1>Hybrid RAG Chat</h1>
          <p>Dense + Sparse Retrieval with Reciprocal Rank Fusion</p>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon"><FiSearch size={48} /></div>
              <h2>Welcome to Hybrid RAG</h2>
              <p>Upload documents (PDF, DOCX, TXT) and ask questions. The system combines <strong>semantic search</strong> (vector embeddings) with <strong>keyword search</strong> (BM25) using Reciprocal Rank Fusion for superior retrieval accuracy.</p>
              <div className="feature-grid">
                <div className="feature-card">
                  <h4>1. Upload</h4>
                  <p>Upload your documents to build the knowledge base</p>
                </div>
                <div className="feature-card">
                  <h4>2. Ask</h4>
                  <p>Ask questions in natural language</p>
                </div>
                <div className="feature-card">
                  <h4>3. Get Answers</h4>
                  <p>Receive accurate, source-cited responses</p>
                </div>
              </div>
              <button className="btn btn-upload welcome-upload" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                <FiUpload /> {uploading ? `Uploading ${uploadProgress}%` : 'Upload Your First Document'}
              </button>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.type}`}>
              {msg.type === 'user' && (
                <div className="message-bubble user-bubble">
                  <p>{msg.content}</p>
                </div>
              )}
              {msg.type === 'assistant' && (
                <AssistantMessage msg={msg} />
              )}
              {msg.type === 'system' && (
                <div className="message-bubble system-bubble">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath, remarkGfm]}
                    rehypePlugins={[rehypeKatex]}
                  >{msg.content}</ReactMarkdown>
                </div>
              )}
              {msg.type === 'error' && (
                <div className="message-bubble error-bubble">
                  <p>{msg.content}</p>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-bubble assistant-bubble">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}

        </div>

        <form className="input-container" onSubmit={handleQuery}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !query.trim()} className="btn btn-send">
            <FiSend />
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;
