import { useState, useRef, useEffect } from 'react';
import { FiUpload, FiSend, FiFile, FiTrash2, FiInfo, FiDatabase, FiCpu, FiSearch, FiChevronDown, FiChevronUp, FiCheck, FiX, FiLoader } from 'react-icons/fi';
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
  const [stats, setStats] = useState({ vector_count: 0, bm25_count: 0, ingested_files: [] });
  const [serverOnline, setServerOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedFile, setExpandedFile] = useState(null);

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

    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const uploadMsgId = Date.now();

    setUploading(true);

    // Add inline progress message to chat
    setMessages((prev) => [
      ...prev,
      {
        type: 'upload-progress',
        id: uploadMsgId,
        filename: file.name,
        fileSize: fileSizeMB,
        status: 'uploading',
        progress: 0,
      },
    ]);

    const updateUploadMsg = (updates) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === uploadMsgId ? { ...msg, ...updates } : msg
        )
      );
    };

    try {
      const result = await uploadDocument(file, (progress) => {
        updateUploadMsg({ progress, status: progress >= 100 ? 'processing' : 'uploading' });
      });

      // Replace progress message with success message
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === uploadMsgId
            ? {
              type: 'upload-success',
              id: uploadMsgId,
              filename: result.filename,
              vectorChunks: result.vector_chunks,
              bm25Chunks: result.bm25_chunks,
              skippedChunks: result.skipped,
            }
            : msg
        )
      );
      fetchStats();
    } catch (err) {
      const detail = err.response?.data?.detail || err.message;
      // Replace progress message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === uploadMsgId
            ? { type: 'error', id: uploadMsgId, content: `Upload failed: ${detail}` }
            : msg
        )
      );
    } finally {
      setUploading(false);
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
      setMessages([]);
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
              stats.ingested_files.map((file, i) => {
                const isObject = typeof file === 'object' && file !== null;
                const fileName = isObject ? file.name : file;
                const vectorChunks = isObject ? (file.vector_chunks !== undefined ? file.vector_chunks : file.chunks) : null;
                const bm25Chunks = isObject ? (file.bm25_chunks !== undefined ? file.bm25_chunks : file.chunks) : null;
                const skippedChunks = isObject ? (file.skipped || 0) : 0;
                const isExpanded = expandedFile === fileName;
                const hasStats = vectorChunks !== null;

                return (
                  <div key={i} className="file-item-group">
                    <div
                      className={`file-item ${isExpanded ? 'expanded' : ''} ${hasStats ? 'clickable' : ''}`}
                      onClick={() => hasStats && setExpandedFile(isExpanded ? null : fileName)}
                    >
                      <div className="file-item-main">
                        <FiFile />
                        <span className="file-name" title={fileName}>{fileName}</span>
                      </div>
                      {hasStats && (
                        <span className="file-toggle-icon">{isExpanded ? '▼' : '▶'}</span>
                      )}
                    </div>
                    {isExpanded && hasStats && (
                      <div className="file-details">
                        <div className="file-stat">
                          <span className="file-stat-label">Vector Chunks:</span>
                          <span className="file-stat-val">{vectorChunks}/{vectorChunks + skippedChunks}</span>
                        </div>
                        <div className="file-stat">
                          <span className="file-stat-label">BM25 Chunks:</span>
                          <span className="file-stat-val">{bm25Chunks}/{bm25Chunks + skippedChunks}</span>
                        </div>

                      </div>
                    )}
                  </div>
                );
              })
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
            <FiUpload /> {uploading ? 'Processing...' : 'Upload Document'}
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
                <FiUpload /> {uploading ? 'Processing...' : 'Upload Your First Document'}
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
              {msg.type === 'clear-success' && (
                <div className="message-bubble clear-success-card">
                  <div className="clear-success-icon-wrap">
                    <FiTrash2 />
                  </div>
                  <div className="clear-success-info">
                    <strong>All data has been cleared successfully.</strong>
                    <span className="clear-success-desc">All uploaded files, vector embeddings, and BM25 indexes have been removed.</span>
                  </div>
                  <button
                    className="btn btn-upload clear-success-upload-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    <FiUpload /> {uploading ? 'Processing...' : 'Upload a New Document'}
                  </button>
                </div>
              )}
              {msg.type === 'upload-progress' && (
                <div className="message-bubble upload-inline-card">
                  <div className="upload-inline-header">
                    <FiFile className="upload-inline-file-icon" />
                    <div className="upload-inline-info">
                      <span className="upload-inline-filename">{msg.filename}</span>
                      <span className="upload-inline-size">{msg.fileSize} MB</span>
                    </div>
                  </div>
                  <div className="upload-inline-progress">
                    <div className="upload-inline-bar-bg">
                      <div
                        className={`upload-inline-bar-fill ${msg.status === 'processing' ? 'processing' : ''}`}
                        style={{ width: msg.status === 'processing' ? '100%' : `${msg.progress}%` }}
                      />
                    </div>
                  </div>
                  <div className="upload-inline-status">
                    <span className="upload-pulse-dot" />
                    <span>{msg.status === 'processing'
                      ? 'Processing — extracting text, generating embeddings & indexing...'
                      : `Uploading to server... ${msg.progress}%`}
                    </span>
                  </div>
                </div>
              )}
              {msg.type === 'upload-success' && (
                <div className="message-bubble upload-success-card">
                  <div className="upload-success-icon-wrap">
                    <FiCheck />
                  </div>
                  <div className="upload-success-info">
                    <strong>{msg.filename}</strong> uploaded successfully
                    <div className="upload-success-stats">
                      <span className="upload-success-chunks">Vector: {msg.vectorChunks || 0}/{((msg.vectorChunks || 0) + (msg.skippedChunks || 0))}</span>
                      <span className="upload-success-chunks">BM25: {msg.bm25Chunks || 0}/{((msg.bm25Chunks || 0) + (msg.skippedChunks || 0))}</span>
                    </div>
                  </div>
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
