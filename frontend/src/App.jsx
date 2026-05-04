import { useState, useRef, useCallback, useEffect } from "react";
import RealTimeDetection from "./pages/RealTimeDetection";

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CLASS_META = {
  pothole: { emoji: "🕳️", color: "#E74C3C", bg: "#2D1A19", label: "Pothole" },
  crack:   { emoji: "💥", color: "#F39C12", bg: "#2D2419", label: "Crack" },
  manhole: { emoji: "🔵", color: "#3498DB", bg: "#192028", label: "Manhole" },
};

const SEVERITY_META = {
  LOW:      { color: "#2ECC71", bg: "#192D22", bar: 25,  label: "Low" },
  MEDIUM:   { color: "#F39C12", bg: "#2D2419", bar: 50,  label: "Medium" },
  HIGH:     { color: "#E67E22", bg: "#2D1E10", bar: 75,  label: "High" },
  CRITICAL: { color: "#E74C3C", bg: "#2D1412", bar: 100, label: "Critical" },
};

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
function formatMs(ms) {
  return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function ConfidencePill({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 75 ? "#2ECC71" : pct >= 50 ? "#F39C12" : "#E74C3C";
  return (
    <span
      style={{
        background: `${color}22`,
        border: `1px solid ${color}66`,
        color,
        borderRadius: 999,
        padding: "2px 10px",
        fontSize: 12,
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 700,
        letterSpacing: "0.04em",
      }}
    >
      {pct}%
    </span>
  );
}

function SeverityBadge({ level }) {
  const meta = SEVERITY_META[level] || SEVERITY_META.LOW;
  return (
    <span
      style={{
        background: meta.bg,
        border: `1px solid ${meta.color}55`,
        color: meta.color,
        borderRadius: 6,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 800,
        letterSpacing: "0.1em",
        textTransform: "uppercase",
      }}
    >
      {meta.label}
    </span>
  );
}

// ─────────────────────────────────────────────
// Canvas overlay for bounding boxes
// ─────────────────────────────────────────────
function DetectionCanvas({ imageUrl, detections, imgDims }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !detections || !imgDims) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const { displayW, displayH, origW, origH } = imgDims;

    canvas.width = displayW;
    canvas.height = displayH;
    ctx.clearRect(0, 0, displayW, displayH);

    const scaleX = displayW / origW;
    const scaleY = displayH / origH;

    detections.forEach((det) => {
      const { x1, y1, x2, y2 } = det.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
      const sx2 = x2 * scaleX, sy2 = y2 * scaleY;
      const bw = sx2 - sx1, bh = sy2 - sy1;

      const meta = CLASS_META[det.class_name] || CLASS_META.pothole;
      const color = meta.color;

      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(sx1, sy1, bw, bh);
      ctx.shadowBlur = 0;

      // Corner accents
      const cl = Math.min(14, bw / 4, bh / 4);
      ctx.lineWidth = 3.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      [[sx1,sy1,1,1],[sx2,sy1,-1,1],[sx1,sy2,1,-1],[sx2,sy2,-1,-1]].forEach(([cx,cy,dx,dy]) => {
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx * cl, cy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + dy * cl); ctx.stroke();
      });
      ctx.shadowBlur = 0;

      // Label background
      const label = `${det.class_name.toUpperCase()} ${Math.round(det.confidence * 100)}%`;
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      const tm = ctx.measureText(label);
      const lw = tm.width + 10, lh = 20;
      const lx = sx1, ly = sy1 > lh + 4 ? sy1 - lh - 4 : sy1 + 4;

      ctx.fillStyle = color + "ee";
      ctx.beginPath();
      ctx.roundRect(lx, ly, lw, lh, 4);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillText(label, lx + 5, ly + 14);
    });
  }, [detections, imgDims]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0, left: 0,
        width: "100%", height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}

// ─────────────────────────────────────────────
// Detection Card
// ─────────────────────────────────────────────
function DetectionCard({ det, index }) {
  const meta = CLASS_META[det.class_name] || CLASS_META.pothole;
  return (
    <div
      style={{
        background: "#111418",
        border: `1px solid ${meta.color}33`,
        borderLeft: `3px solid ${meta.color}`,
        borderRadius: 10,
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        animation: `fadeSlide 0.3s ease ${index * 0.05}s both`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 20 }}>{meta.emoji}</span>
          <span style={{ color: meta.color, fontWeight: 700, fontSize: 14, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            {meta.label}
          </span>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <ConfidencePill value={det.confidence} />
          <SeverityBadge level={det.severity} />
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px", fontSize: 12, color: "#8A9BB0" }}>
        <span>Box: ({det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}) → ({det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)})</span>
        <span>Size: {det.bbox.width.toFixed(0)}×{det.bbox.height.toFixed(0)}px</span>
        <span>Area: {(det.area_ratio * 100).toFixed(2)}% of image</span>
        <span>ID: #{det.id}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Stats Bar
// ─────────────────────────────────────────────
function StatsBar({ label, value, max = 1, color }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#8A9BB0" }}>
        <span>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{typeof value === "number" && value < 1 ? `${(value * 100).toFixed(1)}%` : value}</span>
      </div>
      <div style={{ height: 6, background: "#1E2530", borderRadius: 99 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 99, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────
export default function App() {
  const [dragOver, setDragOver] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [imgDims, setImgDims] = useState(null);
  const [apiStats, setApiStats] = useState(null);

  // ── Page routing (no external router needed) ──
  const [page, setPage] = useState("home"); // "home" | "realtime"
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  // Fetch API stats on mount
  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then((r) => r.json())
      .then(setApiStats)
      .catch(() => {});
  }, [result]);

  const processFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    setImageFile(file);
    setResult(null);
    setError(null);
    setShowAnnotated(false);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    processFile(file);
  }, [processFile]);

  const handleFileChange = (e) => processFile(e.target.files[0]);

  const handleImageLoad = () => {
    if (!imageRef.current) return;
    setImgDims({
      displayW: imageRef.current.offsetWidth,
      displayH: imageRef.current.offsetHeight,
      origW: result?.image_width || imageRef.current.naturalWidth,
      origH: result?.image_height || imageRef.current.naturalHeight,
    });
  };

  const runDetection = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", imageFile);
      const resp = await fetch(
        `${API_URL}/predict?conf=${confThreshold}&iou=0.45&include_annotated_image=false`,
        { method: "POST", body: fd }
      );
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Detection failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    setImgDims(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const summaryMeta = result?.summary
    ? SEVERITY_META[result.summary.overall_severity] || SEVERITY_META.LOW
    : null;

  return (
    <div style={{ minHeight: "100vh", background: "#0A0D12", color: "#E8EDF3", fontFamily: "'Inter', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0A0D12; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #111418; }
        ::-webkit-scrollbar-thumb { background: #2A3441; border-radius: 3px; }
        @keyframes fadeSlide { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header style={{
        borderBottom: "1px solid #1A2130",
        padding: "16px 32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        background: "#0A0D12ee",
        backdropFilter: "blur(10px)",
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: "linear-gradient(135deg, #E74C3C22, #3498DB22)",
            border: "1px solid #3498DB44",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20,
          }}>🚗</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.01em" }}>
              RoadScan<span style={{ color: "#3498DB" }}>.ai</span>
            </div>
            <div style={{ fontSize: 11, color: "#5A6878", marginTop: 1 }}>
              Road Surface Damage Detection System
            </div>
          </div>
        </div>

        {/* ── Nav tabs ──────────────────────────────────────────────────── */}
        <nav style={{ display: "flex", gap: 4, background: "#111820", borderRadius: 10, padding: 4, border: "1px solid #1A2130" }}>
          {[
            { id: "home",     label: "📸 Image Detection" },
            { id: "realtime", label: "📷 Real-Time" },
          ].map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{
                background: page === id ? "#1E2D40" : "transparent",
                border: page === id ? "1px solid #3498DB44" : "1px solid transparent",
                borderRadius: 7,
                color: page === id ? "#E8EDF3" : "#5A6878",
                padding: "7px 16px",
                fontSize: 12,
                fontWeight: page === id ? 700 : 400,
                cursor: "pointer",
                transition: "all .2s",
                fontFamily: "'Inter', sans-serif",
                whiteSpace: "nowrap",
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {apiStats && (
            <div style={{ fontSize: 12, color: "#5A6878", textAlign: "right" }}>
              <div>{apiStats.total_inferences} inferences</div>
              <div>{apiStats.avg_inference_ms.toFixed(0)}ms avg</div>
            </div>
          )}
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            background: "#111418", border: "1px solid #1A2130",
            borderRadius: 8, padding: "6px 12px", fontSize: 12,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: 99, background: "#2ECC71", display: "inline-block", animation: "pulse 2s infinite" }} />
            <span style={{ color: "#2ECC71", fontWeight: 600 }}>YOLOv8s Live</span>
          </div>
        </div>
      </header>

      {/* ── Page content ───────────────────────────────────────────────── */}
      {page === "realtime" ? (
        <RealTimeDetection />
      ) : (
        <main style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px", display: "flex", gap: 24, flexWrap: "wrap" }}>

        {/* LEFT PANEL */}
        <div style={{ flex: "1 1 500px", display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Upload Zone */}
          <div
            onClick={() => !imagePreview && fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${dragOver ? "#3498DB" : imagePreview ? "#1A2130" : "#2A3441"}`,
              borderRadius: 16,
              background: dragOver ? "#192028" : "#0E1219",
              transition: "all 0.2s ease",
              cursor: imagePreview ? "default" : "pointer",
              overflow: "hidden",
              position: "relative",
              minHeight: imagePreview ? "auto" : 260,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {!imagePreview ? (
              <div style={{ textAlign: "center", padding: 40 }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>📸</div>
                <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>Drop road image here</div>
                <div style={{ color: "#5A6878", fontSize: 13, marginBottom: 16 }}>
                  JPG, PNG, WEBP — Max 10MB
                </div>
                <button style={{
                  background: "#3498DB22",
                  border: "1px solid #3498DB55",
                  color: "#3498DB",
                  borderRadius: 8,
                  padding: "8px 20px",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: 13,
                }}>
                  Browse Files
                </button>
              </div>
            ) : (
              <div style={{ width: "100%", position: "relative" }}>
                <img
                  ref={imageRef}
                  src={imagePreview}
                  onLoad={handleImageLoad}
                  alt="Road"
                  style={{ width: "100%", display: "block", borderRadius: 14 }}
                />
                {result && imgDims && (
                  <DetectionCanvas
                    imageUrl={imagePreview}
                    detections={result.detections}
                    imgDims={{
                      ...imgDims,
                      origW: result.image_width,
                      origH: result.image_height,
                    }}
                  />
                )}
                {loading && (
                  <div style={{
                    position: "absolute", inset: 0,
                    background: "#0A0D12cc",
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    gap: 12, borderRadius: 14,
                  }}>
                    <div style={{
                      width: 44, height: 44,
                      border: "3px solid #1A2130",
                      borderTop: "3px solid #3498DB",
                      borderRadius: "50%",
                      animation: "spin 0.8s linear infinite",
                    }} />
                    <div style={{ color: "#3498DB", fontWeight: 600, fontSize: 14 }}>
                      Analyzing road surface...
                    </div>
                  </div>
                )}
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
          </div>

          {/* Controls */}
          <div style={{
            background: "#0E1219",
            border: "1px solid #1A2130",
            borderRadius: 14,
            padding: 20,
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}>
            {/* Confidence Slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <label style={{ fontSize: 13, fontWeight: 600, color: "#8A9BB0" }}>
                  Confidence Threshold
                </label>
                <span style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 13,
                  fontWeight: 700,
                  color: "#3498DB",
                }}>
                  {confThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range" min="0.05" max="0.95" step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#3498DB", cursor: "pointer" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 11, color: "#3A4A5C" }}>
                <span>Sensitive (0.05)</span>
                <span>Strict (0.95)</span>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 10 }}>
              <button
                onClick={runDetection}
                disabled={!imageFile || loading}
                style={{
                  flex: 1,
                  background: imageFile && !loading ? "linear-gradient(135deg, #2471A3, #3498DB)" : "#1A2130",
                  border: "none",
                  borderRadius: 10,
                  color: imageFile && !loading ? "#fff" : "#3A4A5C",
                  padding: "12px 0",
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: imageFile && !loading ? "pointer" : "not-allowed",
                  transition: "all 0.2s",
                  letterSpacing: "0.02em",
                }}
              >
                {loading ? "Analyzing…" : "🔍 Detect Damage"}
              </button>
              {imagePreview && (
                <button
                  onClick={handleReset}
                  style={{
                    background: "#111418",
                    border: "1px solid #2A3441",
                    borderRadius: 10,
                    color: "#8A9BB0",
                    padding: "12px 18px",
                    fontWeight: 600,
                    fontSize: 13,
                    cursor: "pointer",
                  }}
                >
                  ✕ Reset
                </button>
              )}
            </div>

            {error && (
              <div style={{
                background: "#2D1412",
                border: "1px solid #E74C3C44",
                borderRadius: 10,
                padding: "10px 14px",
                color: "#E74C3C",
                fontSize: 13,
                display: "flex", gap: 8, alignItems: "flex-start",
              }}>
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Class legend */}
          <div style={{
            background: "#0E1219",
            border: "1px solid #1A2130",
            borderRadius: 14,
            padding: "14px 20px",
          }}>
            <div style={{ fontSize: 11, color: "#5A6878", marginBottom: 12, letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 700 }}>
              Detection Classes
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {Object.entries(CLASS_META).map(([key, meta]) => (
                <div key={key} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  background: `${meta.color}11`,
                  border: `1px solid ${meta.color}33`,
                  borderRadius: 8,
                  padding: "6px 12px",
                  flex: 1,
                }}>
                  <span>{meta.emoji}</span>
                  <span style={{ color: meta.color, fontSize: 13, fontWeight: 600 }}>{meta.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={{ flex: "1 1 340px", display: "flex", flexDirection: "column", gap: 20 }}>

          {!result && !loading && (
            <div style={{
              background: "#0E1219",
              border: "1px solid #1A2130",
              borderRadius: 16,
              padding: 32,
              textAlign: "center",
              color: "#3A4A5C",
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
            }}>
              <div style={{ fontSize: 48 }}>🛣️</div>
              <div style={{ fontWeight: 600, fontSize: 15 }}>No Analysis Yet</div>
              <div style={{ fontSize: 13 }}>Upload a road image and click<br />"Detect Damage" to begin.</div>
            </div>
          )}

          {result && (
            <>
              {/* Summary Card */}
              <div style={{
                background: `linear-gradient(135deg, ${summaryMeta.bg}, #0E1219)`,
                border: `1px solid ${summaryMeta.color}44`,
                borderRadius: 16,
                padding: 20,
                animation: "fadeSlide 0.4s ease",
              }}>
                <div style={{ fontSize: 11, color: "#5A6878", letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 700, marginBottom: 14 }}>
                  Analysis Summary
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                  <div>
                    <div style={{ fontSize: 32, fontWeight: 800, color: summaryMeta.color }}>
                      {result.summary.total_detections}
                    </div>
                    <div style={{ color: "#8A9BB0", fontSize: 13 }}>Defects Detected</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{
                      background: summaryMeta.bg,
                      border: `1px solid ${summaryMeta.color}66`,
                      color: summaryMeta.color,
                      borderRadius: 10,
                      padding: "6px 14px",
                      fontWeight: 800,
                      fontSize: 15,
                      letterSpacing: "0.06em",
                    }}>
                      {result.summary.overall_severity}
                    </div>
                    <div style={{ color: "#5A6878", fontSize: 11, marginTop: 4 }}>Overall Severity</div>
                  </div>
                </div>

                {/* Class breakdown */}
                {Object.entries(result.summary.class_counts).map(([cls, cnt]) => {
                  const meta = CLASS_META[cls] || {};
                  return (
                    <div key={cls} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, padding: "6px 10px", background: `${meta.color}11`, borderRadius: 8 }}>
                      <span style={{ color: meta.color, fontSize: 13, fontWeight: 600 }}>
                        {meta.emoji} {meta.label}
                      </span>
                      <span style={{ color: meta.color, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
                        {cnt}
                      </span>
                    </div>
                  );
                })}

                {/* Stats */}
                <div style={{ borderTop: "1px solid #1A2130", marginTop: 14, paddingTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                  <StatsBar
                    label="Affected Surface Area"
                    value={result.summary.affected_area_ratio}
                    max={0.3}
                    color={summaryMeta.color}
                  />
                  <StatsBar
                    label="Severity Score"
                    value={result.summary.overall_severity_score}
                    max={1}
                    color={summaryMeta.color}
                  />
                </div>

                {/* Metadata */}
                <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
                  {[
                    ["⏱️", formatMs(result.inference_time_ms)],
                    ["📐", `${result.image_width}×${result.image_height}`],
                    ["🎯", `Conf ≥ ${result.model_confidence_threshold}`],
                    ["🔑", `#${result.request_id}`],
                  ].map(([icon, val]) => (
                    <div key={val} style={{
                      background: "#111418",
                      border: "1px solid #1A2130",
                      borderRadius: 7,
                      padding: "4px 10px",
                      fontSize: 11,
                      color: "#8A9BB0",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      {icon} {val}
                    </div>
                  ))}
                </div>
              </div>

              {/* Detection List */}
              {result.detections.length > 0 && (
                <div style={{
                  background: "#0E1219",
                  border: "1px solid #1A2130",
                  borderRadius: 16,
                  padding: 20,
                }}>
                  <div style={{ fontSize: 11, color: "#5A6878", letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 700, marginBottom: 14 }}>
                    Individual Detections ({result.detections.length})
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 400, overflowY: "auto", paddingRight: 4 }}>
                    {result.detections.map((det, i) => (
                      <DetectionCard key={det.id} det={det} index={i} />
                    ))}
                  </div>
                </div>
              )}

              {result.detections.length === 0 && (
                <div style={{
                  background: "#0E1219",
                  border: "1px solid #2ECC7133",
                  borderRadius: 16,
                  padding: 24,
                  textAlign: "center",
                }}>
                  <div style={{ fontSize: 36, marginBottom: 8 }}>✅</div>
                  <div style={{ color: "#2ECC71", fontWeight: 700, fontSize: 15, marginBottom: 6 }}>
                    No Damage Detected
                  </div>
                  <div style={{ color: "#5A6878", fontSize: 13 }}>
                    Road surface appears clear at confidence ≥ {confThreshold.toFixed(2)}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
      )}

      {/* Footer */}
      <footer style={{
        borderTop: "1px solid #111418",
        padding: "16px 32px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        fontSize: 12,
        color: "#3A4A5C",
      }}>
        <span>RoadScan.ai · YOLOv8 Road Surface Detection</span>
        <span>Backend: <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{API_URL}</span></span>
      </footer>
    </div>
  );
}
