/**
 * RealTimeDetection.jsx
 *
 * Browser-based real-time road damage detection.
 * Captures frames from webcam or uploaded video → sends to /predict-frame → draws
 * bounding boxes on a canvas overlay positioned exactly over the <video> element.
 *
 * Frame pipeline:
 *   video element  →  hidden capture canvas (resized to 640 px wide)
 *   → toBlob(JPEG 0.75)  →  POST /predict-frame
 *   → detections JSON  →  draw on overlay canvas (RAF loop)
 */

import { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
const API_URL        = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TARGET_FPS     = 3;                           // frames sent per second
const FRAME_INTERVAL = Math.round(1000 / TARGET_FPS); // ms between sends
const CAPTURE_WIDTH  = 640;                         // resize before sending
const JPEG_QUALITY   = 0.75;                        // balance quality vs speed

const CLASS_META = {
  pothole: { color: "#E74C3C", glow: "#E74C3C88", label: "Pothole", emoji: "🕳️" },
  crack:   { color: "#F39C12", glow: "#F39C1288", label: "Crack",   emoji: "⚡" },
  manhole: { color: "#3498DB", glow: "#3498DB88", label: "Manhole", emoji: "🔵" },
};

const SEV_COLOR = {
  LOW:      "#2ECC71",
  MEDIUM:   "#F39C12",
  HIGH:     "#E67E22",
  CRITICAL: "#E74C3C",
};

// ─────────────────────────────────────────────────────────────────────────────
// Drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draw all detections onto the overlay canvas.
 * Coordinates come from the model (frame space) and must be scaled
 * to the canvas display size.
 */
function drawDetections(ctx, detections, frameW, frameH, canvasW, canvasH) {
  ctx.clearRect(0, 0, canvasW, canvasH);
  if (!detections || detections.length === 0) return;

  const scaleX = canvasW / frameW;
  const scaleY = canvasH / frameH;

  detections.forEach((det) => {
    const { x1, y1, x2, y2 } = det.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const bw  = sx2 - sx1;
    const bh  = sy2 - sy1;

    const meta  = CLASS_META[det.class_name] || CLASS_META.pothole;
    const color = meta.color;

    // — Glow / shadow —
    ctx.save();
    ctx.shadowColor = meta.glow;
    ctx.shadowBlur  = 14;

    // — Bounding box —
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(sx1, sy1, bw, bh);
    ctx.restore();

    // — Corner accents —
    const cl = Math.max(10, Math.min(18, bw * 0.15, bh * 0.15));
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3.5;
    [
      [sx1, sy1,  1,  1],
      [sx2, sy1, -1,  1],
      [sx1, sy2,  1, -1],
      [sx2, sy2, -1, -1],
    ].forEach(([cx, cy, dx, dy]) => {
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx * cl, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + dy * cl); ctx.stroke();
    });

    // — Label pill —
    const pct   = Math.round(det.confidence * 100);
    const label = `${det.class_name.toUpperCase()}  ${pct}%`;
    ctx.font = "bold 11px 'JetBrains Mono', 'Courier New', monospace";
    const tw = ctx.measureText(label).width;
    const lh = 20;
    const lx = sx1;
    const ly = sy1 > lh + 6 ? sy1 - lh - 4 : sy1 + 4;

    ctx.fillStyle = color + "ee";
    ctx.beginPath();
    ctx.roundRect?.(lx, ly, tw + 10, lh, 4) ?? ctx.rect(lx, ly, tw + 10, lh);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, lx + 5, ly + 14);

    // — Severity badge (top-right of box) —
    const sevColor = SEV_COLOR[det.severity] || "#fff";
    const sev = det.severity.slice(0, 3);
    ctx.font      = "bold 9px sans-serif";
    ctx.fillStyle = sevColor + "cc";
    ctx.fillText(sev, sx2 - 26, sy1 + 12);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function StatPill({ label, value, color = "#8A9BB0" }) {
  return (
    <div style={{
      background: "#111820",
      border: "1px solid #1E2A38",
      borderRadius: 8,
      padding: "6px 12px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minWidth: 72,
    }}>
      <span style={{ fontSize: 18, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>
        {value}
      </span>
      <span style={{ fontSize: 10, color: "#5A6878", textTransform: "uppercase", letterSpacing: "0.07em", marginTop: 1 }}>
        {label}
      </span>
    </div>
  );
}

function DetectionRow({ det, idx }) {
  const meta = CLASS_META[det.class_name] || CLASS_META.pothole;
  const sevColor = SEV_COLOR[det.severity] || "#fff";
  const pct = Math.round(det.confidence * 100);

  return (
    <div
      key={det.id}
      style={{
        background: "#111820",
        border: `1px solid ${meta.color}22`,
        borderLeft: `3px solid ${meta.color}`,
        borderRadius: 9,
        padding: "9px 12px",
        display: "flex",
        alignItems: "center",
        gap: 10,
        animation: "slideIn 0.15s ease",
      }}
    >
      <span style={{ fontSize: 16 }}>{meta.emoji}</span>
      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ color: meta.color, fontWeight: 700, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            {meta.label}
          </span>
          <span style={{
            background: sevColor + "22",
            border: `1px solid ${sevColor}44`,
            color: sevColor,
            borderRadius: 5,
            padding: "1px 6px",
            fontSize: 10,
            fontWeight: 800,
            letterSpacing: "0.07em",
          }}>
            {det.severity}
          </span>
        </div>
        <div style={{ fontSize: 10, color: "#5A6878", marginTop: 2, fontFamily: "'JetBrains Mono', monospace" }}>
          ({Math.round(det.bbox.x1)},{Math.round(det.bbox.y1)}) {Math.round(det.bbox.width)}×{Math.round(det.bbox.height)}px
        </div>
      </div>
      <span style={{
        background: meta.color + "22",
        border: `1px solid ${meta.color}44`,
        color: meta.color,
        borderRadius: 99,
        padding: "2px 9px",
        fontSize: 11,
        fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {pct}%
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function RealTimeDetection() {
  // ── State ────────────────────────────────────────────────────────────────
  const [mode,       setMode]       = useState("idle");    // idle | webcam | video
  const [status,     setStatus]     = useState("Ready");
  const [detections, setDetections] = useState([]);
  const [frameInfo,  setFrameInfo]  = useState(null);      // {w, h, inf_ms}
  const [conf,       setConf]       = useState(0.25);
  const [fps,        setFps]        = useState(0);
  const [error,      setError]      = useState(null);
  const [sending,    setSending]    = useState(false);

  // ── Refs ─────────────────────────────────────────────────────────────────
  const videoRef       = useRef(null);   // <video> element
  const overlayRef     = useRef(null);   // visible canvas over video
  const captureRef     = useRef(null);   // hidden canvas for frame capture
  const streamRef      = useRef(null);   // MediaStream (webcam)
  const intervalRef    = useRef(null);   // setInterval handle
  const inFlightRef    = useRef(false);  // guard: one request at a time
  const detectionsRef  = useRef([]);     // latest detections for RAF draw
  const frameInfoRef   = useRef(null);   // latest frame dims for scaling
  const rafRef         = useRef(null);   // requestAnimationFrame handle
  const fpsCountRef    = useRef(0);      // raw frame counter for FPS
  const fpsTimerRef    = useRef(null);   // 1-second FPS update timer

  // ── FPS meter ────────────────────────────────────────────────────────────
  const startFpsMeter = useCallback(() => {
    fpsCountRef.current = 0;
    fpsTimerRef.current = setInterval(() => {
      setFps(fpsCountRef.current);
      fpsCountRef.current = 0;
    }, 1000);
  }, []);

  const stopFpsMeter = useCallback(() => {
    clearInterval(fpsTimerRef.current);
    setFps(0);
  }, []);

  // ── Canvas overlay draw loop (RAF — independent of send interval) ────────
  const startDrawLoop = useCallback(() => {
    const draw = () => {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const fi  = frameInfoRef.current;
      if (fi) {
        drawDetections(ctx, detectionsRef.current, fi.w, fi.h, canvas.width, canvas.height);
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
  }, []);

  const stopDrawLoop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    // Clear canvas
    const canvas = overlayRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  // ── Resize overlay canvas to match video display size ────────────────────
  const syncCanvasSize = useCallback(() => {
    const video  = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;
    canvas.width  = video.videoWidth  || video.offsetWidth;
    canvas.height = video.videoHeight || video.offsetHeight;
  }, []);

  // ── Capture one frame and send to backend ─────────────────────────────────
  const captureAndSend = useCallback(async () => {
    if (inFlightRef.current) return;      // skip if previous request still pending
    const video  = videoRef.current;
    const canvas = captureRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    // ── Resize to CAPTURE_WIDTH keeping aspect ratio ──
    const scale  = CAPTURE_WIDTH / vw;
    const cw     = CAPTURE_WIDTH;
    const ch     = Math.round(vh * scale);
    canvas.width  = cw;
    canvas.height = ch;

    const ctx2d = canvas.getContext("2d");
    ctx2d.drawImage(video, 0, 0, cw, ch);

    // ── Convert to JPEG blob ──────────────────────────────
    canvas.toBlob(
      async (blob) => {
        if (!blob) return;
        inFlightRef.current = true;
        setSending(true);

        const fd = new FormData();
        fd.append("frame", blob, "frame.jpg");

        try {
          const res = await fetch(
            `${API_URL}/predict-frame?conf=${conf}&iou=0.45`,
            { method: "POST", body: fd }
          );
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
          }
          const data = await res.json();

          // Update detections refs (used by RAF draw loop)
          detectionsRef.current = data.detections;
          frameInfoRef.current  = { w: data.frame_width, h: data.frame_height, inf_ms: data.inference_ms };

          // Update React state for sidebar (batched)
          setDetections(data.detections);
          setFrameInfo({ w: data.frame_width, h: data.frame_height, inf_ms: data.inference_ms });
          setError(null);
          fpsCountRef.current += 1;
        } catch (e) {
          setError(e.message);
        } finally {
          inFlightRef.current = false;
          setSending(false);
        }
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }, [conf]);

  // ── Start interval that calls captureAndSend ──────────────────────────────
  const startSendLoop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(captureAndSend, FRAME_INTERVAL);
  }, [captureAndSend]);

  const stopSendLoop = useCallback(() => {
    clearInterval(intervalRef.current);
    intervalRef.current = null;
  }, []);

  // ── When conf changes while running, restart the send loop ───────────────
  useEffect(() => {
    if (mode !== "idle") {
      stopSendLoop();
      startSendLoop();
    }
  }, [conf]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── WEBCAM ────────────────────────────────────────────────────────────────
  const startWebcam = useCallback(async () => {
    setError(null);
    setStatus("Requesting camera…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;
      video.muted     = true;
      await video.play();

      setMode("webcam");
      setStatus("Webcam active");
      syncCanvasSize();
      startDrawLoop();
      startSendLoop();
      startFpsMeter();
    } catch (e) {
      setError(`Camera error: ${e.message}`);
      setStatus("Error");
    }
  }, [syncCanvasSize, startDrawLoop, startSendLoop, startFpsMeter]);

  // ── VIDEO FILE ────────────────────────────────────────────────────────────
  const handleVideoFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError("Please select a video file (MP4, WebM, MOV…)");
      return;
    }
    setError(null);

    const video = videoRef.current;
    const url   = URL.createObjectURL(file);
    video.srcObject = null;
    video.src       = url;
    video.muted     = true;
    video.loop      = true;

    video.onloadedmetadata = () => {
      video.play();
      setMode("video");
      setStatus(`Playing: ${file.name}`);
      syncCanvasSize();
      startDrawLoop();
      startSendLoop();
      startFpsMeter();
    };
  }, [syncCanvasSize, startDrawLoop, startSendLoop, startFpsMeter]);

  // ── STOP ──────────────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    stopSendLoop();
    stopDrawLoop();
    stopFpsMeter();

    // Release webcam stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      if (video.src) {
        URL.revokeObjectURL(video.src);
        video.src = "";
      }
    }

    detectionsRef.current = [];
    frameInfoRef.current  = null;
    setDetections([]);
    setFrameInfo(null);
    setMode("idle");
    setStatus("Stopped");
    inFlightRef.current = false;
  }, [stopSendLoop, stopDrawLoop, stopFpsMeter]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopSendLoop();
      stopDrawLoop();
      stopFpsMeter();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Sync canvas size whenever video dimensions change ─────────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const handler = () => syncCanvasSize();
    video.addEventListener("resize",        handler);
    video.addEventListener("loadedmetadata", handler);
    return () => {
      video.removeEventListener("resize",        handler);
      video.removeEventListener("loadedmetadata", handler);
    };
  }, [syncCanvasSize]);

  // ── Derived values ────────────────────────────────────────────────────────
  const isRunning    = mode !== "idle";
  const totalDets    = detections.length;
  const classCounts  = detections.reduce((acc, d) => {
    acc[d.class_name] = (acc[d.class_name] || 0) + 1;
    return acc;
  }, {});
  const worstSev     = detections.reduce((worst, d) => {
    const order = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
    return (order[d.severity] ?? 0) > (order[worst] ?? 0) ? d.severity : worst;
  }, "LOW");
  const sevColor     = SEV_COLOR[worstSev] || "#8A9BB0";

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: "flex",
      gap: 20,
      padding: "24px 24px 32px",
      maxWidth: 1200,
      margin: "0 auto",
      flexWrap: "wrap",
    }}>

      <style>{`
        @keyframes slideIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
        @keyframes blink   { 0%,100% { opacity:1; } 50% { opacity:.3; } }
        @keyframes spin    { to { transform:rotate(360deg); } }
      `}</style>

      {/* ── LEFT: Video + controls ─────────────────────────────────────── */}
      <div style={{ flex: "1 1 560px", display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Video container */}
        <div style={{
          position: "relative",
          background: "#060A0E",
          borderRadius: 16,
          overflow: "hidden",
          border: isRunning ? `1px solid ${sevColor}44` : "1px solid #1A2130",
          boxShadow: isRunning ? `0 0 24px ${sevColor}18` : "none",
          transition: "border-color .4s, box-shadow .4s",
          aspectRatio: "16/9",
          minHeight: 280,
        }}>
          {/* Idle placeholder */}
          {!isRunning && (
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              gap: 12, color: "#3A4A5C",
            }}>
              <div style={{ fontSize: 52 }}>📷</div>
              <div style={{ fontSize: 14, fontWeight: 600 }}>Camera preview</div>
              <div style={{ fontSize: 12 }}>Start webcam or upload a video below</div>
            </div>
          )}

          {/* Video element — always in DOM so refs are stable */}
          <video
            ref={videoRef}
            playsInline
            style={{
              width: "100%", height: "100%",
              objectFit: "contain",
              display: isRunning ? "block" : "none",
            }}
          />

          {/* Canvas overlay — drawn over the video */}
          <canvas
            ref={overlayRef}
            style={{
              position: "absolute",
              top: 0, left: 0,
              width: "100%", height: "100%",
              pointerEvents: "none",
              display: isRunning ? "block" : "none",
            }}
          />

          {/* Hidden capture canvas */}
          <canvas ref={captureRef} style={{ display: "none" }} />

          {/* Live badge */}
          {isRunning && (
            <div style={{
              position: "absolute", top: 10, left: 10,
              display: "flex", alignItems: "center", gap: 6,
              background: "#0A0D12cc",
              border: "1px solid #E74C3C44",
              borderRadius: 7,
              padding: "5px 10px",
              fontSize: 11,
              backdropFilter: "blur(6px)",
            }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#E74C3C", display: "inline-block", animation: "blink 1.2s infinite" }} />
              <span style={{ color: "#E74C3C", fontWeight: 700 }}>LIVE</span>
              <span style={{ color: "#5A6878" }}>·</span>
              <span style={{ color: "#8A9BB0", fontFamily: "'JetBrains Mono',monospace" }}>{fps} fps</span>
              {sending && (
                <>
                  <span style={{ color: "#5A6878" }}>·</span>
                  <span style={{ width: 10, height: 10, border: "2px solid #3A4A5C", borderTop: "2px solid #3498DB", borderRadius: "50%", display: "inline-block", animation: "spin .7s linear infinite" }} />
                </>
              )}
            </div>
          )}

          {/* Detection count overlay */}
          {isRunning && (
            <div style={{
              position: "absolute", top: 10, right: 10,
              background: "#0A0D12cc",
              border: `1px solid ${sevColor}44`,
              borderRadius: 7,
              padding: "5px 10px",
              fontSize: 11,
              backdropFilter: "blur(6px)",
              color: sevColor,
              fontWeight: 700,
              fontFamily: "'JetBrains Mono',monospace",
            }}>
              {totalDets} det.
            </div>
          )}

          {/* Status bar */}
          {isRunning && frameInfo && (
            <div style={{
              position: "absolute", bottom: 0, left: 0, right: 0,
              background: "linear-gradient(transparent, #0A0D12cc)",
              padding: "10px 14px 8px",
              display: "flex", gap: 12, alignItems: "center",
              fontSize: 10,
              color: "#5A6878",
              fontFamily: "'JetBrains Mono',monospace",
            }}>
              <span>Frame: {frameInfo.w}×{frameInfo.h}</span>
              <span>·</span>
              <span>Inf: {frameInfo.inf_ms.toFixed(0)}ms</span>
              <span>·</span>
              <span>Conf ≥ {conf.toFixed(2)}</span>
              <span>·</span>
              <span style={{ color: mode === "webcam" ? "#3498DB" : "#F39C12" }}>
                {mode === "webcam" ? "WEBCAM" : "VIDEO"}
              </span>
            </div>
          )}
        </div>

        {/* Controls */}
        <div style={{
          background: "#0E1420",
          border: "1px solid #1A2130",
          borderRadius: 14,
          padding: 18,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}>

          {/* Button row */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>

            {/* Start Webcam */}
            <button
              onClick={startWebcam}
              disabled={isRunning}
              style={{
                flex: "1 1 140px",
                background: !isRunning ? "linear-gradient(135deg, #1A6B9A, #3498DB)" : "#111820",
                border: "none", borderRadius: 10,
                color: !isRunning ? "#fff" : "#3A4A5C",
                padding: "11px 16px",
                fontWeight: 700, fontSize: 13,
                cursor: !isRunning ? "pointer" : "not-allowed",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                transition: "all .2s",
              }}
            >
              <span>📷</span> Start Webcam
            </button>

            {/* Video file upload */}
            <label style={{
              flex: "1 1 140px",
              background: !isRunning ? "#1A2130" : "#111820",
              border: `1px solid ${!isRunning ? "#2A3848" : "#1A2130"}`,
              borderRadius: 10,
              color: !isRunning ? "#8A9BB0" : "#3A4A5C",
              padding: "11px 16px",
              fontWeight: 700, fontSize: 13,
              cursor: !isRunning ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
              userSelect: "none",
            }}>
              <span>🎬</span> Upload Video
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoFile}
                disabled={isRunning}
                style={{ display: "none" }}
              />
            </label>

            {/* Stop */}
            <button
              onClick={stop}
              disabled={!isRunning}
              style={{
                flex: "0 0 auto",
                background: isRunning ? "#2D1412" : "#111820",
                border: `1px solid ${isRunning ? "#E74C3C44" : "#1A2130"}`,
                borderRadius: 10,
                color: isRunning ? "#E74C3C" : "#3A4A5C",
                padding: "11px 20px",
                fontWeight: 700, fontSize: 13,
                cursor: isRunning ? "pointer" : "not-allowed",
                transition: "all .2s",
              }}
            >
              ■ Stop
            </button>
          </div>

          {/* Confidence slider */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#8A9BB0" }}>
                Confidence Threshold
              </span>
              <span style={{
                fontFamily: "'JetBrains Mono',monospace",
                fontSize: 12, fontWeight: 700, color: "#3498DB",
              }}>
                {conf.toFixed(2)}
              </span>
            </div>
            <input
              type="range" min="5" max="95" step="5"
              value={Math.round(conf * 100)}
              onChange={(e) => setConf(parseInt(e.target.value, 10) / 100)}
              style={{ width: "100%", accentColor: "#3498DB", cursor: "pointer" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 10, color: "#3A4A5C" }}>
              <span>More sensitive (0.05)</span>
              <span>More strict (0.95)</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              background: "#2D1412", border: "1px solid #E74C3C44",
              borderRadius: 9, padding: "9px 13px",
              color: "#E74C3C", fontSize: 12,
              display: "flex", gap: 8,
            }}>
              <span>⚠</span> {error}
            </div>
          )}

          {/* Status */}
          <div style={{ fontSize: 11, color: "#3A4A5C", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: isRunning ? "#2ECC71" : "#3A4A5C",
              display: "inline-block",
              animation: isRunning ? "blink 2s infinite" : "none",
            }} />
            {status}
          </div>
        </div>
      </div>

      {/* ── RIGHT: Live detection feed ──────────────────────────────────── */}
      <div style={{ flex: "1 1 280px", display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Stats row */}
        <div style={{ display: "flex", gap: 8 }}>
          <StatPill label="Detections" value={totalDets} color={totalDets > 0 ? sevColor : "#8A9BB0"} />
          <StatPill label="FPS sent"   value={fps}       color={fps >= 2 ? "#2ECC71" : fps > 0 ? "#F39C12" : "#8A9BB0"} />
          <StatPill label="Severity"   value={isRunning && totalDets > 0 ? worstSev.slice(0,3) : "—"} color={sevColor} />
        </div>

        {/* Class summary */}
        <div style={{
          background: "#0E1420", border: "1px solid #1A2130",
          borderRadius: 12, padding: "12px 16px",
        }}>
          <div style={{ fontSize: 10, color: "#5A6878", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700, marginBottom: 10 }}>
            Class Counts
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {Object.entries(CLASS_META).map(([cls, meta]) => (
              <div key={cls} style={{
                flex: 1,
                background: (classCounts[cls] || 0) > 0 ? `${meta.color}11` : "#111820",
                border: `1px solid ${(classCounts[cls] || 0) > 0 ? meta.color + "33" : "#1A2130"}`,
                borderRadius: 8, padding: "7px 8px", textAlign: "center",
                transition: "all .3s",
              }}>
                <div style={{ fontSize: 16 }}>{meta.emoji}</div>
                <div style={{
                  fontSize: 16, fontWeight: 700,
                  color: (classCounts[cls] || 0) > 0 ? meta.color : "#3A4A5C",
                  fontFamily: "'JetBrains Mono',monospace",
                }}>
                  {classCounts[cls] || 0}
                </div>
                <div style={{ fontSize: 9, color: "#5A6878", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {meta.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Detection list */}
        <div style={{
          background: "#0E1420", border: "1px solid #1A2130",
          borderRadius: 12, padding: 14, flex: 1,
        }}>
          <div style={{ fontSize: 10, color: "#5A6878", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700, marginBottom: 10 }}>
            Live Detections
            {totalDets > 0 && (
              <span style={{
                marginLeft: 6, background: "#3498DB22",
                border: "1px solid #3498DB44", color: "#3498DB",
                borderRadius: 99, padding: "1px 7px", fontSize: 10,
              }}>
                {totalDets}
              </span>
            )}
          </div>

          <div style={{
            display: "flex", flexDirection: "column", gap: 7,
            maxHeight: 360, overflowY: "auto", paddingRight: 2,
          }}>
            {!isRunning && (
              <div style={{ color: "#3A4A5C", fontSize: 12, textAlign: "center", padding: "24px 0" }}>
                Start webcam or video to see live detections
              </div>
            )}
            {isRunning && detections.length === 0 && (
              <div style={{ color: "#3A4A5C", fontSize: 12, textAlign: "center", padding: "24px 0" }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>✅</div>
                No damage detected in frame
              </div>
            )}
            {detections.map((det, i) => (
              <DetectionRow key={`${det.id}-${i}`} det={det} idx={i} />
            ))}
          </div>
        </div>

        {/* How-to hint */}
        {!isRunning && (
          <div style={{
            background: "#0E1420", border: "1px solid #1A2130",
            borderRadius: 12, padding: "12px 16px",
          }}>
            <div style={{ fontSize: 10, color: "#5A6878", textTransform: "uppercase", letterSpacing: "0.07em", fontWeight: 700, marginBottom: 8 }}>
              Quick guide
            </div>
            {[
              ["📷", "Click Start Webcam for live camera feed"],
              ["🎬", "Or upload an MP4 / WebM / MOV file"],
              ["⚡", `Frames sent at ${TARGET_FPS} fps to /predict-frame`],
              ["🎯", "Adjust threshold to tune sensitivity"],
              ["■", "Click Stop to release camera / video"],
            ].map(([icon, text]) => (
              <div key={text} style={{ display: "flex", gap: 8, fontSize: 11, color: "#6A7A8C", marginBottom: 5 }}>
                <span>{icon}</span><span>{text}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
