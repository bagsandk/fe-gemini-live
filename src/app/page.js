"use client";

import { useRef, useState } from "react";
import { GoogleGenAI, Modality } from "@google/genai";

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState([]);
  const [history, setHistory] = useState([]);

  // Refs
  const sessionRef = useRef(null);
  const sessionIdRef = useRef(null);

  const capCtxRef = useRef(null); // AudioContext capture
  const playCtxRef = useRef(null); // AudioContext playback
  const streamRef = useRef(null);
  const mediaNodeRef = useRef(null);
  const workletNodeRef = useRef(null);

  const queueRef = useRef([]); // Float32 playback queue
  const playingRef = useRef(false);

  const isOpenRef = useRef(false);
  const connectingRef = useRef(false);
  const sentSinceLastTurnRef = useRef(false);
  const lastChunkAtRef = useRef(0);
  const turnTimerRef = useRef(null);

  // ===== Transkrip per turn =====
  // live = string kumulatif terakhir yang terlihat dari server
  // agg  = gabungan ('append delta') untuk disimpan sebagai final
  const userLiveRef = useRef("");
  const userAggRef = useRef("");
  const modelLiveRef = useRef("");
  const modelAggRef = useRef("");

  const appendLog = (s) => setLogs((p) => [...p, s]);

  // ==== utils ====
  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

  // Tambah delta: ambil perbedaan dari versi sebelumnya & append ke penampung
  function appendDelta(liveRef, aggRef, nextText) {
    const next = norm(nextText);
    const prev = liveRef.current || "";
    if (!prev) {
      liveRef.current = next;
      aggRef.current += next;
      return;
    }
    if (next.startsWith(prev)) {
      // kasus normal: kumulatif makin panjang
      aggRef.current += next.slice(prev.length);
    } else if (prev.startsWith(next)) {
      // ASR “rewind” (lebih pendek) -> jangan append
    } else {
      // fallback: cari prefix yang sama, lalu append sisanya
      let i = 0;
      const m = Math.min(prev.length, next.length);
      while (i < m && prev[i] === next[i]) i++;
      const add = next.slice(i);
      if (add) aggRef.current += add;
    }
    liveRef.current = next;
  }

  // History: upsert baris terakhir untuk live view
  function upsertHistory(who, text) {
    const t = new Date().toLocaleTimeString();
    setHistory((h) => {
      const hh = h.slice();
      const last = hh[hh.length - 1];
      if (last && last.who === who) {
        hh[hh.length - 1] = { who, text, t };
        return hh;
      }
      hh.push({ who, text, t });
      return hh;
    });
  }

  function int16ToFloat32(int16buf) {
    const int16 = new Int16Array(int16buf);
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
    return f32;
  }
  function b64ToUint8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }
  function int16ToBase64(int16) {
    const u8 = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
    let bin = "";
    const CHUNK = 0x8000;
    for (let i = 0; i < u8.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK));
    }
    return btoa(bin);
  }

  // === playback ===
  function pumpAudio() {
    if (!playCtxRef.current || playingRef.current) return;
    playingRef.current = true;
    const ctx = playCtxRef.current;

    (async () => {
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {}
      }
      while (queueRef.current.length) {
        const f32 = queueRef.current.shift();
        const buf = ctx.createBuffer(1, f32.length, 24000);
        buf.copyToChannel(f32, 0, 0);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        const done = new Promise((r) => (src.onended = r));
        src.start(0);
        await done;
      }
      playingRef.current = false;
    })();
  }

  // auto commit turn kalau diam
  function ensureTurnCommit() {
    clearInterval(turnTimerRef.current);
    const SILENCE_MS = 1800;
    turnTimerRef.current = setInterval(() => {
      if (!isOpenRef.current) return;
      const now = Date.now();
      if (
        sentSinceLastTurnRef.current &&
        now - lastChunkAtRef.current > SILENCE_MS
      ) {
        sessionRef.current?.sendRealtimeInput?.({ turnComplete: true });
        sentSinceLastTurnRef.current = false;
        appendLog("↪️ turnComplete");
      }
    }, 300);
  }

  // === POST transcript ke BE ===
  function getApiBase() {
    try {
      return new URL(process.env.NEXT_PUBLIC_TOKEN_ENDPOINT).origin;
    } catch {
      return "http://localhost:8000";
    }
  }
  async function saveTranscript(role, text, turnId) {
    if (!text) return;
    try {
      const res = await fetch(`${getApiBase()}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current || "no-session-id",
          role,
          text,
          turn_id: turnId || null,
        }),
      });
      if (!res.ok)
        appendLog(`POST /messages ${res.status}: ${await res.text()}`);
      else appendLog(`saved ${role} transcript (${text.length} chars)`);
    } catch (e) {
      appendLog(`saveTranscript error: ${e?.message || e}`);
    }
  }

  // >>> konsisten model token & sesi (half-cascade)
  async function fetchToken() {
    const url = process.env.NEXT_PUBLIC_TOKEN_ENDPOINT;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "gemini-live-2.5-flash-preview" }),
    });
    if (!res.ok)
      throw new Error(`POST /token ${res.status}: ${await res.text()}`);
    const j = await res.json();
    if (!j.token) throw new Error("No token from backend");
    sessionIdRef.current = j.session_id || "no-session-id";
    return j.token;
  }

  async function start() {
    if (connectingRef.current || connected) return;
    connectingRef.current = true;

    try {
      // Capture ctx
      if (!capCtxRef.current) {
        capCtxRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();
      } else if (capCtxRef.current.state === "suspended") {
        await capCtxRef.current.resume();
      }
      appendLog(`CaptureContext SR: ${capCtxRef.current.sampleRate} Hz`);

      // Playback ctx
      if (!playCtxRef.current) {
        try {
          playCtxRef.current = new (window.AudioContext ||
            window.webkitAudioContext)({ sampleRate: 24000 });
        } catch {
          playCtxRef.current = new (window.AudioContext ||
            window.webkitAudioContext)();
        }
      } else if (playCtxRef.current.state === "suspended") {
        await playCtxRef.current.resume();
      }
      appendLog(`PlaybackContext SR: ${playCtxRef.current.sampleRate} Hz`);

      // mic
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      streamRef.current = stream;
      const media = new MediaStreamAudioSourceNode(capCtxRef.current, {
        mediaStream: stream,
      });
      mediaNodeRef.current = media;

      // token & connect
      const apiKey = await fetchToken();
      const ai = new GoogleGenAI({ apiKey, apiVersion: "v1alpha" });

      const model = "gemini-live-2.5-flash-preview";
      const config = {
        responseModalities: [Modality.AUDIO],
        inputAudioTranscription: {}, // aktifkan transkrip user
        outputAudioTranscription: {}, // aktifkan transkrip AI
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
        },
        systemInstruction:
          "Kamu asisten ramah. Jawab singkat dalam bahasa yang sama dengan pengguna.",
      };

      const live = await ai.live.connect({
        model,
        config,
        callbacks: {
          onopen: () => {
            appendLog("WS opened");
            isOpenRef.current = true;
            setConnected(true);
            connectingRef.current = false;
            ensureTurnCommit();
            // reset buffer turn
            userLiveRef.current = "";
            userAggRef.current = "";
            modelLiveRef.current = "";
            modelAggRef.current = "";
          },
          onmessage: async (msg) => {
            // AUDIO out
            if (msg.data) {
              let ab = null;
              if (msg.data instanceof ArrayBuffer) ab = msg.data;
              else if (msg.data instanceof Blob)
                ab = await msg.data.arrayBuffer();
              else if (msg.data?.buffer instanceof ArrayBuffer)
                ab = msg.data.buffer;
              if (ab) {
                if (playCtxRef.current?.state === "suspended") {
                  try {
                    await playCtxRef.current.resume();
                  } catch {}
                }
                queueRef.current.push(int16ToFloat32(ab));
                pumpAudio();
              }
            }

            // TRANSKRIP: pakai transcription saja untuk live,
            // tapi simpan ke penampung via append-delta supaya final "full".
            const sc = msg.serverContent || {};
            const turnId = sc?.modelTurn?.turnId;

            if (sc.inputTranscription?.text) {
              appendDelta(userLiveRef, userAggRef, sc.inputTranscription.text);
              upsertHistory("You", userLiveRef.current);
            }
            if (sc.outputTranscription?.text) {
              appendDelta(
                modelLiveRef,
                modelAggRef,
                sc.outputTranscription.text
              );
              upsertHistory("Gemini", modelLiveRef.current);
            }

            // (opsional) audio inline via parts
            for (const p of sc.modelTurn?.parts || []) {
              const inl = p?.inlineData;
              if (
                inl?.data &&
                typeof inl.data === "string" &&
                (inl?.mimeType || "").startsWith("audio/")
              ) {
                const u8 = b64ToUint8(inl.data);
                queueRef.current.push(int16ToFloat32(u8.buffer));
                pumpAudio();
              }
            }

            // FINALIZE per turn: kirim gabungan 'agg'.
            if (sc.turnComplete || sc?.modelTurn?.turnComplete) {
              // kalau ada teks lengkap di parts saat turnComplete, pilih yang lebih panjang
              const modelParts = (sc?.modelTurn?.parts || [])
                .map((p) => p?.text || "")
                .filter(Boolean)
                .join(" ");
              const userParts = (sc?.userTurn?.parts || [])
                .map((p) => p?.text || "")
                .filter(Boolean)
                .join(" ");

              let finalUser = norm(
                (userAggRef.current || "").length >= userParts.length
                  ? userAggRef.current
                  : userParts
              );
              let finalModel = norm(
                (modelAggRef.current || "").length >= modelParts.length
                  ? modelAggRef.current
                  : modelParts
              );

              if (!finalUser) finalUser = norm(userLiveRef.current);
              if (!finalModel) finalModel = norm(modelLiveRef.current);

              await saveTranscript("user", finalUser, turnId);
              await saveTranscript("model", finalModel, turnId);

              // reset buffers turn
              userLiveRef.current = "";
              userAggRef.current = "";
              modelLiveRef.current = "";
              modelAggRef.current = "";
            }
          },
          onerror: (e) =>
            appendLog("WS error: " + (e?.message || JSON.stringify(e))),
          onclose: (e) => {
            appendLog(`WS closed: code=${e?.code} reason=${e?.reason || ""}`);
            isOpenRef.current = false;
            setConnected(false);
            teardownAudio();
            clearInterval(turnTimerRef.current);
            connectingRef.current = false;
          },
        },
      });

      // Worklet: resample → 16k Int16 → base64 → kirim
      await capCtxRef.current.audioWorklet.addModule(
        URL.createObjectURL(
          new Blob(
            [
              `
class PCM16ResampleWorklet extends AudioWorkletProcessor {
  constructor() { super(); this.targetRate = 16000; }
  _downsample(input) {
    const ratio = sampleRate / this.targetRate;
    if (ratio === 1) return input;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i=0; i<outLen; i++) out[i] = input[Math.floor(i*ratio)] || 0;
    return out;
  }
  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    const ds = this._downsample(input);
    const pcm = new Int16Array(ds.length);
    for (let i=0; i<ds.length; i++) {
      const s = Math.max(-1, Math.min(1, ds[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    this.port.postMessage(pcm, [pcm.buffer]);
    return true;
  }
}
registerProcessor('pcm16-resample-wl', PCM16ResampleWorklet);
        `,
            ],
            { type: "application/javascript" }
          )
        )
      );

      const node = new AudioWorkletNode(
        capCtxRef.current,
        "pcm16-resample-wl",
        { numberOfInputs: 1, numberOfOutputs: 0 }
      );
      node.port.onmessage = (ev) => {
        if (!isOpenRef.current) return;
        const int16 = ev.data; // Int16Array @ 16kHz
        const base64 = int16ToBase64(int16);
        try {
          sessionRef.current?.sendRealtimeInput?.({
            audio: { data: base64, mimeType: "audio/pcm;rate=16000" },
          });
          lastChunkAtRef.current = Date.now();
          sentSinceLastTurnRef.current = true;
        } catch {}
      };

      media.connect(node);
      workletNodeRef.current = node;
      sessionRef.current = live;
    } catch (e) {
      appendLog("start() error: " + (e?.message || e));
      setConnected(false);
      isOpenRef.current = false;
      connectingRef.current = false;
      teardownAudio();
    }
  }

  function teardownAudio() {
    try {
      workletNodeRef.current?.port &&
        (workletNodeRef.current.port.onmessage = null);
    } catch {}
    try {
      mediaNodeRef.current?.disconnect();
    } catch {}
    try {
      workletNodeRef.current?.disconnect();
    } catch {}
    try {
      streamRef.current?.getTracks()?.forEach((t) => t.stop());
    } catch {}
    queueRef.current = [];
    playingRef.current = false;
    try {
      capCtxRef.current?.suspend?.();
    } catch {}
    try {
      playCtxRef.current?.suspend?.();
    } catch {}
  }

  async function stop() {
    try {
      isOpenRef.current = false;
      await sessionRef.current?.close?.();
    } catch {}
    setConnected(false);
    teardownAudio();
    clearInterval(turnTimerRef.current);
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui" }}>
      <h1>AI at Home Test realtime voice</h1>

      <div style={{ display: "flex", gap: 12, margin: "12px 0" }}>
        {!connected ? (
          <button
            onClick={start}
            style={{ padding: 10 }}
            disabled={connectingRef.current}
          >
            {connectingRef.current ? "Connecting..." : "Start"}
          </button>
        ) : (
          <button onClick={stop} style={{ padding: 10 }}>
            Stop
          </button>
        )}
      </div>

      <h3>Chat History</h3>
      <div
        style={{
          border: "1px solid #ccc",
          padding: 12,
          height: 260,
          overflow: "auto",
        }}
      >
        {history.map((h, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <strong>{h.who}</strong> <small>{h.t}</small>
            <div>{h.text}</div>
          </div>
        ))}
      </div>

      <h3>Logs</h3>
      <pre style={{ whiteSpace: "pre-wrap", padding: 12 }}>
        {logs.join("\n")}
      </pre>
    </main>
  );
}
