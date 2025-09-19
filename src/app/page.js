"use client";

import { useRef, useState, useEffect } from "react";
import { GoogleGenAI, Modality } from "@google/genai";

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState([]);
  const [history, setHistory] = useState([]);

  const sessionRef = useRef(null);
  const streamRef = useRef(null);
  const mediaNodeRef = useRef(null);
  const workletNodeRef = useRef(null);
  const audioCtxRef = useRef(null);

  const queueRef = useRef([]); // playback queue (Float32)
  const playingRef = useRef(false);

  const isOpenRef = useRef(false);
  const connectingRef = useRef(false);
  const sentSinceLastTurnRef = useRef(false);
  const lastChunkAtRef = useRef(0);
  const turnTimerRef = useRef(null);

  const appendLog = (s) => setLogs((p) => [...p, s]);
  const appendHistory = (who, text) => {
    if (!text) return;
    setHistory((h) => [
      ...h,
      { who, text, t: new Date().toLocaleTimeString() },
    ]);
  };

  // ==== utils ====
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

  // Float32Array -> Int16Array
  function float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }
  // Int16Array -> base64
  function int16ToBase64(int16) {
    const u8 = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
    let bin = "";
    const CHUNK = 0x8000;
    for (let i = 0; i < u8.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK));
    }
    return btoa(bin);
  }

  // === playback: FIX urutan start() ===
  function pumpAudio() {
    if (!audioCtxRef.current || playingRef.current) return;
    playingRef.current = true;
    const ctx = audioCtxRef.current;

    (async () => {
      while (queueRef.current.length) {
        const f32 = queueRef.current.shift();
        const buf = ctx.createBuffer(1, f32.length, 24000); // model out @24k
        buf.copyToChannel(f32, 0, 0);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        const done = new Promise((r) => (src.onended = r));
        src.start(0); // ✨ start DULU
        await done; // lalu tunggu selesai
      }
      playingRef.current = false;
    })();
  }

  // auto commit turn kalau diam
  function ensureTurnCommit() {
    clearInterval(turnTimerRef.current);
    turnTimerRef.current = setInterval(() => {
      if (!isOpenRef.current) return;
      const now = Date.now();
      if (sentSinceLastTurnRef.current && now - lastChunkAtRef.current > 900) {
        sessionRef.current?.sendRealtimeInput?.({ turnComplete: true });
        sentSinceLastTurnRef.current = false;
        appendLog("↪️ turnComplete");
      }
    }, 300);
  }

  async function fetchToken() {
    const url = process.env.NEXT_PUBLIC_TOKEN_ENDPOINT;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemini-2.5-flash-preview-native-audio-dialog",
      }),
    });
    if (!res.ok)
      throw new Error(`POST /token ${res.status}: ${await res.text()}`);
    const j = await res.json();
    if (!j.token) throw new Error("No token from backend");
    return j.token;
  }

  async function start() {
    if (connectingRef.current || connected) return;
    connectingRef.current = true;

    try {
      // AudioContext
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();
      } else if (audioCtxRef.current.state === "suspended") {
        await audioCtxRef.current.resume();
      }
      appendLog(`AudioContext SR: ${audioCtxRef.current.sampleRate} Hz`);

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
      const media = new MediaStreamAudioSourceNode(audioCtxRef.current, {
        mediaStream: stream,
      });
      mediaNodeRef.current = media;

      // token & connect
      const apiKey = await fetchToken();
      const ai = new GoogleGenAI({ apiKey, apiVersion: "v1alpha" });

      const model = "gemini-2.5-flash-preview-native-audio-dialog";
      const config = {
        responseModalities: [Modality.AUDIO],
        inputAudioTranscription: { languageCode: "id-ID" },
        outputAudioTranscription: { languageCode: "id-ID" },
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
          },
          onmessage: async (msg) => {
            // 1) AUDIO: ArrayBuffer / Blob
            if (msg.data) {
              let ab = null;
              if (msg.data instanceof ArrayBuffer) ab = msg.data;
              else if (msg.data instanceof Blob)
                ab = await msg.data.arrayBuffer();
              else if (msg.data?.buffer instanceof ArrayBuffer)
                ab = msg.data.buffer;
              if (ab) {
                queueRef.current.push(int16ToFloat32(ab));
                pumpAudio();
              }
            }
            // 2) AUDIO: inlineData base64 (kadang dikirim via parts)
            const sc = msg.serverContent || {};
            const parts = sc.modelTurn?.parts || [];
            for (const p of parts) {
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
            // 3) TRANSKRIP / TEKS (dua arah)
            if (sc.inputTranscription?.text)
              appendHistory("You", sc.inputTranscription.text.trim());
            if (sc.outputTranscription?.text)
              appendHistory("Gemini", sc.outputTranscription.text.trim());
            parts.forEach((p) => {
              if (p.text) appendHistory("Gemini", p.text.trim());
            });
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
      await audioCtxRef.current.audioWorklet.addModule(
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
        audioCtxRef.current,
        "pcm16-resample-wl",
        { numberOfInputs: 1, numberOfOutputs: 0 }
      );
      node.port.onmessage = (ev) => {
        if (!isOpenRef.current) return;
        const int16 = ev.data; // Int16Array @ 16kHz

        // encode base64
        const base64 = (function toB64(int16) {
          const u8 = new Uint8Array(
            int16.buffer,
            int16.byteOffset,
            int16.byteLength
          );
          let bin = "";
          const CH = 0x8000;
          for (let i = 0; i < u8.length; i += CH)
            bin += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
          return btoa(bin);
        })(int16);

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
      audioCtxRef.current?.suspend?.();
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
