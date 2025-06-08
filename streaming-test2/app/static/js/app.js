const camStatus = document.getElementById('cameraAgentStatus');
const camLog = document.getElementById('cameraLogOutput');
const audStatus = document.getElementById('audioAgentStatus');
const audLog = document.getElementById('audioLogOutput');
const startBtn = document.getElementById('startAudioButton');
const stopBtn = document.getElementById('stopAudioButton');
const textIn = document.getElementById('textInput');
const sendBtn = document.getElementById('sendTextButton');
const agentSpeech = document.getElementById('currentAgentResponse');

let sse = null; let audioCtx = null; let pcmPlayer = null; let micStream = null; let pcmRecorder = null;
const sessId = Date.now().toString() + Math.random().toString().substring(2,8);
console.log("Client Session ID:", sessId);

function b64ToAb(b64) { const s = atob(b64); const l = s.length; const u = new Uint8Array(l); while(l--) u[l] = s.charCodeAt(l); return u.buffer; }
function abToB64(ab) { let r = ''; const b = new Uint8Array(ab); const l = b.byteLength; for(let i=0;i<l;i++) r+=String.fromCharCode(b[i]); return btoa(r); }

async function fetchLog(type, el) {
    try {
        const res = await fetch(`/logs/${type}`);
        if(!res.ok) { el.textContent = `Error: ${res.statusText}`; return; }
        const logs = await res.json();
        if(logs.error) el.textContent = `Error in log data: ${logs.error}`;
        else if(Array.isArray(logs)) el.textContent = logs.map(l => `[${l.timestamp || ''}] ${l.speaker || l.type || ''}: ${l.text || l.comment_by_llm || l.description_by_cv || ''}`).join('\n');
        else el.textContent = "Bad log format.";
    } catch (e) { el.textContent = `Failed to fetch ${type} log: ${e}`; }
}
async function fetchCamStatus() { try { const r = await fetch('/agent/camera/status'); const d = await r.json(); camStatus.textContent = `Status: ${d.status}${d.monitoring?' (Monitoring)':''}`; } catch (e) { camStatus.textContent = "Status: Error"; }}

async function setupAudio() {
    if(!audioCtx) audioCtx = new AudioContext({sampleRate: 24000}); // Player rate
    if(audioCtx.state === 'suspended') await audioCtx.resume();
    try {
        if(!audioCtx.audioWorklet) { console.error("AudioWorklet not supported!"); return false; }
        await audioCtx.audioWorklet.addModule('/static/js/pcm-player-processor.js');
        pcmPlayer = new AudioWorkletNode(audioCtx, 'pcm-player-processor');
        pcmPlayer.connect(audioCtx.destination);
        await audioCtx.audioWorklet.addModule('/static/js/pcm-recorder-processor.js');
        // pcmRecorder will be created on mic access
        console.log("Audio Worklets setup."); return true;
    } catch (e) { console.error("Audio Worklet setup failed:", e); audStatus.textContent = "Error: Audio setup failed."; return false; }
}

function connectSSE(isAudio = true) {
    if(sse) sse.close();
    sse = new EventSource(`/events/${sessId}?is_audio=${isAudio}`);
    sse.onopen = () => { console.log("SSE open."); audStatus.textContent = isAudio?"Audio Connected":"Text Connected"; startBtn.disabled=true; stopBtn.disabled=false; textIn.disabled=false; sendBtn.disabled=false; agentSpeech.textContent=""; };
    sse.onmessage = (evt) => {
        const msg = JSON.parse(evt.data); console.log("[AGENT->CLIENT]", msg);
        if(msg.error) { console.error("SSE Server Error:", msg.error); audStatus.textContent = `Error: ${msg.error}`; agentSpeech.textContent = `Error: ${msg.error}`; return; }
        if(msg.turn_complete) { console.log("Agent turn done."); if(audStatus.textContent.includes("Speaking")) audStatus.textContent="Audio Connected"; return; }
        if((msg.mime_type==="audio/pcm"||msg.mime_type==="audio/opus") && pcmPlayer && msg.data) { pcmPlayer.port.postMessage(b64ToAb(msg.data)); audStatus.textContent="Agent Speaking..."; agentSpeech.textContent=""; }
        else if(msg.mime_type==="text/plain") { if(!audStatus.textContent.includes("Speaking")) audStatus.textContent="Agent Responding..."; agentSpeech.textContent += msg.data; }
        fetchLog('audio', audLog);
    };
    sse.onerror = (e) => { console.error("SSE error:", e); audStatus.textContent="Conn Error. Retry..."; if(sse)sse.close(); setTimeout(()=>connectSSE(isAudio), 3000);};
}

async function sendAudio(ab) {
    if(!sse || sse.readyState !== EventSource.OPEN) { console.warn("SSE not open for audio."); return; }
    try { const r = await fetch(`/send/${sessId}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mime_type:'audio/pcm', data:abToB64(ab)})}); if(!r.ok) console.error("Send audio error:", r.statusText); } catch (e) { console.error("Send audio exception:", e); }
}
async function sendText(txt) {
    if(!sse || sse.readyState !== EventSource.OPEN) { alert("Not connected!"); return; }
    if(!txt.trim()) return;
    try { const r = await fetch(`/send/${sessId}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mime_type:'text/plain', data:txt})}); if(!r.ok) alert(`Send error: ${r.statusText}`); else agentSpeech.textContent=""; } catch (e) { alert(`Send exception: ${e}`);}
    textIn.value = "";
}

async function startMic() {
    if(!audioCtx && !(await setupAudio())) { alert("Audio system failed to init."); return;}
    if(audioCtx.state === 'suspended') await audioCtx.resume();
    try {
        micStream = await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000, channelCount:1}}); // Recorder rate
        if(!pcmRecorder) pcmRecorder = new AudioWorkletNode(audioCtx, 'pcm-recorder-processor');
        pcmRecorder.port.onmessage = (evt) => { if(evt.data instanceof ArrayBuffer && evt.data.byteLength>0) sendAudio(evt.data);};
        const srcNode = audioCtx.createMediaStreamSource(micStream);
        srcNode.connect(pcmRecorder); // DO NOT connect pcmRecorder to audioCtx.destination
        console.log("Mic access OK, recording."); audStatus.textContent="Recording..."; startBtn.disabled=true; stopBtn.disabled=false; textIn.disabled=false; sendBtn.disabled=false;
    } catch (e) { console.error("Mic/record error:", e); audStatus.textContent="Error: Mic failed."; alert("Mic failed. Check permissions.");}
}
function stopMicSSE() {
    if(micStream) micStream.getTracks().forEach(t=>t.stop()); micStream=null; console.log("Mic stopped.");
    if(pcmRecorder) pcmRecorder.disconnect(); // Disconnect but don't nullify, can be reused.
    if(sse) sse.close(); console.log("SSE closed by client.");
    audStatus.textContent="Idle"; startBtn.disabled=false; stopBtn.disabled=true; textIn.disabled=true; sendBtn.disabled=true;
}

startBtn.addEventListener('click', async () => { if(!(await setupAudio())) { alert("Audio setup failed!"); return; } connectSSE(true); setTimeout(startMic, 200);});
stopBtn.addEventListener('click', stopMicSSE);
sendBtn.addEventListener('click', () => sendText(textIn.value));
textIn.addEventListener('keypress', (e) => { if(e.key==='Enter') sendText(textIn.value);});

function init() {
    console.log("App init, session:", sessId);
    fetchLog('camera',camLog); fetchLog('audio',audLog); fetchCamStatus();
    setInterval(()=>{ fetchLog('camera',camLog); fetchLog('audio',audLog); fetchCamStatus(); }, 5000);
    setupAudio(); // Init audio system on load
}
init();
