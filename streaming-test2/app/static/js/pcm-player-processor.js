// Actual ADK example code should be used. This is a simplified placeholder.
console.log("PCM Player Processor placeholder loaded.");
class PCMPlayerProcessor extends AudioWorkletProcessor {
    constructor(options) { super(options); this.buffer = []; this.port.onmessage = e => { const d=new Int16Array(e.data); for(let i=0;i<d.length;i++) this.buffer.push(d[i]/32768.0);}; }
    process(i, outputs) { const o=outputs[0][0]; for(let i=0;i<o.length;i++) o[i]=this.buffer.length?this.buffer.shift():0; return true; }
}
try { registerProcessor('pcm-player-processor', PCMPlayerProcessor); } catch (e) { console.error("PCMPlayerProcessor registration failed:",e); }
