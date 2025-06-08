// Actual ADK example code should be used. This is a simplified placeholder.
console.log("PCM Recorder Processor placeholder loaded.");
class PCMRecorderProcessor extends AudioWorkletProcessor {
    constructor(){super();}
    process(inputs){ const id=inputs[0]; if(id&&id.length>0){const cd=id[0]; if(cd instanceof Float32Array){const p=new Int16Array(cd.length);for(let i=0;i<cd.length;i++){let s=Math.max(-1,Math.min(1,cd[i]));p[i]=s<0?s*0x8000:s*0x7FFF;}this.port.postMessage(p.buffer,[p.buffer]);}} return true;}
}
try { registerProcessor('pcm-recorder-processor', PCMRecorderProcessor); } catch (e) { console.error("PCMRecorderProcessor registration failed:",e); }
