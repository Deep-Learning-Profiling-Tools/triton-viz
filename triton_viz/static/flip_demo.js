// Lightweight 2D canvas demo to visualize stepwise flip via view/reshape + swap
// API: createFlipDemo(containerElement, options) -> cleanup()
// options: { length: number, steps?: number[] } where steps are segment sizes halving each step

export function createFlipDemo(containerElement, options) {
    const length = Math.max(1, (options && options.length) || 16);
    const steps = (options && options.steps) || computeSteps(length);

    const wrapper = document.createElement('div');
    Object.assign(wrapper.style, {
        position: 'absolute', left: '0', top: '0', right: '0', bottom: '0',
        display: 'flex', flexDirection: 'column', gap: '10px', padding: '10px'
    });

    const toolbar = document.createElement('div');
    Object.assign(toolbar.style, { display: 'flex', gap: '8px', alignItems: 'center' });
    const playBtn = document.createElement('button'); playBtn.textContent = 'Play';
    const stepLabel = document.createElement('span'); stepLabel.textContent = 'Step: 0';
    const speedSel = document.createElement('select');
    ;['0.5x','1x','2x','4x'].forEach(s=>{ const o=document.createElement('option'); o.value=s; o.textContent=s; if(s==='1x') o.selected=true; speedSel.appendChild(o); });
    toolbar.appendChild(playBtn); toolbar.appendChild(stepLabel); toolbar.appendChild(speedSel);

    const canvas = document.createElement('canvas');
    canvas.width = Math.min(1200, containerElement.clientWidth - 20);
    canvas.height = Math.min(500, containerElement.clientHeight - 20);
    const ctx = canvas.getContext('2d');

    wrapper.appendChild(toolbar);
    wrapper.appendChild(canvas);
    containerElement.appendChild(wrapper);

    // data: 0..length-1
    let data = new Array(length).fill(0).map((_,i)=>i);
    let currentStep = 0;
    let playing = false;
    let rafId = 0;

    function computeSteps(n){
        // e.g. for 16 -> [8,4,2,1]
        const arr=[]; let s = Math.floor(n/2);
        while(s>=1){ arr.push(s); s=Math.floor(s/2);} return arr;
    }

    function lerp(a,b,t){ return a+(b-a)*t; }

    function hsv(h,s,v){
        let c=v*s; let x=c*(1-Math.abs((h/60)%2-1)); let m=v-c; let r,g,b;
        if(h<60){[r,g,b]=[c,x,0]} else if(h<120){[r,g,b]=[x,c,0]} else if(h<180){[r,g,b]=[0,c,x]}
        else if(h<240){[r,g,b]=[0,x,c]} else if(h<300){[r,g,b]=[x,0,c]} else {[r,g,b]=[c,0,x]}
        return `rgb(${Math.round((r+m)*255)},${Math.round((g+m)*255)},${Math.round((b+m)*255)})`;
    }

    function drawArray(arr, y, boxW, boxH, highlightPairs){
        for(let i=0;i<arr.length;i++){
            const x = 20 + i*boxW;
            const hue = lerp(0, 350, i/(arr.length-1||1));
            ctx.fillStyle = hsv(hue, 1, 1);
            ctx.strokeStyle = '#0a0a1a';
            ctx.lineWidth = 2;
            ctx.fillRect(x, y, boxW-2, boxH);
            ctx.strokeRect(x, y, boxW-2, boxH);
            ctx.fillStyle = '#fff';
            ctx.font = '14px monospace';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(String(arr[i]), x + (boxW-2)/2, y + boxH/2);
        }
        if (highlightPairs && highlightPairs.length){
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            highlightPairs.forEach(([i,j])=>{
                const x1 = 20 + i*boxW + (boxW-2)/2;
                const x2 = 20 + j*boxW + (boxW-2)/2;
                const yy = y + boxH + 6;
                ctx.beginPath(); ctx.moveTo(x1, yy); ctx.lineTo(x2, yy); ctx.stroke();
            });
        }
    }

    function calcPairs(seg){
        // pair indices across two adjacent blocks of size `seg` inside each group of size `2*seg`
        const pairs=[]; const n=length; const group = Math.max(1, 2*seg);
        for(let g=0; g<n; g+=group){
            const leftStart = g;
            const rightStart = g + seg;
            const leftLen = Math.min(seg, Math.max(0, n - leftStart));
            const rightLen = Math.min(seg, Math.max(0, n - rightStart));
            const pairLen = Math.min(leftLen, rightLen);
            for(let k=0;k<pairLen;k++){
                pairs.push([leftStart + k, rightStart + k]);
            }
        }
        return pairs;
    }

    function performSwap(arr, seg){
        // swap neighboring blocks of length `seg` within each group of length `2*seg`
        const res = arr.slice();
        const n = res.length; const group = Math.max(1, 2*seg);
        for(let g=0; g<n; g+=group){
            const leftStart = g;
            const rightStart = g + seg;
            const leftLen = Math.min(seg, Math.max(0, n - leftStart));
            const rightLen = Math.min(seg, Math.max(0, n - rightStart));
            const len = Math.min(leftLen, rightLen);
            for(let k=0;k<len;k++){
                const i = leftStart + k, j = rightStart + k;
                const tmp=res[i]; res[i]=res[j]; res[j]=tmp;
            }
        }
        return res;
    }

    function drawFrame(t){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.fillStyle='#1e1e28'; ctx.fillRect(0,0,canvas.width,canvas.height);
        const boxW = Math.max(18, Math.min(42, (canvas.width-40)/length));
        const boxH = 26;

        // initial array
        ctx.fillStyle = '#fff'; ctx.font = '16px sans-serif'; ctx.fillText('initial array', 20, 24);
        drawArray(new Array(length).fill(0).map((_,i)=>i), 40, boxW, boxH);

        let y = 100;
        for(let s=0;s<steps.length;s++){
            const seg = steps[s];
            ctx.fillStyle = '#fff'; ctx.font='14px sans-serif';
            ctx.fillText(`step ${s} view`, 20, y-6);
            drawArray(data, y, boxW, boxH);
            const pairs = calcPairs(seg);
            ctx.fillText(`step ${s} swap`, 20 + Math.min(canvas.width-20, (length*boxW)/2), y-6);
            // Draw swapped preview to the right
            const swapped = performSwap(data, seg);
            const offsetX = Math.min(canvas.width-20, (length*boxW)/2);
            ctx.save(); ctx.translate(offsetX, 0);
            drawArray(swapped, y, boxW, boxH, pairs);
            ctx.restore();
            y += boxH + 60;
            if (s === currentStep) break; // show up to current step during animation
        }

        if (currentStep === steps.length) {
            ctx.fillStyle = '#fff'; ctx.font='16px sans-serif'; ctx.fillText('final reshape', 20, y-6);
            drawArray(data, y, boxW, boxH);
        }
    }

    function tick(){
        drawFrame();
        if (!playing) return;
        const rate = speedSel.value==='0.5x'?800: speedSel.value==='1x'?400: speedSel.value==='2x'?200:100; // ms per step
        if (_lastStepTime === 0 || Date.now() - _lastStepTime >= rate){
            if (currentStep < steps.length){
                data = performSwap(data, steps[currentStep]);
                currentStep += 1;
                stepLabel.textContent = `Step: ${currentStep}`;
                _lastStepTime = Date.now();
            } else {
                playing = false; playBtn.textContent = 'Replay';
            }
        }
        rafId = requestAnimationFrame(tick);
    }
    let _lastStepTime = 0;

    playBtn.addEventListener('click', ()=>{
        if (!playing && currentStep===steps.length){
            // reset
            data = new Array(length).fill(0).map((_,i)=>i);
            currentStep = 0; stepLabel.textContent='Step: 0';
        }
        playing = !playing; playBtn.textContent = playing ? 'Pause' : (currentStep===steps.length ? 'Replay' : 'Play');
        if (playing){ _lastStepTime = 0; cancelAnimationFrame(rafId); rafId = requestAnimationFrame(tick);}
    });

    // initial paint
    drawFrame(0);

    return function cleanup(){ cancelAnimationFrame(rafId); if (wrapper && wrapper.remove) wrapper.remove(); };
}
