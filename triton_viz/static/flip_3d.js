import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import { setupScene, setupGeometries, createCube, CUBE_SIZE, GAP } from './load_utils.js';

// 3D Flip visualization with layered rows (initial + step_i view + step_i swap)
// API: createFlip3D(container, { length, steps }) => cleanup()
export function createFlip3D(containerElement, options) {
    // overlay wrapper to guarantee visibility above existing content
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'absolute', left: '0', top: '0', right: '0', bottom: '0',
        zIndex: 3000, pointerEvents: 'auto'
    });
    containerElement.appendChild(overlay);
    const length = Math.max(2, (options && options.length) || 32);
    const steps = (options && options.steps) || computeSteps(length);

    const { scene, camera, renderer } = setupScene(overlay, 0x15151b);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.06;

    const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

    // Utilities
    function hsv(h,s,v){
        const c=v*s; const x=c*(1-Math.abs((h/60)%2-1)); const m=v-c; let r,g,b;
        if(h<60){[r,g,b]=[c,x,0]} else if(h<120){[r,g,b]=[x,c,0]} else if(h<180){[r,g,b]=[0,c,x]}
        else if(h<240){[r,g,b]=[0,x,c]} else if(h<300){[r,g,b]=[x,0,c]} else {[r,g,b]=[c,0,x]}
        return new THREE.Color(r+m, g+m, b+m);
    }
    const hueAt = (val)=> 360 * (val/(length-1||1));

    function createTextSprite(text){
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = 'Bold 28px Arial';
        const metrics = ctx.measureText(text);
        canvas.width = metrics.width + 8; canvas.height = 40;
        ctx.font = 'Bold 28px Arial';
        ctx.fillStyle = 'white'; ctx.fillText(text, 0, 28);
        const tex = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
        const sp = new THREE.Sprite(mat); sp.scale.set(4, 1.2, 1);
        return sp;
    }

    // Build sequences with side-by-side panels per step
    function computeSteps(n){ const arr=[]; let s=Math.floor(n/2); while(s>=1){ arr.push(s); s=Math.floor(s/2);} return arr; }
    function swapBlocksArray(array, seg){
        const res = array.slice(); const group = Math.max(1, 2*seg);
        for(let g=0; g<res.length; g+=group){
            const l = Math.min(seg, Math.max(0, res.length - g));
            const r = Math.min(seg, Math.max(0, res.length - (g+seg)));
            const len = Math.min(l, r);
            for(let k=0;k<len;k++){
                const i=g+k, j=g+seg+k; const t=res[i]; res[i]=res[j]; res[j]=t;
            }
        }
        return res;
    }

    function layoutWidth(seg, n){
        const groups = Math.ceil(n / (2*seg));
        const perGroup = seg * (CUBE_SIZE+GAP);
        const groupGap = CUBE_SIZE * 1.2;
        return groups*perGroup + Math.max(0, groups-1)*groupGap;
    }

    function buildPanel(arrVals, seg, y, labelText, xOffset){
        const panel = new THREE.Group();
        const groupGap = CUBE_SIZE * 1.2;
        const per = (CUBE_SIZE+GAP);
        const groups = Math.ceil(arrVals.length / (2*seg));
        for(let i=0;i<arrVals.length;i++){
            const g = Math.floor(i / (2*seg));
            const p = i % (2*seg);
            const row = (p >= seg) ? 1 : 0;
            const col = p % seg;
            const color = hsv(hueAt(arrVals[i]), 1, 1);
            const cube = createCube(color, 'Flip', col, row, g, cubeGeometry, edgesGeometry, lineMaterial);
            const x = xOffset + g*(seg*per + groupGap) + col*per;
            const yPos = y - row*per;
            cube.position.set(x, yPos, 0);
            panel.add(cube);
        }
        const label = createTextSprite(labelText);
        label.position.set(xOffset - (CUBE_SIZE*3.2), y + CUBE_SIZE*0.5, 0);
        panel.add(label);
        return panel;
    }

    // Build all layers
    const layers = [];
    const root = new THREE.Group(); scene.add(root);

    const rowGap = CUBE_SIZE * 4.2; // vertical distance per step row
    const columnGap = CUBE_SIZE * 6.0; // horizontal gap between view and swap panels

    let posY = 0;
    // initial 1D row (single panel)
    const initPanel = new THREE.Group();
    for (let i=0;i<length;i++){
        const color = hsv(hueAt(i), 1, 1);
        const cube = createCube(color, 'Flip', i, 0, 0, cubeGeometry, edgesGeometry, lineMaterial);
        cube.position.set(i*(CUBE_SIZE+GAP), posY, 0);
        initPanel.add(cube);
    }
    const initLabel = createTextSprite('initial array');
    initLabel.position.set(-CUBE_SIZE*3.2, posY + CUBE_SIZE*0.5, 0);
    initPanel.add(initLabel);
    root.add(initPanel); layers.push(initPanel);

    posY -= rowGap;
    let arr = Array.from({length}, (_,i)=>i);
    for (let s=0; s<steps.length; s++){
        const seg = steps[s];
        const lw = layoutWidth(seg, length);
        const viewX = 0;
        const swapX = viewX + lw + columnGap;
        const viewPanel = buildPanel(arr, seg, posY, `step ${s} view`, viewX);
        const swapped = swapBlocksArray(arr, seg);
        const swapPanel = buildPanel(swapped, seg, posY, `step ${s} swap`, swapX);
        root.add(viewPanel); root.add(swapPanel);
        layers.push(viewPanel); layers.push(swapPanel);
        arr = swapped.slice();
        posY -= rowGap;
    }

    // final reshape (1D row)
    const finalPanel = new THREE.Group();
    for (let i=0;i<arr.length;i++){
        const color = hsv(hueAt(arr[i]), 1, 1);
        const cube = createCube(color, 'Flip', i, 0, 0, cubeGeometry, edgesGeometry, lineMaterial);
        cube.position.set(i*(CUBE_SIZE+GAP), posY, 0);
        finalPanel.add(cube);
    }
    const finalLabel = createTextSprite('final reshape');
    finalLabel.position.set(-CUBE_SIZE*3.2, posY + CUBE_SIZE*0.5, 0);
    finalPanel.add(finalLabel);
    root.add(finalPanel); layers.push(finalPanel);

    // Set initial visibilities
    layers.forEach((g, i)=> g.visible = (i===0));

    // Center camera to see all layers
    const totalW = Math.max(layoutWidth(steps[0]||Math.floor(length/2), length) + columnGap + layoutWidth(steps[0]||Math.floor(length/2), length), length*(CUBE_SIZE+GAP));
    const totalH = (layers.length) * rowGap;
    camera.position.set(totalW*0.6, -totalH*0.35, Math.max(6, totalW*0.9));
    camera.lookAt(new THREE.Vector3(totalW*0.5, -totalH*0.4, 0));

    // UI controls
    const ui = document.createElement('div');
    Object.assign(ui.style, { position: 'absolute', top: '10px', left: '10px', display: 'flex', gap: '8px', zIndex: 2000 });
    const playBtn = document.createElement('button'); playBtn.textContent = 'Play';
    const stepLabel = document.createElement('span'); stepLabel.textContent = 'Layer: 1/' + layers.length; stepLabel.style.color = '#fff';
    const speedSel = document.createElement('select'); ['0.5x','1x','2x','4x'].forEach(s=>{ const o=document.createElement('option'); o.value=s; o.textContent=s; if(s==='1x') o.selected=true; speedSel.appendChild(o); });
    ui.appendChild(playBtn); ui.appendChild(stepLabel); ui.appendChild(speedSel); overlay.appendChild(ui);
    const closeBtn = document.createElement('button'); closeBtn.textContent = 'Close';
    ui.appendChild(closeBtn);

    // Animation loop: reveal rows sequentially
    let playing=false; let lastTs=0; let idxVisible=0; let raf=0;
    function rate(){ return speedSel.value==='0.5x'?800: speedSel.value==='1x'?400: speedSel.value==='2x'?200:100; }
    function loop(){
        controls.update();
        if (playing && (lastTs===0 || performance.now()-lastTs>=rate())){
            if (idxVisible < layers.length-1){
                idxVisible += 1; layers[idxVisible].visible = true; stepLabel.textContent = `Layer: ${idxVisible+1}/${layers.length}`; lastTs = performance.now();
            } else { playing=false; playBtn.textContent='Replay'; }
        }
        renderer.render(scene, camera);
        raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);

    playBtn.addEventListener('click', ()=>{
        if (!playing && idxVisible===layers.length-1){
            // reset visibility
            layers.forEach((r, i)=> r.visible = (i===0)); idxVisible = 0; stepLabel.textContent = `Layer: 1/${layers.length}`;
        }
        playing = !playing; playBtn.textContent = playing? 'Pause' : (idxVisible===layers.length-1? 'Replay' : 'Play');
        if (playing) lastTs = 0;
    });

    closeBtn.addEventListener('click', ()=> cleanup());

    function cleanup(){ cancelAnimationFrame(raf); controls.dispose(); renderer.dispose?.(); if (overlay && overlay.remove) overlay.remove(); }
    return cleanup;
}
