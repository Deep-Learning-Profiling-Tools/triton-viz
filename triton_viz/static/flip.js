import * as THREE from 'https://esm.sh/three@0.155.0';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';
import { setupScene, setupGeometries, createCube, CUBE_SIZE, GAP } from './load_utils.js';
import { createVectorText } from './dimension_utils.js';

export function createFlipVisualization(containerElement, op, viewState = null) {
    const API_BASE = window.__TRITON_VIZ_API__ || '';
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'absolute', left: '0', top: '0', right: '0', bottom: '0',
        zIndex: 3000, pointerEvents: 'auto'
    });
    containerElement.appendChild(overlay);
    try { window.current_op_uuid = op.uuid; } catch (e) {}

    const { scene, camera, renderer } = setupScene(overlay, 0x15151b);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.06;
    const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

    // Helper color mapping
    function hsv(h,s,v){
        const c=v*s; const x=c*(1-Math.abs((h/60)%2-1)); const m=v-c; let r,g,b;
        if(h<60){[r,g,b]=[c,x,0]} else if(h<120){[r,g,b]=[x,c,0]} else if(h<180){[r,g,b]=[0,c,x]}
        else if(h<240){[r,g,b]=[0,x,c]} else if(h<300){[r,g,b]=[x,0,c]} else {[r,g,b]=[c,0,x]}
        return new THREE.Color(r+m, g+m, b+m);
    }
    const shape = op.output_shape || op.input_shape || [];
    const dim = op.dim || 0;
    const is2D = (shape.length === 2);
    const H = is2D ? (shape[0]||8) : 1;
    const W = is2D ? (shape[1]||16) : ((shape[0]||16));

    // Build input and output layers
    const root = new THREE.Group(); scene.add(root);

    // Input row
    const inputGroup = new THREE.Group();
    for (let r=0;r<H;r++){
        for (let c=0;c<W;c++){
            const idx = r*W + c;
            const hue = (idx/((H*W)-1||1))*350;
            const cube = createCube(hsv(hue,1,1), 'Input', c, r, 0, cubeGeometry, edgesGeometry, lineMaterial);
            cube.position.set(c*(CUBE_SIZE+GAP), -r*(CUBE_SIZE+GAP), 0);
            inputGroup.add(cube);
        }
    }
    root.add(inputGroup);

    // Output row (flipped mapping)
    const outputGroup = new THREE.Group();
    const mapCoord = (r, c) => {
        if (!is2D) return { rr: 0, cc: (W-1-c) };
        if (dim === 0) return { rr: (H-1-r), cc: c };
        return { rr: r, cc: (W-1-c) };
    };
    for (let r=0;r<H;r++){
        for (let c=0;c<W;c++){
            const idx = r*W + c; const hue = (idx/((H*W)-1||1))*350;
            const { rr, cc } = mapCoord(r, c);
            const cube = createCube(hsv(hue,1,1), 'Output', cc, rr, 0, cubeGeometry, edgesGeometry, lineMaterial);
            cube.position.set(cc*(CUBE_SIZE+GAP), -rr*(CUBE_SIZE+GAP), - (CUBE_SIZE*6));
            outputGroup.add(cube);
        }
    }
    root.add(outputGroup);

    // Title sprites
    function makeLabel(text){
        return createVectorText(text, 'white', { fontSize: 0.8 });
    }
    const labIn = makeLabel('Input'); labIn.position.set(-CUBE_SIZE*3.2, CUBE_SIZE*0.5, 0); inputGroup.add(labIn);
    const labOut = makeLabel('Output (flipped)'); labOut.position.set(-CUBE_SIZE*3.2, -H*(CUBE_SIZE+GAP) - CUBE_SIZE*4.0, - (CUBE_SIZE*6)); outputGroup.add(labOut);

    // Layout: separate layers in Z
    inputGroup.position.set(0, 0, 0);
    outputGroup.position.set(0, -H*(CUBE_SIZE+GAP) - CUBE_SIZE*4.0, 0);

    // Interaction panel
    const sideMenu = document.createElement('div');
    Object.assign(sideMenu.style, { position:'absolute', top:'10px', right:'10px', width:'220px', padding:'10px', background:'rgba(0,0,0,0.7)', color:'#fff', borderRadius:'6px', zIndex:'2000' });
    overlay.appendChild(sideMenu);

    function toggleShowCode() {
        if (window.__tritonVizCodeToggle) {
            return window.__tritonVizCodeToggle();
        }
        return false;
    }

    if (window.setOpControlHandlers) {
        window.setOpControlHandlers({ toggleShowCode });
    }
    if (window.setOpControlState) {
        window.setOpControlState({ colorize: false, showCode: false });
    }

    async function getFlipValue(which, r, c){
        const body = { uuid: op.uuid, which, x: c, y: r };
        const res = await fetch(`${API_BASE}/api/getFlipValue`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        return await res.json();
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredCube = null;
    function clearHoverOutlines(){
        // reset outlines on all cubes to avoid residual highlights
        const groups = [inputGroup, outputGroup];
        for (const g of groups){
            if (!g) continue;
            for (const child of g.children){
                const ho = child && child.getObjectByName && child.getObjectByName('hoverOutline');
                if (ho) ho.visible = false;
            }
        }
    }
    function syncHoverHighlight(){
        // periodically enforce: only current hovered cube is highlighted
        clearHoverOutlines();
        if (hoveredCube){
            const ho = hoveredCube.getObjectByName && hoveredCube.getObjectByName('hoverOutline');
            if (ho) ho.visible = true;
        }
    }
    // run every ~60ms (about 16 FPS) to avoid leaving residual highlights
    const _hoverTimer = setInterval(syncHoverHighlight, 60);
    function _updateMouseNDC(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        const dpr = renderer.getPixelRatio();
        const px = (event.clientX - rect.left) * dpr;
        const py = (event.clientY - rect.top ) * dpr;
        const w = rect.width * dpr, h = rect.height * dpr;
        mouse.x = (px / w) * 2 - 1; mouse.y = -(py / h) * 2 + 1;
    }
    function findTopLevel(obj){ let n=obj; while(n && !(n.userData && n.userData.tensorName)) n=n.parent; return n; }
    function updatePanel(which, r, c, value){
        if (!which){ sideMenu.innerHTML=''; return; }
        sideMenu.innerHTML = `<h3>${which} Tensor</h3><p>Row: ${r+1}</p><p>Col: ${c+1}</p><p>Value: ${value !== undefined ? value : '...'}</p>`;
    }

    async function onMouseMove(event){
        _updateMouseNDC(event); raycaster.setFromCamera(mouse, camera);
        const objs = [...inputGroup.children, ...outputGroup.children];
        const hits = raycaster.intersectObjects(objs, true);
        // only update target; periodic timer will enforce single highlight
        hoveredCube = null;
        if (hits.length === 0){ updatePanel('',0,0,undefined); return; }
        let cube = findTopLevel(hits[0].object); if (!cube) { updatePanel('',0,0,undefined); return; }
        hoveredCube = cube;
        const which = cube.userData.tensorName === 'Input' ? 'input' : 'output';
        const r = cube.userData.tensor1 || 0; const c = cube.userData.tensor0 || 0;
        try { const res = await getFlipValue(which, r, c); updatePanel(which, r, c, res.value); } catch(e){ updatePanel(which, r, c, undefined); }
        // do not toggle here; timer will handle visibility to avoid residuals
    }
    overlay.addEventListener('mousemove', onMouseMove);
    overlay.addEventListener('mouseleave', ()=>{ hoveredCube=null; updatePanel('',0,0,undefined); syncHoverHighlight(); });

    // Camera framing
    const box = new THREE.Box3().setFromObject(root); const center = box.getCenter(new THREE.Vector3()); const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z); const fov = camera.fov * (Math.PI/180); let cameraZ = Math.abs(maxDim/2/Math.tan(fov/2)); cameraZ *= 1.6;
    camera.position.set(center.x, center.y, center.z + cameraZ); camera.lookAt(center);

    function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
    animate();

    const closeBtn = document.createElement('button'); closeBtn.textContent = 'Close'; Object.assign(closeBtn.style, { position:'absolute', top:'50px', left:'10px', zIndex:'2001' }); overlay.appendChild(closeBtn);
    function cleanup(){
        if (window.setOpControlHandlers) {
            window.setOpControlHandlers(null);
        }
        if (window.setOpControlState) {
            window.setOpControlState({ colorize: false, showCode: false });
        }
        if (window.__tritonVizCodeHide && !window.__tritonVizPreserveCodePanel) {
            window.__tritonVizCodeHide();
        }
        clearInterval(_hoverTimer);
        overlay.removeEventListener('mousemove', onMouseMove);
        overlay.removeEventListener('mouseleave', ()=>{});
        if (overlay && overlay.remove) overlay.remove();
    }
    closeBtn.addEventListener('click', cleanup);
    try { window.current_op_uuid = op.uuid; } catch (e) {}
    return cleanup;
}
