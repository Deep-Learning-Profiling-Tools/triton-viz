import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';
import { OrbitControls } from 'https://esm.sh/three@0.155.0/examples/jsm/controls/OrbitControls.js';

export function createMatMulVisualization(containerElement, op) {
    const { input_shape, other_shape, output_shape } = op;
    console.log(op.uuid)
    fetch('/api/setop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ uuid: op.uuid }),
    })
    .then(response => response.json())
    .then(data => console.log('Set current op:', data))
    .catch((error) => console.error('Error:', error));
    let currentStep = 0;
    const totalSteps = input_shape[1];
    let frame = 0;


    const sideMenu = document.createElement('div');
    sideMenu.style.position = 'absolute';
    sideMenu.style.top = '10px';
    sideMenu.style.right = '10px';
    sideMenu.style.width = '200px';
    sideMenu.style.padding = '10px';
    sideMenu.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    sideMenu.style.color = 'white';
    sideMenu.style.fontFamily = 'Arial, sans-serif';
    sideMenu.style.fontSize = '14px';
    sideMenu.style.borderRadius = '5px';
    containerElement.appendChild(sideMenu);
    let hoveredCube = null;




    async function getElementValue( matrixName, row, col) {
        let uuid = op.uuid;
        const response = await fetch('/api/getValue', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ uuid, matrixName, row, col, currentStep }),
        });
        return await response.json();
    }


    function updateSideMenu(matrix, x, y) {
        if (!matrix) {
            sideMenu.innerHTML = '';
            return;
        }

        let matrixName;
        let dims;
        if (matrix === matrixA) {
            matrixName = 'A';
            dims = input_shape;
        } else if (matrix === matrixB) {
            matrixName = 'B';
            dims = other_shape;
        } else if (matrix === matrixC) {
            matrixName = 'C';
            dims = output_shape;
        } else {
            sideMenu.innerHTML = '';
            return;
        }
        console.log(matrixName, "x:", (x + 1), "y:", (y + 1));
        sideMenu.innerHTML = `
            <h3 style="margin-top: 0;">Matrix ${matrixName}</h3>
            <p>Row: ${y + 1}</p>
            <p>Column: ${x + 1}</p>
            <p>Dimensions: ${dims[0]} x ${dims[1]}</p>
        `;
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    // persistent calibration offsets shared with Load view for consistency
    let mouseDx = Number(localStorage.getItem('viz_mouse_dx') || 0);
    let mouseDy = Number(localStorage.getItem('viz_mouse_dy') || 0);


    function _updateMouseNDC(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        const dpr = (window.devicePixelRatio || 1);
        const px = (event.clientX - rect.left + mouseDx) * dpr;
        const py = (event.clientY - rect.top  + mouseDy) * dpr;
        const w = rect.width * dpr, h = rect.height * dpr;
        mouse.x = (px / w) * 2 - 1;
        mouse.y = -(py / h) * 2 + 1;
    }

    async function onMouseMove(event) {
        _updateMouseNDC(event);

        raycaster.setFromCamera(mouse, camera);

        const allMatrixChildren = [
            ...(matrixA ? matrixA.children : []),
            ...(matrixB ? matrixB.children : []),
            ...(matrixC ? matrixC.children : [])
        ];

        const intersects = raycaster.intersectObjects(allMatrixChildren, true);

        if (hoveredCube) {
            hoveredCube.getObjectByName('hoverOutline').visible = false;
            hoveredCube = null;
        }

        if (intersects.length > 0) {
            // Find the actual cube (parent of the intersected object)
            hoveredCube = intersects[0].object;
            while (hoveredCube && !hoveredCube.matrixName) {
                hoveredCube = hoveredCube.parent;
            }

            if (hoveredCube) {
                const hoverOutline = hoveredCube.getObjectByName('hoverOutline');
                if (hoverOutline) {
                    hoverOutline.visible = true;
                }
                const res = await getElementValue(hoveredCube.matrixName, hoveredCube.matrixRow, hoveredCube.matrixCol);
                // Log the matrix name, row, and column of the hovered cube
                console.log(
                    // `Matrix: ${hoveredCube.matrixName}, ` +
                    // `Row: ${hoveredCube.matrixRow + 1}, ` +
                    // `Column: ${hoveredCube.matrixCol + 1}`+
                    `Value: ${res.value}`
                );

                updateSideMenu(hoveredCube.matrixName, hoveredCube.matrixRow, hoveredCube.matrixCol);
            }
        } else {
            updateSideMenu(null);
        }
    }





    const CUBE_SIZE = 0.2;
    const GAP = 0.05;

    const COLOR_A = new THREE.Color(0.53, 0.81, 0.98);
    const COLOR_B = new THREE.Color(1.0, 0.65, 0.0);
    const COLOR_C = new THREE.Color(1.0, 1.0, 1.0);
    const COLOR_HIGHLIGHT = new THREE.Color(0.0, 0.0, 1.0);
    const COLOR_FILLED = new THREE.Color(0.0, 0.0, 1.0);
    const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);
    const COLOR_EDGE = new THREE.Color(0.3, 0.3, 0.3);
    const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);

    const scene = new THREE.Scene();
    scene.background = COLOR_BACKGROUND;
    const camera = new THREE.PerspectiveCamera(45, containerElement.clientWidth / containerElement.clientHeight, 0.1, 1000);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const dpr = (window.devicePixelRatio || 1);
    renderer.setPixelRatio(dpr);
    renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    containerElement.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const edgesGeometry = new THREE.EdgesGeometry(cubeGeometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: COLOR_EDGE });

    function createCube(color, matrixName, i, j) {
        const cubeMaterial = new THREE.MeshPhongMaterial({ color: color });
        const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
        const edges = new THREE.LineSegments(edgesGeometry, lineMaterial);
        cube.add(edges);

        const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
        const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
        const hoverOutline = new THREE.LineSegments(hoverEdgesGeometry, new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
        hoverOutline.visible = false;
        hoverOutline.name = 'hoverOutline';
        cube.add(hoverOutline);

        cube.name = `${matrixName}_cube_${i}_${j}`;
        cube.matrixName = matrixName;
        cube.matrixRow = i;
        cube.matrixCol = j;

        return cube;
    }

    function createMatrix(dimensions, position, color, matrixName) {
        const matrix = new THREE.Group();
        matrix.userData.dimensions = dimensions;
        for (let i = 0; i < dimensions[0]; i++) {
            for (let j = 0; j < dimensions[1]; j++) {
                const cube = createCube(color, matrixName, i, j);
                cube.position.set(
                    position.x + j * (CUBE_SIZE + GAP),
                    position.y - i * (CUBE_SIZE + GAP),
                    position.z
                );
                matrix.add(cube);
            }
        }
        return matrix;
    }

    const matrixA = createMatrix(input_shape, new THREE.Vector3(-10, 10, 0), COLOR_A, 'A');
    const matrixB = createMatrix(other_shape, new THREE.Vector3(0, 10, 0), COLOR_B, 'B');
    const matrixC = createMatrix(output_shape, new THREE.Vector3(-5, -4, 0), COLOR_C, 'C');

    scene.add(matrixA);
    scene.add(matrixB);
    scene.add(matrixC);

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    const box = new THREE.Box3().setFromObject(scene);
    box.getCenter(center);
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.copy(center);
    controls.update();

    let isPaused = false;

    const totalFrames = input_shape[0] * other_shape[1];

    function highlightCubes(matrix, indices, highlightColor) {
        indices.forEach(([i, j]) => {
            if (i >= 0 && i < matrix.userData.dimensions[0] && j >= 0 && j < matrix.userData.dimensions[1]) {
                const index = i * matrix.userData.dimensions[1] + j;
                if (index < matrix.children.length) {
                    matrix.children[index].material.color.copy(highlightColor);

                }
            }
        });
    }

    function resetColors() {
        matrixA.children.forEach(cube => cube.material.color.copy(COLOR_A));
        matrixB.children.forEach(cube => cube.material.color.copy(COLOR_B));
        matrixC.children.forEach(cube => cube.material.color.copy(COLOR_C));
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        if (!isPaused && frame < totalFrames) {
            resetColors();

            const row = Math.floor(frame / other_shape[1]);
            const col = frame % other_shape[1];
            currentStep = frame % totalSteps + 1;

            const highlightA = Array.from({ length: input_shape[1] }, (_, i) => [row, i]);
            const highlightB = Array.from({ length: other_shape[0] }, (_, i) => [i, col]);
            const highlightC = [[row, col]];

            highlightCubes(matrixA, highlightA, COLOR_HIGHLIGHT);
            highlightCubes(matrixB, highlightB, COLOR_HIGHLIGHT);
            highlightCubes(matrixC, highlightC, COLOR_FILLED);

            frame++;
        }

        renderer.render(scene, camera);
    }

    function onResize() {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    }

    const controlPanel = document.createElement('div');
    controlPanel.style.position = 'absolute';
    controlPanel.style.bottom = '10px';
    controlPanel.style.left = '10px';
    controlPanel.style.display = 'flex';
    controlPanel.style.gap = '10px';

    const playPauseButton = document.createElement('button');
    playPauseButton.textContent = 'Play/Pause';
    playPauseButton.addEventListener('click', () => {
        isPaused = !isPaused;
    });

    const resetButton = document.createElement('button');
    resetButton.textContent = 'Reset';
    resetButton.addEventListener('click', () => {
        frame = 0;
        resetColors();
    });

    controlPanel.appendChild(playPauseButton);
    controlPanel.appendChild(resetButton);

    // Color by Value (for C matrix) toggle + legend (mono blue)
    const colorToggle = document.createElement('button');
    colorToggle.textContent = 'Color by Value: OFF';
    controlPanel.appendChild(colorToggle);
    let colorOn = false;
    let legendEl = null;
    function destroyLegend(){ if(legendEl && legendEl.remove) legendEl.remove(); legendEl=null; }
    function createLegend(min,max){
        destroyLegend();
        const w = document.createElement('div');
        Object.assign(w.style,{position:'absolute', left:'10px', bottom:'60px', background:'rgba(0,0,0,0.6)', color:'#fff', padding:'6px 8px', borderRadius:'6px'});
        const c = document.createElement('canvas'); c.width=220; c.height=10; const ctx=c.getContext('2d');
        for(let x=0;x<c.width;x++){ const t=x/(c.width-1); const r=t,g=0.2,b=1-t; ctx.fillStyle=`rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`; ctx.fillRect(x,0,1,c.height);}
        const lab=document.createElement('div'); lab.style.display='flex'; lab.style.justifyContent='space-between'; lab.style.marginTop='2px'; lab.innerHTML=`<span>${min.toFixed(3)}</span><span>${max.toFixed(3)}</span>`;
        const ttl=document.createElement('div'); ttl.textContent='Value (C)'; ttl.style.marginBottom='4px'; ttl.style.opacity='0.9';
        w.appendChild(ttl); w.appendChild(c); w.appendChild(lab); containerElement.appendChild(w); legendEl=w;
    }
    async function fetchCValues(){
        try{ const res=await fetch('/api/getMatmulC',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uuid: op.uuid})});
            return await res.json(); }catch(e){ return {error:String(e)} }
    }
    function applyColorCWithData(data){ if(!colorOn||!data||data.error) return; try{
            const M=(data.shape||[])[0]||0, N=(data.shape||[])[1]||0; const vals2d=data.values||[]; const vals=[]; const cells=[];
            matrixC.children.forEach((cube,i)=>{ const row=Math.floor(i/N), col=i%N; cells.push({cube,row,col}); vals.push((row<M&&col<N)? vals2d[row][col]:0.0); });
            const mn=Math.min(...vals), mx=Math.max(...vals);
            cells.forEach((e,idx)=>{ const v=vals[idx]; const u=(mx===mn)?0.5:(v-mn)/(mx-mn); const r=u,g=0.2,b=1-u; e.cube.material.color.setRGB(r,g,b); });
            createLegend(mn,mx);
        }catch(e){}
    }
    colorToggle.addEventListener('click', async ()=>{ colorOn=!colorOn; colorToggle.textContent=`Color by Value: ${colorOn?'ON':'OFF'}`; if(!colorOn){ destroyLegend(); resetColors(); } else { const data=await fetchCValues(); applyColorCWithData(data); } });

    containerElement.appendChild(controlPanel);

    const PAN_SPEED = 0.1;
    const TILT_SPEED = 0.02;
    const ZOOM_SPEED = 0.5;
    let cameraRotation = new THREE.Euler(0, 0, 0, 'YXZ');

    function onKeyDown(event) {
        switch (event.key.toLowerCase()) {
            case 'w':
                camera.position.y += PAN_SPEED;
                break;
            case 's':
                camera.position.y -= PAN_SPEED;
                break;
            case 'a':
                camera.position.x -= PAN_SPEED;
                break;
            case 'd':
                camera.position.x += PAN_SPEED;
                break;
            case 'arrowup':
                cameraRotation.x -= TILT_SPEED;
                break;
            case 'arrowdown':
                cameraRotation.x += TILT_SPEED;
                break;
            case 'arrowleft':
                cameraRotation.y -= TILT_SPEED;
                break;
            case 'arrowright':
                cameraRotation.y += TILT_SPEED;
                break;
            case 'o':
                camera.position.z += ZOOM_SPEED;
                break;
            case 'p':
                camera.position.z -= ZOOM_SPEED;
                break;
        }
        camera.setRotationFromEuler(cameraRotation);
        camera.updateProjectionMatrix();
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('keydown', onKeyDown);
    containerElement.addEventListener('mousemove', onMouseMove);
    try { window.current_op_uuid = op.uuid; } catch (e) {}

    // Mouse wheel zoom for matmul view
    const WHEEL_ZOOM_SPEED = 0.5;
    containerElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        camera.position.z += direction * WHEEL_ZOOM_SPEED;
        camera.updateProjectionMatrix();
    }, { passive: false });

    animate();


    function cleanup() {
        window.removeEventListener('resize', onResize);
        window.removeEventListener('keydown', onKeyDown);
        containerElement.removeEventListener('mousemove', onMouseMove);
        containerElement.innerHTML = '';
        renderer.dispose();
        scene.clear();
    }

    return cleanup;
}
