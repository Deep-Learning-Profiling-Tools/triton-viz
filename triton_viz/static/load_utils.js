import { createCadDimension, createVectorText } from './dimension_utils.js';
import * as THREE from 'https://esm.sh/three@0.155.0';

export const CUBE_SIZE = 0.2;
export const GAP = 0.05;
export const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);
export const COLOR_EDGE = new THREE.Color(0.5, 0.5, 0.5);

const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);

export function setupScene(container, backgroundColor = 0x000000) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    try {
        if ('outputEncoding' in renderer) renderer.outputEncoding = THREE.sRGBEncoding;
        if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.NoToneMapping;
    } catch (e) {}

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    return { scene, camera, renderer };
}

export function setupGeometries() {
    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const normals = cubeGeometry.attributes.normal;
    const colorArray = new Float32Array(cubeGeometry.attributes.position.count * 3);
    const lightDir = new THREE.Vector3(0.35, 0.6, 0.7).normalize();
    for (let i = 0; i < normals.count; i++) {
        const nx = normals.getX(i);
        const ny = normals.getY(i);
        const nz = normals.getZ(i);
        const ndotl = Math.max(0, nx * lightDir.x + ny * lightDir.y + nz * lightDir.z);
        const shade = 0.6 + 0.4 * ndotl;
        const base = i * 3;
        colorArray[base] = shade; colorArray[base + 1] = shade; colorArray[base + 2] = shade;
    }
    cubeGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    const edgesGeometry = new THREE.EdgesGeometry(cubeGeometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: COLOR_EDGE, linewidth: 0.1, transparent: true, opacity: 0.3 });
    return { cubeGeometry, edgesGeometry, lineMaterial };
}

export function createTensor(shape, coords, color, tensorName, cubeGeometry) {
    console.log(`Creating ${tensorName} tensor:`, shape, coords);
    const tensor = new THREE.Group();
    let depth, height, width;
    if (shape.length === 1) { width = shape[0]; height = 1; depth = 1; }
    else if (shape.length === 2) { height = shape[0]; width = shape[1]; depth = 1; }
    else { [depth, height, width] = shape; }

    const spacing = CUBE_SIZE + GAP;
    const centerX = (width - 1) * spacing / 2;
    const centerY = -((height - 1) * spacing / 2);
    const centerZ = -((depth - 1) * spacing / 2);

    const isGlobal = tensorName === 'Global';
    const isDense = isGlobal || !coords; // Treat as dense if named Global or no coords provided
    const instanceCount = isDense ? width * height * depth : coords.length;

    const mesh = new THREE.InstancedMesh(cubeGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true }), instanceCount);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);
    const baseColor = color instanceof THREE.Color ? color.clone() : new THREE.Color(color);
    mesh.userData.color_base = baseColor;
    mesh.userData.shape_raw = shape;
    mesh.userData.tensorName = tensorName;
    mesh.userData.shape = { depth, height, width };

    const matrix = new THREE.Matrix4();
    const highlightedIndices = new Set();

    if (isDense) {
        let best = [0, 1, 2];
        if (coords && coords.length > 0) {
            const samples = coords.slice(0, Math.min(256, coords.length));
            const maxIncoming = [0, 0, 0];
            samples.forEach(c => { if (c[0]>maxIncoming[0]) maxIncoming[0]=c[0]; if (c[1]>maxIncoming[1]) maxIncoming[1]=c[1]; if (c[2]>maxIncoming[2]) maxIncoming[2]=c[2]; });
            const target = [width - 1, height - 1, depth - 1];
            const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
            const score = (p) => { let s=0; for(let i=0;i<3;i++){ s+=Math.abs(maxIncoming[i]-target[p[i]]); if(maxIncoming[i]>target[p[i]]) s+=1000; } return s; };
            let bestScore = score(best);
            perms.forEach(p => { let sc=score(p); if(sc<bestScore){ best=p; bestScore=sc; } });
            const remap = (c) => [c[best.indexOf(0)], c[best.indexOf(1)], c[best.indexOf(2)]];
            coords.forEach(c => { const [x,y,z]=remap(c); if(x>=0&&x<width&&y>=0&&y<height&&z>=0&&z<depth) highlightedIndices.add(z*(width*height)+y*width+x); });
        }
        const coordList = [];
        let idx = 0;
        for (let z=0; z<depth; z++) {
            for (let y=0; y<height; y++) {
                for (let x=0; x<width; x++) {
                    matrix.setPosition(x*spacing-centerX, -y*spacing-centerY, -z*spacing-centerZ);
                    mesh.setMatrixAt(idx, matrix);
                    mesh.setColorAt(idx, highlightedIndices.has(idx) ? COLOR_SLICE : baseColor);
                    idx++; coordList.push([x,y,z]);
                }
            }
        }
        mesh.userData.coords = coordList;
    } else {
        const cArr = coords || [];
        cArr.forEach(([x, y, z], idx) => {
            matrix.setPosition(x*spacing-centerX, -y*spacing-centerY, -z*spacing-centerZ);
            mesh.setMatrixAt(idx, matrix); mesh.setColorAt(idx, baseColor);
        });
        mesh.userData.coords = cArr;
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingBox(); mesh.computeBoundingSphere();
    tensor.add(mesh);
    tensor.userData.mesh = mesh;
    tensor.userData.color = baseColor;
    tensor.userData.highlightedIndices = highlightedIndices;
    return tensor;
}

export function updateTensorHighlights(tensor, data, highlightColor, baseColor) {
    if (!tensor || !tensor.userData.mesh) return;
    const mesh = tensor.userData.mesh;
    const coordsList = mesh.userData.coords;
    if (!coordsList) return;
    const count = mesh.count;
    const hl = (highlightColor instanceof THREE.Color) ? highlightColor : new THREE.Color(highlightColor);
    const base = (baseColor instanceof THREE.Color) ? baseColor : new THREE.Color(baseColor);
    let isHighlighted = () => false;
    if (data && data.type === 'descriptor') {
        const { start, shape } = data;
        const [sx, sy, sz] = start || [0,0,0], [dx, dy, dz] = shape || [0,0,0];
        const ex = sx+dx, ey = sy+dy, ez = sz+dz;
        isHighlighted = (x,y,z) => (x>=sx&&x<ex && y>=sy&&y<ey && z>=sz&&z<ez);
    } else if (data && Array.isArray(data.data)) {
        const set = new Set(); data.data.forEach(c => set.add(`${c[0]},${c[1]},${c[2]}`));
        isHighlighted = (x,y,z) => set.has(`${x},${y},${z}`);
    }
    const highlightSet = new Set();
    for (let i=0; i<count; i++) {
        const c = coordsList[i];
        if (c && isHighlighted(c[0],c[1],c[2])) { mesh.setColorAt(i, hl); highlightSet.add(i); }
        else { mesh.setColorAt(i, base); }
    }
    mesh.instanceColor.needsUpdate = true;
}

export function calculateTensorSize(shape) {
    let d, h, w; if (shape.length===1) { w=shape[0]; h=1; d=1; } else if (shape.length===2) { h=shape[0]; w=shape[1]; d=1; } else { [d,h,w]=shape; }
    return new THREE.Vector3(w*(CUBE_SIZE+GAP), h*(CUBE_SIZE+GAP), d*(CUBE_SIZE+GAP));
}

export function setupCamera(scene, camera) {
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI/180);
    let cameraZ = Math.abs(maxDim/2/Math.tan(fov/2)) * 1.5;
    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center); return { center, cameraZ };
}

export function setupEventListeners(container, camera, renderer, onMouseMove, onKeyDown, onRender) {
    container.addEventListener('mousemove', onMouseMove);
    window.addEventListener('keydown', (e) => { onKeyDown(e); if (onRender) onRender(); });
    container.addEventListener('wheel', (e) => { e.preventDefault(); camera.position.z += e.deltaY*0.005; camera.updateProjectionMatrix(); if (onRender) onRender(); }, { passive: false });
}

export function cameraControls(camera, cameraRotation) {
    const PAN = 0.1, TILT = 0.02, ZOOM = 0.5;
    return function(e) {
        switch (e.key.toLowerCase()) {
            case 'w': camera.position.y += PAN; break; case 's': camera.position.y -= PAN; break;
            case 'a': camera.position.x -= PAN; break; case 'd': camera.position.x += PAN; break;
            case 'arrowup': cameraRotation.x -= TILT; break; case 'arrowdown': cameraRotation.x += TILT; break;
            case 'arrowleft': cameraRotation.y -= TILT; break; case 'arrowright': cameraRotation.y += TILT; break;
            case 'o': camera.position.z += ZOOM; break; case 'p': camera.position.z -= ZOOM; break;
        }
        camera.setRotationFromEuler(cameraRotation); camera.updateProjectionMatrix();
    };
}

export function addLabels(scene, globalTensor, sliceTensor, colorOrBg = '#ffffff') {
    const sprites = [];
    sprites.push(addLabel(scene, "Global Tensor", globalTensor.position, colorOrBg));
    sprites.push(...addAxisLabels(scene, globalTensor, colorOrBg, globalTensor.userData.color));
    return sprites;
}

function addAxisLabels(scene, tensor, colorOrBg, overrideColor) {
    const groups = []; const shape = tensor?.userData?.mesh?.userData?.shape; if (!shape) return groups;
    const bbox = new THREE.Box3().setFromObject(tensor);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const AXIS_COLORS = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', AXIS_COLORS.x, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${shape.height}`, 'y', AXIS_COLORS.y, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${shape.depth}`, 'z', AXIS_COLORS.z, { offset: offsetBase }));
    return groups;
}

function computeLabelPalette(colorOrBg) {
    let l = 0; try { const c = (colorOrBg instanceof THREE.Color) ? colorOrBg : new THREE.Color(colorOrBg || 0x000000); l = 0.2126*c.r + 0.7152*c.g + 0.0722*c.b; } catch(e){}
    return l > 0.55 ? { fill: '#111111', stroke: '#f8fafc' } : { fill: '#ffffff', stroke: '#0f172a' };
}

function addLabel(scene, text, position, colorOrBg) {
    const { fill, stroke } = computeLabelPalette(colorOrBg);
    const vectorText = createVectorText(text, fill, { fontSize: 0.8, depthTest: false, strokeWidth: 0.03, strokeColor: stroke });
    vectorText.position.set(position.x, position.y + 2, position.z); scene.add(vectorText); return vectorText;
}
