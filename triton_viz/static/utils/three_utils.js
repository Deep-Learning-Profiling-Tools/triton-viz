import { createCadDimension, createVectorText } from './dimension_utils.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
export const CUBE_SIZE = 0.2;
export const GAP = 0.05;
export const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);
export const COLOR_EDGE = new THREE.Color(0.5, 0.5, 0.5);
const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);
// quick feature test for WebGL availability in the current browser.
export function canUseWebgl() {
    if (typeof document === 'undefined')
        return false;
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    return !!gl;
}
// render a visible warning when WebGL is missing or disabled.
export function renderWebglWarning(container) {
    const existing = container.querySelector('.webgl-warning');
    if (existing) {
        return () => existing.remove();
    }
    const warning = document.createElement('div');
    warning.className = 'webgl-warning info-card';
    warning.setAttribute('role', 'alert');
    warning.setAttribute('aria-live', 'assertive');
    warning.textContent = 'WebGL is required for the 3D visualizer. Enable WebGL in your browser settings and reload this page.';
    container.appendChild(warning);
    return () => warning.remove();
}
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
        if ('outputEncoding' in renderer)
            renderer.outputEncoding = THREE.sRGBEncoding;
        if ('outputColorSpace' in renderer)
            renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.NoToneMapping;
    }
    catch (e) { }
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
        colorArray[base] = shade;
        colorArray[base + 1] = shade;
        colorArray[base + 2] = shade;
    }
    cubeGeometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    const edgesGeometry = new THREE.EdgesGeometry(cubeGeometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: COLOR_EDGE, linewidth: 0.1, transparent: true, opacity: 0.3 });
    return { cubeGeometry, edgesGeometry, lineMaterial };
}
export function createTensor(shape, coords, color, tensorName, cubeGeometry, _edgesGeometry = null, _lineMaterial = null, options = {}) {
    console.log(`Creating ${tensorName} tensor:`, shape, coords);
    const tensor = new THREE.Group();
    const normalizedShape = normalizeTensorShape(shape);
    const { depth, height, width } = shapeDepthHeightWidth(normalizedShape);
    const isGlobal = tensorName === 'Global';
    const isDense = isGlobal || !coords; // Treat as dense if named Global or no coords provided
    const instanceCount = isDense ? productOfShape(normalizedShape) : coords.length;
    const mesh = new THREE.InstancedMesh(cubeGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true }), instanceCount);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);
    const baseColor = color instanceof THREE.Color ? color.clone() : new THREE.Color(color);
    mesh.userData.color_base = baseColor;
    mesh.userData.shape_raw = normalizedShape;
    mesh.userData.tensorName = tensorName;
    mesh.userData.shape = { depth, height, width };
    const matrix = new THREE.Matrix4();
    const highlightedIndices = new Set();
    if (isDense) {
        let best = [0, 1, 2];
        if (coords && coords.length > 0) {
            const samples = coords.slice(0, Math.min(256, coords.length));
            const maxIncoming = [0, 0, 0];
            samples.forEach((c) => {
                const [cx = 0, cy = 0, cz = 0] = c;
                if (cx > (maxIncoming[0] ?? 0))
                    maxIncoming[0] = cx;
                if (cy > (maxIncoming[1] ?? 0))
                    maxIncoming[1] = cy;
                if (cz > (maxIncoming[2] ?? 0))
                    maxIncoming[2] = cz;
            });
            const target = [width - 1, height - 1, depth - 1];
            const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
            const score = (p) => {
                let s = 0;
                for (let i = 0; i < 3; i++) {
                    const index = p[i] ?? 0;
                    const targetVal = target[index] ?? 0;
                    const maxVal = maxIncoming[i] ?? 0;
                    s += Math.abs(maxVal - targetVal);
                    if (maxVal > targetVal)
                        s += 1000;
                }
                return s;
            };
            let bestScore = score(best);
            perms.forEach((p) => { const sc = score(p); if (sc < bestScore) {
                best = p;
                bestScore = sc;
            } });
            const remap = (c) => [
                c[best.indexOf(0)] ?? 0,
                c[best.indexOf(1)] ?? 0,
                c[best.indexOf(2)] ?? 0,
            ];
            coords.forEach((c) => { const [x, y, z] = remap(c); if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth)
                highlightedIndices.add(z * (width * height) + y * width + x); });
        }
        const coordList = [];
        const displayCoords = [];
        const fullCoords = [];
        for (let idx = 0; idx < instanceCount; idx += 1) {
            const displayCoord = unravelIndex(idx, normalizedShape);
            const fullCoord = options.mapDisplayCoordToFull
                ? options.mapDisplayCoordToFull(displayCoord.slice())
                : displayCoord.slice();
            const legacyCoord = legacyCoordFromDisplay(displayCoord);
            const position = positionForDisplayCoord(displayCoord, normalizedShape);
            matrix.setPosition(position.x, position.y, position.z);
            mesh.setMatrixAt(idx, matrix);
            mesh.setColorAt(idx, highlightedIndices.has(idx) ? COLOR_SLICE : baseColor);
            coordList.push(legacyCoord);
            displayCoords.push(displayCoord);
            fullCoords.push(fullCoord);
        }
        mesh.userData.coords = coordList;
        mesh.userData.coords_display = displayCoords;
        mesh.userData.coords_full = fullCoords;
    }
    else {
        const cArr = coords || [];
        const displayCoords = [];
        const fullCoords = [];
        const spacing = CUBE_SIZE + GAP;
        const centerX = (width - 1) * spacing / 2;
        const centerY = -((height - 1) * spacing / 2);
        const centerZ = -((depth - 1) * spacing / 2);
        cArr.forEach(([x, y, z], idx) => {
            matrix.setPosition(x * spacing - centerX, -y * spacing - centerY, -z * spacing - centerZ);
            mesh.setMatrixAt(idx, matrix);
            mesh.setColorAt(idx, baseColor);
            const displayCoord = [z, y, x];
            displayCoords.push(displayCoord);
            fullCoords.push(options.mapDisplayCoordToFull ? options.mapDisplayCoordToFull(displayCoord.slice()) : displayCoord);
        });
        mesh.userData.coords = cArr;
        mesh.userData.coords_display = displayCoords;
        mesh.userData.coords_full = fullCoords;
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor)
        mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    tensor.add(mesh);
    tensor.userData.mesh = mesh;
    tensor.userData.color = baseColor;
    tensor.userData.highlightedIndices = highlightedIndices;
    return tensor;
}
export function updateTensorHighlights(tensor, data, highlightColor, baseColor, matchCoords = null) {
    if (!tensor || !tensor.userData.mesh)
        return;
    const mesh = tensor.userData.mesh;
    const coordsLegacy = mesh.userData.coords;
    const coordsFull = mesh.userData.coords_full || mesh.userData.coords_display || coordsLegacy;
    if (!coordsFull)
        return;
    const count = mesh.count;
    const hl = (highlightColor instanceof THREE.Color) ? highlightColor : new THREE.Color(highlightColor);
    const base = (baseColor instanceof THREE.Color) ? baseColor : new THREE.Color(baseColor);
    let isHighlighted = (_coord, _legacy) => false;
    if (data && data.type === 'descriptor') {
        const { start, shape, stride } = data;
        const rank = Math.max(start?.length || 0, shape?.length || 0, stride?.length || 0, coordsFull[0]?.length || 0);
        const starts = Array.from({ length: rank }, (_, axis) => Number(start?.[axis] ?? 0));
        const shapes = Array.from({ length: rank }, (_, axis) => Number(shape?.[axis] ?? 0));
        const strides = Array.from({ length: rank }, (_, axis) => Math.max(1, Math.abs(Number(stride?.[axis] ?? 1))));
        const inAxis = (coord, axisStart, axisShape, axisStride) => {
            if (axisShape <= 0)
                return false;
            const delta = coord - axisStart;
            if (delta < 0 || delta % axisStride !== 0)
                return false;
            return (delta / axisStride) < axisShape;
        };
        isHighlighted = (coord) => {
            for (let axis = 0; axis < rank; axis += 1) {
                if (!inAxis(coord[axis] ?? 0, starts[axis] ?? 0, shapes[axis] ?? 0, strides[axis] ?? 1))
                    return false;
            }
            return true;
        };
    }
    else if (data && Array.isArray(data.data)) {
        const set = new Set();
        data.data.forEach((c) => set.add(c.join(',')));
        isHighlighted = (coord) => set.has(coord.join(','));
    }
    else if (typeof matchCoords === 'function') {
        isHighlighted = (_coord, legacy) => matchCoords(legacy[0], legacy[1], legacy[2]);
    }
    for (let i = 0; i < count; i += 1) {
        const coord = coordsFull[i] || [];
        const legacy = coordsLegacy?.[i] || [0, 0, 0];
        if (isHighlighted(coord, legacy))
            mesh.setColorAt(i, hl);
        else
            mesh.setColorAt(i, base);
    }
    if (mesh.instanceColor)
        mesh.instanceColor.needsUpdate = true;
}
function normalizeTensorShape(shape) {
    if (!Array.isArray(shape) || shape.length === 0)
        return [1];
    return shape.map((dim) => Math.max(1, Number(dim) || 1));
}
function shapeDepthHeightWidth(shape) {
    const rank = shape.length;
    const width = shape[rank - 1] ?? 1;
    const height = rank >= 2 ? (shape[rank - 2] ?? 1) : 1;
    const depth = rank >= 3 ? (shape[rank - 3] ?? 1) : 1;
    return { depth, height, width };
}
function productOfShape(shape) {
    let count = 1;
    shape.forEach((dim) => {
        count *= Math.max(1, Number(dim) || 1);
    });
    return count;
}
function unravelIndex(index, shape) {
    const coord = new Array(shape.length).fill(0);
    let remainder = index;
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
        const dim = Math.max(1, shape[axis] ?? 1);
        coord[axis] = remainder % dim;
        remainder = Math.floor(remainder / dim);
    }
    return coord;
}
function legacyCoordFromDisplay(coord) {
    const rank = coord.length;
    if (rank === 1)
        return [coord[0] ?? 0, 0, 0];
    if (rank === 2)
        return [coord[1] ?? 0, coord[0] ?? 0, 0];
    return [coord[rank - 1] ?? 0, coord[rank - 2] ?? 0, coord[rank - 3] ?? 0];
}
function positionForDisplayCoord(coord, shape) {
    const baseCell = { x: CUBE_SIZE, y: CUBE_SIZE, z: CUBE_SIZE };
    return recursivePosition(coord, shape, baseCell);
}
function recursivePosition(coord, shape, cellExtent) {
    if (shape.length <= 3)
        return baseGridPosition(coord, shape, cellExtent);
    const split = shape.length - 3;
    const outerShape = shape.slice(0, split);
    const innerShape = shape.slice(split);
    const outerCoord = coord.slice(0, split);
    const innerCoord = coord.slice(split);
    const innerExtent = recursiveExtent(innerShape, cellExtent);
    const outerPos = recursivePosition(outerCoord, outerShape, innerExtent);
    const innerPos = recursivePosition(innerCoord, innerShape, cellExtent);
    return new THREE.Vector3(outerPos.x + innerPos.x, outerPos.y + innerPos.y, outerPos.z + innerPos.z);
}
function recursiveExtent(shape, cellExtent) {
    if (shape.length <= 3)
        return baseGridExtent(shape, cellExtent);
    const split = shape.length - 3;
    const outerShape = shape.slice(0, split);
    const innerShape = shape.slice(split);
    const innerExtent = recursiveExtent(innerShape, cellExtent);
    return recursiveExtent(outerShape, innerExtent);
}
function baseGridExtent(shape, cellExtent) {
    const { depth, height, width } = shapeDepthHeightWidth(shape);
    const stepX = cellExtent.x + GAP;
    const stepY = cellExtent.y + GAP;
    const stepZ = cellExtent.z + GAP;
    return {
        x: (width - 1) * stepX + cellExtent.x,
        y: (height - 1) * stepY + cellExtent.y,
        z: (depth - 1) * stepZ + cellExtent.z,
    };
}
function baseGridPosition(coord, shape, cellExtent) {
    const { depth, height, width } = shapeDepthHeightWidth(shape);
    const stepX = cellExtent.x + GAP;
    const stepY = cellExtent.y + GAP;
    const stepZ = cellExtent.z + GAP;
    const centerX = (width - 1) * stepX / 2;
    const centerY = (height - 1) * stepY / 2;
    const centerZ = (depth - 1) * stepZ / 2;
    const xAxis = coord[shape.length - 1] ?? 0;
    const yAxis = shape.length >= 2 ? (coord[shape.length - 2] ?? 0) : 0;
    const zAxis = shape.length >= 3 ? (coord[shape.length - 3] ?? 0) : 0;
    return new THREE.Vector3(xAxis * stepX - centerX, -yAxis * stepY + centerY, -zAxis * stepZ + centerZ);
}
export function calculateTensorSize(shape) {
    let d = 1, h = 1, w = 1;
    if (shape.length === 1) {
        w = shape[0] ?? 1;
    }
    else if (shape.length === 2) {
        h = shape[0] ?? 1;
        w = shape[1] ?? 1;
    }
    else {
        d = shape[0] ?? 1;
        h = shape[1] ?? 1;
        w = shape[2] ?? 1;
    }
    return new THREE.Vector3(w * (CUBE_SIZE + GAP), h * (CUBE_SIZE + GAP), d * (CUBE_SIZE + GAP));
}
export function fitCameraToBounds(camera, bounds, center, padding = 1.15) {
    const target = center ?? bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const vFov = camera.fov * (Math.PI / 180);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const fitHeight = (size.y / 2) / Math.tan(vFov / 2);
    const fitWidth = (size.x / 2) / Math.tan(hFov / 2);
    const fitDepth = size.z / 2;
    // pick the dominant distance so the whole box fits with a small padding
    const cameraZ = Math.max(fitHeight, fitWidth, fitDepth) * padding;
    camera.position.set(target.x, target.y, target.z + cameraZ);
    // keep clip planes wide enough to avoid slicing big tensors
    camera.near = Math.max(0.1, cameraZ - size.z * 2);
    camera.far = Math.max(camera.far, cameraZ + size.z * 4);
    camera.lookAt(target);
    camera.updateProjectionMatrix();
    return { center: target, cameraZ };
}
export function setupCamera(scene, camera) {
    const box = new THREE.Box3().setFromObject(scene);
    return fitCameraToBounds(camera, box);
}
export function setupEventListeners(container, camera, renderer, onMouseMove, onKeyDown, onRender) {
    const wheelOptions = { passive: false };
    const handleMouseMove = (event) => onMouseMove(event);
    const handleKeyDown = (event) => { onKeyDown(event); if (onRender)
        onRender(); };
    const handleWheel = (event) => {
        event.preventDefault();
        camera.position.z += event.deltaY * 0.005;
        camera.updateProjectionMatrix();
        if (onRender)
            onRender();
    };
    container.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('keydown', handleKeyDown);
    container.addEventListener('wheel', handleWheel, wheelOptions);
    return () => {
        container.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('keydown', handleKeyDown);
        container.removeEventListener('wheel', handleWheel, wheelOptions);
    };
}
export function cameraControls(camera, cameraRotation) {
    const PAN = 0.1, TILT = 0.02, ZOOM = 0.5;
    return function (e) {
        switch (e.key.toLowerCase()) {
            case 'w':
                camera.position.y += PAN;
                break;
            case 's':
                camera.position.y -= PAN;
                break;
            case 'a':
                camera.position.x -= PAN;
                break;
            case 'd':
                camera.position.x += PAN;
                break;
            case 'arrowup':
                cameraRotation.x -= TILT;
                break;
            case 'arrowdown':
                cameraRotation.x += TILT;
                break;
            case 'arrowleft':
                cameraRotation.y -= TILT;
                break;
            case 'arrowright':
                cameraRotation.y += TILT;
                break;
            case 'o':
                camera.position.z += ZOOM;
                break;
            case 'p':
                camera.position.z -= ZOOM;
                break;
        }
        camera.setRotationFromEuler(cameraRotation);
        camera.updateProjectionMatrix();
    };
}
export function addLabels(scene, globalTensor, sliceTensor, colorOrBg = '#ffffff') {
    const sprites = [];
    sprites.push(addLabel(scene, "Global Tensor", globalTensor.position, colorOrBg));
    sprites.push(...addAxisLabels(scene, globalTensor, colorOrBg, globalTensor.userData.color));
    return sprites;
}
function addAxisLabels(scene, tensor, colorOrBg, overrideColor) {
    const groups = [];
    const shape = tensor?.userData?.mesh?.userData?.shape;
    if (!shape)
        return groups;
    const bbox = new THREE.Box3().setFromObject(tensor);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const AXIS_COLORS = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', AXIS_COLORS.x, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${shape.height}`, 'y', AXIS_COLORS.y, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${shape.depth}`, 'z', AXIS_COLORS.z, { offset: offsetBase }));
    return groups;
}
function computeLabelPalette(colorOrBg) {
    let l = 0;
    try {
        const c = (colorOrBg instanceof THREE.Color) ? colorOrBg : new THREE.Color(colorOrBg || 0x000000);
        l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    }
    catch (e) { }
    return l > 0.55 ? { fill: '#111111', stroke: '#f8fafc' } : { fill: '#ffffff', stroke: '#0f172a' };
}
function addLabel(scene, text, position, colorOrBg) {
    const { fill, stroke } = computeLabelPalette(colorOrBg);
    const vectorText = createVectorText(text, fill, { fontSize: 0.8, depthTest: false, strokeWidth: 0.03, strokeColor: stroke });
    vectorText.position.set(position.x, position.y + 2, position.z);
    scene.add(vectorText);
    return vectorText;
}
