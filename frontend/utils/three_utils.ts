import { createCadDimension, createVectorText } from './dimension_utils.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import type { TensorHighlights } from '../types/types.js';

type TensorShape = number[];
type TensorCoords = [number, number, number];
type ColorInput = any;
type HighlightPredicate = (x: number, y: number, z: number) => boolean;
type ThreeScene = any;
type ThreeCamera = any;
type ThreeRenderer = any;
type ThreeGeometry = any;
type ThreeMaterial = any;
type ThreeGroup = any;
type ThreeVector3 = any;

export const CUBE_SIZE = 0.2;
export const GAP = 0.05;
export const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);
export const COLOR_EDGE = new THREE.Color(0.5, 0.5, 0.5);

const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);

export function setupScene(container: HTMLElement, backgroundColor = 0x000000): {
    scene: ThreeScene;
    camera: ThreeCamera;
    renderer: ThreeRenderer;
} {
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

export function setupGeometries(): {
    cubeGeometry: ThreeGeometry;
    edgesGeometry: ThreeGeometry;
    lineMaterial: ThreeMaterial;
} {
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

export function createTensor(
    shape: TensorShape,
    coords: TensorCoords[] | null,
    color: ColorInput,
    tensorName: string,
    cubeGeometry: ThreeGeometry,
    _edgesGeometry: ThreeGeometry | null = null,
    _lineMaterial: ThreeMaterial | null = null,
): ThreeGroup {
    console.log(`Creating ${tensorName} tensor:`, shape, coords);
    const tensor = new THREE.Group();
    let depth = 1, height = 1, width = 1;
    if (shape.length === 1) { width = shape[0] ?? 1; }
    else if (shape.length === 2) { height = shape[0] ?? 1; width = shape[1] ?? 1; }
    else { depth = shape[0] ?? 1; height = shape[1] ?? 1; width = shape[2] ?? 1; }

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
    const highlightedIndices = new Set<number>();

    if (isDense) {
        let best = [0, 1, 2];
        if (coords && coords.length > 0) {
            const samples = coords.slice(0, Math.min(256, coords.length));
            const maxIncoming = [0, 0, 0];
            samples.forEach((c: TensorCoords) => {
                const [cx = 0, cy = 0, cz = 0] = c;
                if (cx > (maxIncoming[0] ?? 0)) maxIncoming[0] = cx;
                if (cy > (maxIncoming[1] ?? 0)) maxIncoming[1] = cy;
                if (cz > (maxIncoming[2] ?? 0)) maxIncoming[2] = cz;
            });
            const target = [width - 1, height - 1, depth - 1];
            const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
            const score = (p: number[]): number => {
                let s = 0;
                for (let i = 0; i < 3; i++) {
                    const index = p[i] ?? 0;
                    const targetVal = target[index] ?? 0;
                    const maxVal = maxIncoming[i] ?? 0;
                    s += Math.abs(maxVal - targetVal);
                    if (maxVal > targetVal) s += 1000;
                }
                return s;
            };
            let bestScore = score(best);
            perms.forEach((p: number[]) => { const sc=score(p); if(sc<bestScore){ best=p; bestScore=sc; } });
            const remap = (c: TensorCoords): TensorCoords => [
                c[best.indexOf(0)] ?? 0,
                c[best.indexOf(1)] ?? 0,
                c[best.indexOf(2)] ?? 0,
            ];
            coords.forEach((c: TensorCoords) => { const [x,y,z]=remap(c); if(x>=0&&x<width&&y>=0&&y<height&&z>=0&&z<depth) highlightedIndices.add(z*(width*height)+y*width+x); });
        }
        const coordList: TensorCoords[] = [];
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
        cArr.forEach(([x, y, z]: TensorCoords, idx: number) => {
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

export function updateTensorHighlights(
    tensor: ThreeGroup,
    data: TensorHighlights | null | undefined,
    highlightColor: ColorInput,
    baseColor: ColorInput,
    matchCoords: HighlightPredicate | null = null,
): void {
    if (!tensor || !tensor.userData.mesh) return;
    const mesh = tensor.userData.mesh;
    const coordsList: TensorCoords[] | undefined = mesh.userData.coords;
    if (!coordsList) return;
    const count = mesh.count;
    const hl = (highlightColor instanceof THREE.Color) ? highlightColor : new THREE.Color(highlightColor);
    const base = (baseColor instanceof THREE.Color) ? baseColor : new THREE.Color(baseColor);
    let isHighlighted: HighlightPredicate = (_x, _y, _z) => false;
    if (data && data.type === 'descriptor') {
        const { start, shape } = data;
        const sx = start?.[0] ?? 0;
        const sy = start?.[1] ?? 0;
        const sz = start?.[2] ?? 0;
        const dx = shape?.[0] ?? 0;
        const dy = shape?.[1] ?? 0;
        const dz = shape?.[2] ?? 0;
        const ex = sx+dx, ey = sy+dy, ez = sz+dz;
        isHighlighted = (x,y,z) => (x>=sx&&x<ex && y>=sy&&y<ey && z>=sz&&z<ez);
    } else if (data && Array.isArray(data.data)) {
        const set = new Set<string>(); data.data.forEach((c: number[]) => { const [x=0,y=0,z=0]=c; set.add(`${x},${y},${z}`); });
        isHighlighted = (x,y,z) => set.has(`${x},${y},${z}`);
    } else if (typeof matchCoords === 'function') {
        isHighlighted = matchCoords;
    }
    const highlightSet = new Set();
    for (let i=0; i<count; i++) {
        const c = coordsList[i];
        if (c && isHighlighted(c[0],c[1],c[2])) { mesh.setColorAt(i, hl); highlightSet.add(i); }
        else { mesh.setColorAt(i, base); }
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

export function calculateTensorSize(shape: TensorShape): ThreeVector3 {
    let d = 1, h = 1, w = 1; if (shape.length===1) { w=shape[0] ?? 1; } else if (shape.length===2) { h=shape[0] ?? 1; w=shape[1] ?? 1; } else { d=shape[0] ?? 1; h=shape[1] ?? 1; w=shape[2] ?? 1; }
    return new THREE.Vector3(w*(CUBE_SIZE+GAP), h*(CUBE_SIZE+GAP), d*(CUBE_SIZE+GAP));
}

export function setupCamera(scene: ThreeScene, camera: ThreeCamera): { center: ThreeVector3; cameraZ: number } {
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI/180);
    let cameraZ = Math.abs(maxDim/2/Math.tan(fov/2)) * 1.5;
    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center); return { center, cameraZ };
}

export function setupEventListeners(
    container: HTMLElement,
    camera: ThreeCamera,
    renderer: ThreeRenderer,
    onMouseMove: (event: MouseEvent) => void,
    onKeyDown: (event: KeyboardEvent) => void,
    onRender?: () => void,
): () => void {
    const wheelOptions = { passive: false } as AddEventListenerOptions;
    const handleMouseMove = (event: MouseEvent) => onMouseMove(event);
    const handleKeyDown = (event: KeyboardEvent) => { onKeyDown(event); if (onRender) onRender(); };
    const handleWheel = (event: WheelEvent) => {
        event.preventDefault();
        camera.position.z += event.deltaY * 0.005;
        camera.updateProjectionMatrix();
        if (onRender) onRender();
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

export function cameraControls(camera: ThreeCamera, cameraRotation: any): (e: KeyboardEvent) => void {
    const PAN = 0.1, TILT = 0.02, ZOOM = 0.5;
    return function(e: KeyboardEvent): void {
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

export function addLabels(
    scene: ThreeScene,
    globalTensor: ThreeGroup,
    sliceTensor: ThreeGroup,
    colorOrBg: ColorInput = '#ffffff',
): any[] {
    const sprites = [];
    sprites.push(addLabel(scene, "Global Tensor", globalTensor.position, colorOrBg));
    sprites.push(...addAxisLabels(scene, globalTensor, colorOrBg, globalTensor.userData.color));
    return sprites;
}

function addAxisLabels(
    scene: ThreeScene,
    tensor: ThreeGroup,
    colorOrBg: ColorInput,
    overrideColor: ColorInput,
): any[] {
    const groups: any[] = []; const shape = tensor?.userData?.mesh?.userData?.shape; if (!shape) return groups;
    const bbox = new THREE.Box3().setFromObject(tensor);
    const offsetBase = (CUBE_SIZE + GAP) * 1.5;
    const AXIS_COLORS = { x: '#f87171', y: '#4ade80', z: '#60a5fa' };
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z), `${shape.width}`, 'x', AXIS_COLORS.x, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z), `${shape.height}`, 'y', AXIS_COLORS.y, { offset: offsetBase }));
    groups.push(createCadDimension(scene, new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z), new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z), `${shape.depth}`, 'z', AXIS_COLORS.z, { offset: offsetBase }));
    return groups;
}

function computeLabelPalette(colorOrBg: ColorInput): { fill: string; stroke: string } {
    let l = 0; try { const c = (colorOrBg instanceof THREE.Color) ? colorOrBg : new THREE.Color(colorOrBg || 0x000000); l = 0.2126*c.r + 0.7152*c.g + 0.0722*c.b; } catch(e){}
    return l > 0.55 ? { fill: '#111111', stroke: '#f8fafc' } : { fill: '#ffffff', stroke: '#0f172a' };
}

function addLabel(scene: ThreeScene, text: string, position: ThreeVector3, colorOrBg: ColorInput): any {
    const { fill, stroke } = computeLabelPalette(colorOrBg);
    const vectorText = createVectorText(text, fill, { fontSize: 0.8, depthTest: false, strokeWidth: 0.03, strokeColor: stroke });
    vectorText.position.set(position.x, position.y + 2, position.z); scene.add(vectorText); return vectorText;
}
