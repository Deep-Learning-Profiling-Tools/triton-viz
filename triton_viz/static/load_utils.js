import * as THREE from 'https://esm.sh/three@0.155.0/build/three.module.js';

export const CUBE_SIZE = 0.2;
export const GAP = 0.05;
export const COLOR_HOVER = new THREE.Color(1.0, 1.0, 0.0);     // Yellow for hover effect
export const COLOR_EDGE = new THREE.Color(0.5, 0.5, 0.5);      // Gray (for cube edges)

const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (starting color for global slice)

export function setupScene(container, backgroundColor = 0x000000) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    // cap device pixel ratio to reduce fillrate
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // 确保输出在 sRGB 空间里渲染，避免颜色被 gamma 压暗
    try {
        if ('outputEncoding' in renderer) {
            renderer.outputEncoding = THREE.sRGBEncoding;
        }
        if ('outputColorSpace' in renderer) {
            renderer.outputColorSpace = THREE.SRGBColorSpace;
        }
        renderer.toneMapping = THREE.NoToneMapping;
    } catch (e) {
        // 忽略旧版 three 的不兼容字段
    }

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
    // 细且半透明的边线材质；WebGL 会 clamp 线宽，但通过降低不透明度来让线"看起来更细"
    const lineMaterial = new THREE.LineBasicMaterial({
        color: COLOR_EDGE,
        linewidth: 0.1,
        transparent: true,
        opacity: 0.3, // baseline 稍微更实一点，具体不透明度在主题里再微调
    });
    return { cubeGeometry, edgesGeometry, lineMaterial };
}

export function createCube(color, tensorName, x, y, z, cubeGeometry, edgesGeometry, lineMaterial) {
    const cubeMaterial = new THREE.MeshPhongMaterial({ color: color, vertexColors: false });
    const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
    const edges = new THREE.LineSegments(edgesGeometry, lineMaterial);
    edges.name = 'edgeOutline';
    cube.add(edges);
    cube.userData.edges = edges;

    const hoverGeometry = new THREE.BoxGeometry(CUBE_SIZE * 1.05, CUBE_SIZE * 1.05, CUBE_SIZE * 1.05);
    const hoverEdgesGeometry = new THREE.EdgesGeometry(hoverGeometry);
    const hoverOutline = new THREE.LineSegments(hoverEdgesGeometry, new THREE.LineBasicMaterial({ color: COLOR_HOVER }));
    hoverOutline.visible = false;
    hoverOutline.name = 'hoverOutline';
    cube.add(hoverOutline);

    cube.userData.tensor0 = x;
    cube.userData.tensor1 = y;
    cube.userData.tensor2 = z;
    cube.userData.tensorName = tensorName;

    cube.name = `${tensorName}_cube_${x}_${y}_${z}`;

    return cube;
}

export function createTensor(shape, coords, color, tensorName, cubeGeometry) {
    console.log(`Creating ${tensorName} tensor:`, shape, coords);
    const tensor = new THREE.Group();
    // Normalize shape to width (X), height (Y), depth (Z)
    let width, height, depth;
    if (shape.length === 1) {
        width = shape[0];
        height = 1;
        depth = 1;
    } else if (shape.length === 2) {
        // Backend provides (H, W) for 2D tensors; interpret as width=W, height=H
        height = shape[0];
        width = shape[1];
        depth = 1;
    } else {
        // Assume incoming order already matches [width, height, depth]
        [width, height, depth] = shape;
    }

    const spacing = CUBE_SIZE + GAP;
    const centerX = (width - 1) * spacing / 2;
    const centerY = -((height - 1) * spacing / 2);
    const centerZ = -((depth - 1) * spacing / 2);
    const isGlobal = tensorName === 'Global';
    const instanceCount = isGlobal ? width * height * depth : coords.length;
    const cubeMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true });
    const mesh = new THREE.InstancedMesh(cubeGeometry, cubeMaterial, instanceCount);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instanceCount * 3), 3);
    const matrix = new THREE.Matrix4();
    const baseColor = color instanceof THREE.Color ? color.clone() : new THREE.Color(color);
    const highlightedIndices = new Set();

    if (isGlobal) {
        console.log(`Creating global tensor with dimensions: ${width}x${height}x${depth}`);
        // auto-detect coordinate axis order from incoming coords (try first N samples)
        const samples = coords.slice(0, Math.min(256, coords.length));
        const maxIncoming = [0, 0, 0];
        for (const [a, b, c] of samples) {
            if (a > maxIncoming[0]) maxIncoming[0] = a;
            if (b > maxIncoming[1]) maxIncoming[1] = b;
            if (c > maxIncoming[2]) maxIncoming[2] = c;
        }
        const target = [width - 1, height - 1, depth - 1];
        const perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        function scorePerm(p) {
            // sum of absolute diffs; heavy penalty if any exceeds target
            let s = 0;
            for (let i = 0; i < 3; i++) {
                const diff = Math.abs(maxIncoming[i] - target[p[i]]);
                s += diff;
                if (maxIncoming[i] > target[p[i]]) s += 1000; // penalize out-of-range
            }
            return s;
        }
        let best = perms[0];
        let bestScore = scorePerm(best);
        for (let i = 1; i < perms.length; i++) {
            const sc = scorePerm(perms[i]);
            if (sc < bestScore) {
                best = perms[i];
                bestScore = sc;
            }
        }

        const remap = ([a, b, c]) => {
            const arr = [a, b, c];
            return [arr[best.indexOf(0)], arr[best.indexOf(1)], arr[best.indexOf(2)]];
        };

        console.log(`Highlighting ${coords.length} coordinates in global tensor. perm used: ${best}`);
        coords.forEach(([A, B, C]) => {
            const [x, y, z] = remap([A, B, C]);
            if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
                console.warn(`Could not find cube at (${A}, ${B}, ${C}) -> mapped to out-of-range (${x}, ${y}, ${z})`);
                return;
            }
            const idx = z * (width * height) + y * width + x;
            highlightedIndices.add(idx);
        });

        let idx = 0;
        for (let z = 0; z < depth; z++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    matrix.setPosition(x * spacing - centerX, -y * spacing - centerY, -z * spacing - centerZ);
                    mesh.setMatrixAt(idx, matrix);
                    mesh.setColorAt(idx, highlightedIndices.has(idx) ? COLOR_SLICE : baseColor);
                    idx++;
                }
            }
        }
    } else {
        console.log(`Creating slice tensor with ${coords.length} coordinates`);
        coords.forEach(([x, y, z], idx) => {
            matrix.setPosition(x * spacing - centerX, -y * spacing - centerY, -z * spacing - centerZ);
            mesh.setMatrixAt(idx, matrix);
            mesh.setColorAt(idx, baseColor);
        });
        mesh.userData.coords = coords;
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.userData.tensorName = tensorName;
    mesh.userData.shape = { width, height, depth };
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();

    tensor.add(mesh);
    tensor.userData.mesh = mesh;
    tensor.userData.highlightedIndices = highlightedIndices;

    console.log(`Created ${tensorName} tensor with ${instanceCount} cubes`);
    return tensor;
}

export function calculateTensorSize(shape) {
    // Normalize shape for size calculation consistent with createTensor
    let width, height, depth;
    if (shape.length === 1) {
        width = shape[0];
        height = 1;
        depth = 1;
    } else if (shape.length === 2) {
        // (H, W) -> width=W, height=H
        height = shape[0];
        width = shape[1];
        depth = 1;
    } else {
        [width, height, depth] = shape;
    }
    return new THREE.Vector3(
        width * (CUBE_SIZE + GAP),
        height * (CUBE_SIZE + GAP),
        depth * (CUBE_SIZE + GAP)
    );
}

export function interpolateColor(color1, color2, factor) {
    return new THREE.Color().lerpColors(color1, color2, factor);
}

export function updateCubeColor(tensor, coord, startColor, endColor, factor) {
    const cube = tensor.children.find(c =>
        c.userData &&
        c.userData.tensor0 === coord[0] &&
        c.userData.tensor1 === coord[1] &&
        c.userData.tensor2 === coord[2]
    );
    if (cube) {
        cube.material.color.copy(interpolateColor(startColor, endColor, factor));
    }
}

export function setupCamera(scene, camera) {
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center);

    return { center, cameraZ };
}

export function setupEventListeners(containerElement, camera, renderer, onMouseMove, onKeyDown, onRender, onCameraChange) {
    window.addEventListener('resize', () => {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
        if (onRender) onRender();
    });
    containerElement.addEventListener('mousemove', onMouseMove);
    window.addEventListener('keydown', (event) => {
        onKeyDown(event);
        if (onCameraChange) onCameraChange();
        if (onRender) onRender();
    });

    // mouse wheel zoom
    const WHEEL_ZOOM_SPEED = 0.5;
    containerElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        camera.position.z += direction * WHEEL_ZOOM_SPEED;
        camera.updateProjectionMatrix();
        if (onCameraChange) onCameraChange();
        if (onRender) onRender();
    }, { passive: false });
}

export function cameraControls(camera, cameraRotation) {
    const PAN_SPEED = 0.1;
    const TILT_SPEED = 0.02;
    const ZOOM_SPEED = 0.5;

    return function onKeyDown(event) {
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
            case ' ':
                break;
        }
        camera.setRotationFromEuler(cameraRotation);
        camera.updateProjectionMatrix();
    };
}

export function addLabels(scene, globalTensor, sliceTensor, colorOrBg = '#ffffff') {
    const sprites = [];
    sprites.push(addLabel(scene, "Global Tensor", globalTensor.position, colorOrBg));
    sprites.push(addLabel(scene, "Slice Tensor", sliceTensor.position, colorOrBg));
    return sprites;
}

function computeLabelPalette(colorOrBg) {
    // 保留原有简洁标签样式，仅根据背景亮度调整文字/描边颜色
    // three.js Color 无 getLuminance，这里手算
    let luminance = 0;
    try {
        const c = (colorOrBg instanceof THREE.Color) ? colorOrBg : new THREE.Color(colorOrBg || 0x000000);
        luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    } catch (e) {
        luminance = 0;
    }
    const isLightBg = luminance > 0.55;
    return isLightBg
        ? { fill: '#111111', stroke: '#f8fafc' }   // 白底用深文字+浅描边
        : { fill: '#ffffff', stroke: '#0f172a' };  // 暗底用亮文字+深描边
}

function addLabel(scene, text, position, colorOrBg) {
    // 轻量文本标签：无背景，仅描边+填充，避免"熊猫眼"效果
    const paddingX = 6;
    const paddingY = 6;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = 'Bold 24px Arial';
    const metrics = ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = 28;
    canvas.width = Math.ceil(textWidth + paddingX * 2);
    canvas.height = Math.ceil(textHeight + paddingY * 2);

    const { fill, stroke } = computeLabelPalette(colorOrBg);
    ctx.font = 'Bold 24px Arial';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.fillStyle = fill;
    const y = canvas.height / 2;
    ctx.strokeText(text, paddingX, y);
    ctx.fillText(text, paddingX, y);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(position.x, position.y + 2, position.z);
    const scaleX = Math.max(1, canvas.width / 64);
    const scaleY = Math.max(0.6, canvas.height / 48);
    sprite.scale.set(scaleX, scaleY, 1);
    scene.add(sprite);
    return sprite;
}
