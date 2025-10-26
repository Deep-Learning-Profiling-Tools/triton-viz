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
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    // Honour device pixel ratio to align raycaster with drawn pixels
    const dpr = (window.devicePixelRatio || 1);
    renderer.setPixelRatio(dpr);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    return { scene, camera, renderer };
}

export function setupGeometries() {
    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const edgesGeometry = new THREE.EdgesGeometry(cubeGeometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: COLOR_EDGE });
    return { cubeGeometry, edgesGeometry, lineMaterial };
}

export function createCube(color, tensorName, x, y, z, cubeGeometry, edgesGeometry, lineMaterial) {
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

    // Add custom properties to store tensor coordinates (x, y, z)
    cube.userData.tensor0 = x;
    cube.userData.tensor1 = y;
    cube.userData.tensor2 = z;
    cube.userData.tensorName = tensorName;

    cube.name = `${tensorName}_cube_${x}_${y}_${z}`;

    return cube;
}

export function createTensor(shape, coords, color, tensorName, cubeGeometry, edgesGeometry, lineMaterial) {
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

    if (tensorName === 'Global') {
        console.log(`Creating global tensor with dimensions: ${width}x${height}x${depth}`);
        for (let z = 0; z < depth; z++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const cube = createCube(color, tensorName, x, y, z, cubeGeometry, edgesGeometry, lineMaterial);
                    cube.position.set(
                        x * (CUBE_SIZE + GAP),
                        -y * (CUBE_SIZE + GAP),
                        -z * (CUBE_SIZE + GAP)
                    );
                    tensor.add(cube);
                }
            }
        }
        // Build deterministic index from placement order to avoid mismatch
        const indexOf = (x, y, z) => z * (width * height) + y * width + x;

        // Auto-detect coordinate axis order from incoming coords (try first N samples)
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
            const idx = indexOf(x, y, z);
            const cube = tensor.children[idx];
            if (cube && cube.userData && cube.userData.tensor0 === x && cube.userData.tensor1 === y && cube.userData.tensor2 === z) {
                cube.material.color.set(COLOR_SLICE);
            } else {
                console.warn(`Could not find cube at (${A}, ${B}, ${C}) -> mapped (${x}, ${y}, ${z})`);
            }
        });
    } else {
        console.log(`Creating slice tensor with ${coords.length} coordinates`);
        coords.forEach(([x, y, z]) => {
            const cube = createCube(color, tensorName, x, y, z, cubeGeometry, edgesGeometry, lineMaterial);
            cube.position.set(
                x * (CUBE_SIZE + GAP),
                -y * (CUBE_SIZE + GAP),
                -z * (CUBE_SIZE + GAP)
            );
            tensor.add(cube);
        });
    }

    console.log(`Created ${tensorName} tensor with ${tensor.children.length} cubes`);
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

export function setupEventListeners(containerElement, camera, renderer, onMouseMove, onKeyDown) {
    window.addEventListener('resize', () => {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    });
    containerElement.addEventListener('mousemove', onMouseMove);
    window.addEventListener('keydown', onKeyDown);

    // Mouse wheel zoom
    const WHEEL_ZOOM_SPEED = 0.5;
    containerElement.addEventListener('wheel', (event) => {
        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        camera.position.z += direction * WHEEL_ZOOM_SPEED;
        camera.updateProjectionMatrix();
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
