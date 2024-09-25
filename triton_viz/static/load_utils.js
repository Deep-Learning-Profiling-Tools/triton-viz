import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';

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

    // Add custom properties to store tensor coordinates
    cube.userData.tensor0 = z;
    cube.userData.tensor1 = y;
    cube.userData.tensor2 = x;
    cube.userData.tensorName = tensorName;

    cube.name = `${tensorName}_cube_${x}_${y}_${z}`;

    return cube;
}

export function createTensor(shape, coords, color, tensorName, cubeGeometry, edgesGeometry, lineMaterial) {
    console.log(`Creating ${tensorName} tensor:`, shape, coords);
    const tensor = new THREE.Group();
    let [width, height, depth] = shape;
    depth = depth || 1;
    height = height || 1;

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

        console.log(`Highlighting ${coords.length} coordinates in global tensor`);
        coords.forEach(([x, y, z]) => {
            const cube = tensor.children.find(c =>
                c.userData.tensor0 === x && c.userData.tensor1 === y && c.userData.tensor2 === z
            );
            if (cube) {
                cube.material.color.set(COLOR_SLICE);
                console.log(`Highlighted cube at (${x}, ${y}, ${z})`);
            } else {
                console.warn(`Could not find cube at (${x}, ${y}, ${z})`);
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
    const [width, height, depth] = shape;
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
        c.tensor0 === coord[0] && c.tensor1 === coord[1] && c.tensor2 === coord[2]
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
