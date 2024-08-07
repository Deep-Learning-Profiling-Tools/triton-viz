import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';

export function createLoadVisualization(containerElement, op) {
    const CUBE_SIZE = 0.2;
    const GAP = 0.05;

    // Colors
    const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray
    const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (starting color for global slice)
    const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0); // Magenta (starting color for left slice)
    const COLOR_LOADED = new THREE.Color(1.0, 0.8, 0.0);    // Gold (final color for both slices)
    const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black
    const COLOR_EDGE = new THREE.Color(0.5, 0.5, 0.5);      // Gray (for cube edges)

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = COLOR_BACKGROUND;
    const camera = new THREE.PerspectiveCamera(45, containerElement.clientWidth / containerElement.clientHeight, 0.1, 1000);
    camera.position.set(15, -15, 30);
    camera.lookAt(8, -5, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    containerElement.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Create cube geometry and materials
    const cubeGeometry = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    const edgesGeometry = new THREE.EdgesGeometry(cubeGeometry);
    const lineMaterial = new THREE.LineBasicMaterial({ color: COLOR_EDGE });

    function createCube(color) {
        const cubeMaterial = new THREE.MeshPhongMaterial({ color: color });
        const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
        const edges = new THREE.LineSegments(edgesGeometry, lineMaterial);
        cube.add(edges);
        return cube;
    }

    function createTensorPositions(dimensions, offset) {
        const positions = [];
        for (let i = 0; i < dimensions[0]; i++) {
            for (let j = 0; j < dimensions[1]; j++) {
                const x = (j + offset[1]) * (CUBE_SIZE + GAP);
                const y = -(i + offset[0]) * (CUBE_SIZE + GAP);
                const z = offset[2] * (CUBE_SIZE + GAP);
                positions.push(new THREE.Vector3(x, y, z));
            }
        }
        return positions;
    }

    // Create tensors
    const globalPositions = createTensorPositions(op['global_shape'], [0, 0, 0]);
    const leftSlicePositions = createTensorPositions(op['slice_shape'], [-20, 32, 1]);

    const globalTensor = new THREE.Group();
    const leftSliceTensor = new THREE.Group();

    globalPositions.forEach((position, index) => {
        const i = Math.floor(index / op['global_shape'][1]);
        const j = index % op['global_shape'][1];
        const isInSlice = i >= op['slice_coords'][0][0] && i <= op['slice_coords'][2][0] &&
                          j >= op['slice_coords'][0][1] && j <= op['slice_coords'][1][1];
        const color = isInSlice ? COLOR_SLICE : COLOR_GLOBAL;
        const cube = createCube(color);
        cube.position.copy(position);
        globalTensor.add(cube);
    });

    leftSlicePositions.forEach((position) => {
        const cube = createCube(COLOR_LEFT_SLICE);
        cube.position.copy(position);
        leftSliceTensor.add(cube);
    });

    scene.add(globalTensor);
    scene.add(leftSliceTensor);

    // Animation
    let isPaused = false;
    let frame = 0;
    const totalFrames = op['slice_shape'][0] * op['slice_shape'][1] * 2 + 30;

    function interpolateColor(color1, color2, factor) {
        return new THREE.Color().lerpColors(color1, color2, factor);
    }

    function animate() {
        requestAnimationFrame(animate);

        if (!isPaused && frame < totalFrames) {
            const [i, j] = [Math.floor(frame / 2 / op['slice_shape'][1]), Math.floor(frame / 2) % op['slice_shape'][1]];
            const factor = (frame % 2) / 1.0;
            
            const global_i = op['slice_coords'][0][0] + i;
            const global_j = op['slice_coords'][0][1] + j;
            if (global_i < op['global_shape'][0] && global_j < op['global_shape'][1]) {
                const globalIndex = global_i * op['global_shape'][1] + global_j;
                globalTensor.children[globalIndex].material.color.copy(
                    interpolateColor(COLOR_SLICE, COLOR_LOADED, factor)
                );
            }
            
            const leftSliceIndex = i * op['slice_shape'][1] + j;
            leftSliceTensor.children[leftSliceIndex].material.color.copy(
                interpolateColor(COLOR_LEFT_SLICE, COLOR_LOADED, factor)
            );
            
            frame++;
        }

        renderer.render(scene, camera);
    }

    // Handle window resize
    function onWindowResize() {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    }

    // Camera controls
    const cameraSpeed = 0.1;
    const zoomSpeed = 0.5;

    function onKeyDown(event) {
        switch(event.key) {
            case ' ':
                isPaused = !isPaused;
                break;
            case 'ArrowLeft':
                camera.position.x -= cameraSpeed;
                break;
            case 'ArrowRight':
                camera.position.x += cameraSpeed;
                break;
            case 'ArrowUp':
                camera.position.y += cameraSpeed;
                break;
            case 'ArrowDown':
                camera.position.y -= cameraSpeed;
                break;
            case 'o':
                camera.position.z -= zoomSpeed;
                break;
            case 'p':
                camera.position.z += zoomSpeed;
                break;
        }
        camera.lookAt(8, -5, 0);
    }

    containerElement.addEventListener('keydown', onKeyDown);
    window.addEventListener('resize', onWindowResize);

    // Start animation
    animate();

    // Return cleanup function
    return function cleanup() {
        window.removeEventListener('resize', onWindowResize);
        containerElement.removeEventListener('keydown', onKeyDown);
        containerElement.removeChild(renderer.domElement);
    };
}