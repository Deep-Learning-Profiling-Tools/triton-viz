import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';



export function createMatMulVisualization(containerElement, op) {
    const { input_shape, other_shape, output_shape } = op;

    const CUBE_SIZE = 0.2;
    const GAP = 0.05;

    // Colors
    const COLOR_A = new THREE.Color(0.53, 0.81, 0.98);  // Light Blue
    const COLOR_B = new THREE.Color(1.0, 0.65, 0.0);    // Orange
    const COLOR_C = new THREE.Color(1.0, 1.0, 1.0);     // White
    const COLOR_HIGHLIGHT = new THREE.Color(0.0, 0.0, 1.0);  // Blue (for highlighting)
    const COLOR_FILLED = new THREE.Color(0.0, 0.0, 1.0);  // Blue (for filled elements in C)
    const COLOR_BACKGROUND = new THREE.Color(0, 0, 0);  // Black
    const COLOR_EDGE = new THREE.Color(0.3, 0.3, 0.3);  // Light Gray (for cube edges)

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = COLOR_BACKGROUND;
    const camera = new THREE.PerspectiveCamera(45, containerElement.clientWidth / containerElement.clientHeight, 0.1, 1000);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    containerElement.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
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

    function createMatrix(dimensions, position, color) {
        const matrix = new THREE.Group();
        matrix.userData.dimensions = dimensions;
        for (let i = 0; i < dimensions[0]; i++) {
            for (let j = 0; j < dimensions[1]; j++) {
                const cube = createCube(color);
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

    // Create matrices
    const matrixA = createMatrix(input_shape, new THREE.Vector3(-10, 10, 0), COLOR_A);
    const matrixB = createMatrix(other_shape, new THREE.Vector3(0, 10, 0), COLOR_B);
    const matrixC = createMatrix(output_shape, new THREE.Vector3(-5, -4, 0), COLOR_C);

    scene.add(matrixA);
    scene.add(matrixB);
    scene.add(matrixC);

    // Center camera
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    const box = new THREE.Box3().setFromObject(scene);
    box.getCenter(center);
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5; // Zoom out a little so objects don't fill the screen

    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center);

    // Animation control
    let isPaused = false;
    let frame = 0;
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
    }

    function animate() {
        requestAnimationFrame(animate);

        if (!isPaused && frame < totalFrames) {
            resetColors();

            const row = Math.floor(frame / other_shape[1]);
            const col = frame % other_shape[1];

            // Highlight entire row in A and entire column in B
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

    // Handle container resize
    function onResize() {
        camera.aspect = containerElement.clientWidth / containerElement.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerElement.clientWidth, containerElement.clientHeight);
    }

    // Create control panel
    const controlPanel = document.createElement('div');
    controlPanel.style.position = 'absolute';
    controlPanel.style.bottom = '10px';
    controlPanel.style.left = '10px';
    controlPanel.style.display = 'flex';
    controlPanel.style.gap = '10px';

    // Add controls
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

    const closeButton = document.createElement('button');
    closeButton.textContent = 'Close';
    closeButton.addEventListener('click', () => {
        cleanup();
    });

    controlPanel.appendChild(playPauseButton);
    controlPanel.appendChild(resetButton);
    controlPanel.appendChild(closeButton);

    containerElement.appendChild(controlPanel);

    // Keyboard controls
    function onKeyDown(event) {
        switch (event.key) {
            case 'ArrowLeft':
                camera.position.x -= 0.5;
                break;
            case 'ArrowRight':
                camera.position.x += 0.5;
                break;
            case 'ArrowUp':
                camera.position.y += 0.5;
                break;
            case 'ArrowDown':
                camera.position.y -= 0.5;
                break;
        }
        camera.lookAt(center);
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('keydown', onKeyDown);

    // Start animation
    animate();

    // Cleanup function
    function cleanup() {
        window.removeEventListener('resize', onResize);
        window.removeEventListener('keydown', onKeyDown);
        containerElement.innerHTML = '';
    }

    // Return cleanup function
    return cleanup;
}