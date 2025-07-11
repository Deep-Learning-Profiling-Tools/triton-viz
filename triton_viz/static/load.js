import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.module.js';
import {
    setupScene,
    setupGeometries,
    createTensor,
    calculateTensorSize,
    updateCubeColor,
    setupCamera,
    setupEventListeners,
    cameraControls
} from './load_utils.js';

export function createLoadVisualization(containerElement, op) {

        console.log(op.uuid);
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
        let frame = 0;
        let isPaused = false;

        const sideMenu = createSideMenu(containerElement);
        let hoveredCube = null;

        const COLOR_GLOBAL = new THREE.Color(0.2, 0.2, 0.2);    // Dark Gray
        const COLOR_SLICE = new THREE.Color(0.0, 0.7, 1.0);     // Cyan (starting color for global slice)
        const COLOR_LEFT_SLICE = new THREE.Color(1.0, 0.0, 1.0); // Magenta (starting color for left slice)
        const COLOR_LOADED = new THREE.Color(1.0, 0.8, 0.0);    // Gold (final color for both slices)
        const COLOR_BACKGROUND = new THREE.Color(0.0, 0.0, 0.0);  // Black

        const { scene, camera, renderer } = setupScene(containerElement, COLOR_BACKGROUND);
        const { cubeGeometry, edgesGeometry, lineMaterial } = setupGeometries();

        const globalTensor = createTensor(op.global_shape, op.global_coords, COLOR_GLOBAL, 'Global', cubeGeometry, edgesGeometry, lineMaterial);
        const sliceTensor = createTensor(op.slice_shape, op.slice_coords, COLOR_LEFT_SLICE, 'Slice', cubeGeometry, edgesGeometry, lineMaterial);

        // Position slice tensor
        const globalSize = calculateTensorSize(op.global_shape);
        sliceTensor.position.set(globalSize.x + 5, 0, 0); // Adjusted tensor spacing

        scene.add(globalTensor);
        scene.add(sliceTensor);

        addLabels(scene, globalTensor, sliceTensor);
        setupCamera(scene, camera);

        const totalFrames = op.global_coords.length * 2 + 30;

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        const onKeyDown = cameraControls(camera, new THREE.Euler(0, 0, 0, 'YXZ'));
        setupEventListeners(containerElement, camera, renderer, onMouseMove, onKeyDown);
        animate();

        async function onMouseMove(event) {
            mouse.x = (event.clientX / containerElement.clientWidth) * 2 - 1;
            mouse.y = -(event.clientY / containerElement.clientHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            const allTensorChildren = [
                ...globalTensor.children,
                ...sliceTensor.children
            ];

            const intersects = raycaster.intersectObjects(allTensorChildren, true);

            if (hoveredCube) {
                hoveredCube.getObjectByName('hoverOutline').visible = false;
                hoveredCube = null;
            }

            if (intersects.length > 0) {
                hoveredCube = intersects[0].object;
                while (hoveredCube && !hoveredCube.tensorName) {
                    hoveredCube = hoveredCube.parent;
                }

                if (hoveredCube) {
                    const hoverOutline = hoveredCube.getObjectByName('hoverOutline');
                    if (hoverOutline) {
                        hoverOutline.visible = true;
                    }

                    updateSideMenu(hoveredCube.tensorName, hoveredCube.tensor0, hoveredCube.tensor1, hoveredCube.tensor2, undefined);

                    const res = await getElementValue(hoveredCube.tensorName, hoveredCube.tensor0, hoveredCube.tensor1, hoveredCube.tensor2);

                    updateSideMenu(hoveredCube.tensorName, hoveredCube.tensor0, hoveredCube.tensor1, hoveredCube.tensor2, res.value);

                    console.log(`Value: ${res.value}`);
                }
            } else {
                updateSideMenu(null);
            }
        }

        function animate() {
            requestAnimationFrame(animate);

            if (!isPaused && frame < totalFrames) {
                const index = Math.floor(frame / 2);
                const factor = (frame % 2) / 1.0;

                if (index < op.global_coords.length) {
                    const globalCoord = op.global_coords[index];
                    const sliceCoord = op.slice_coords[index];

                    updateCubeColor(globalTensor, globalCoord, COLOR_GLOBAL, COLOR_SLICE, factor);
                    updateCubeColor(sliceTensor, sliceCoord, COLOR_LEFT_SLICE, COLOR_LOADED, factor);

                    highlightCurrentOperation(globalTensor, globalCoord, sliceTensor, sliceCoord);
                    updateInfoPanel(globalCoord, sliceCoord, index);
                }

                frame++;
            }

            renderer.render(scene, camera);
        }

        function highlightCurrentOperation(globalTensor, globalCoord, sliceTensor, sliceCoord) {
            globalTensor.children.forEach(cube => cube.material.emissive.setHex(0x000000));
            sliceTensor.children.forEach(cube => cube.material.emissive.setHex(0x000000));

            const globalCube = globalTensor.children.find(c =>
                c.tensor0 === globalCoord[0] && c.tensor1 === globalCoord[1] && c.tensor2 === globalCoord[2]
            );
            const sliceCube = sliceTensor.children.find(c =>
                c.tensor0 === sliceCoord[0] && c.tensor1 === sliceCoord[1] && c.tensor2 === sliceCoord[2]
            );

            if (globalCube) globalCube.material.emissive.setHex(0x444444);
            if (sliceCube) sliceCube.material.emissive.setHex(0x444444);
        }

        async function getElementValue(tensorName, x, y, z) {
            let uuid = op.uuid;
            const response = await fetch('/api/getLoadValue', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ uuid, tensorName, x, y, z }),
            });
            return await response.json();
        }

        function updateSideMenu(tensorName, x, y, z, value) {
            if (!tensorName) {
                sideMenu.innerHTML = '';
                return;
            }

            let dims = tensorName === 'Global' ? op.global_shape : op.slice_shape;
            sideMenu.innerHTML = `
                <h3 style="margin-top: 0;">${tensorName} Tensor</h3>
                <p>X: ${x + 1}</p>
                <p>Y: ${y + 1}</p>
                <p>Z: ${z + 1}</p>
                <p>Dimensions: ${dims.join(' x ')}</p>
                <p>Value: ${value !== undefined ? value : 'Loading...'}</p>
            `;
        }

        function updateInfoPanel(globalCoord, sliceCoord, index) {
            sideMenu.innerHTML = `
                <h3>Current Operation</h3>
                <p>Global Coords: (${globalCoord.join(', ')})</p>
                <p>Slice Coords: (${sliceCoord.join(', ')})</p>
                <p>Progress: ${index + 1}/${op.global_coords.length}</p>
            `;
        }

        function createSideMenu(container) {
            const menu = document.createElement('div');
            menu.style.position = 'absolute';
            menu.style.top = '10px';
            menu.style.right = '10px';
            menu.style.width = '200px';
            menu.style.padding = '10px';
            menu.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
            menu.style.color = 'white';
            menu.style.fontFamily = 'Arial, sans-serif';
            menu.style.fontSize = '14px';
            menu.style.borderRadius = '5px';
            container.appendChild(menu);
            return menu;
        }

        function addLabels(scene, globalTensor, sliceTensor) {
            addLabel(scene, "Global Tensor", globalTensor.position);
            addLabel(scene, "Slice Tensor", sliceTensor.position);
        }

        function addLabel(scene, text, position) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            context.font = 'Bold 24px Arial';
            context.fillStyle = 'white';
            context.fillText(text, 0, 24);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.position.set(position.x, position.y + 2, position.z);
            sprite.scale.set(4, 2, 1);
            scene.add(sprite);
        }

}
