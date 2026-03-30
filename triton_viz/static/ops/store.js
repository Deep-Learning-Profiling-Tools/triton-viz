import { createTensorVisualization } from '../components/tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import { tensorBoundsSizeForShape } from '../utils/three_utils.js';
export function createStoreVisualization(containerElement, op, viewState = null) {
    const globalSize = tensorBoundsSizeForShape(op.global_shape || []);
    const sliceSize = tensorBoundsSizeForShape(op.slice_shape || []);
    const gap = 1.5;
    const sliceX = globalSize.x / 2 + sliceSize.x / 2 + gap;
    return createTensorVisualization(containerElement, op, {
        type: 'Store',
        tensorConfigs: [
            { name: 'Global', shape: op.global_shape || [], color: new THREE.Color(0.2, 0.2, 0.2), position: [0, 0, 0], endpoint: 'getLoadTensor', source: 'GLOBAL' },
            { name: 'Slice', shape: op.slice_shape || [], color: new THREE.Color(0.95, 0.2, 0.95), position: [sliceX, 0, 0], endpoint: 'getLoadTensor', source: 'SLICE' },
        ],
        colors: {
            GLOBAL: new THREE.Color(0.2, 0.2, 0.2),
            HIGHLIGHT: new THREE.Color(1.0, 0.55, 0.0)
        },
        viewState
    });
}
