import { createTensorVisualization } from '../components/tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
export function createTransferVisualization(containerElement, op, viewState = null) {
    const isStoreLike = (op.mem_dst || '').toUpperCase() === 'HBM';
    return createTensorVisualization(containerElement, op, {
        type: 'Transfer',
        colors: {
            GLOBAL: new THREE.Color(0.2, 0.2, 0.2),
            HIGHLIGHT: isStoreLike
                ? new THREE.Color(1.0, 0.55, 0.0)
                : new THREE.Color(0.0, 0.7, 1.0)
        },
        viewState
    });
}
