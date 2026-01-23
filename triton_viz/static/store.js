import { createTensorVisualization, createOverallVisualization } from './tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';

export function createStoreVisualization(containerElement, op, viewState = null) {
    return createTensorVisualization(containerElement, op, {
        type: 'Store',
        colors: {
            GLOBAL: new THREE.Color(0.2, 0.2, 0.2),
            HIGHLIGHT: new THREE.Color(1.0, 0.55, 0.0)
        },
        hasHeatmap: true,
        viewState
    });
}

export function createStoreOverallVisualization(containerElement, op) {
    return createOverallVisualization(containerElement, op, { type: 'Store' });
}
