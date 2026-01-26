import { createTensorVisualization } from './tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import type { OpRecord } from './types.js';

export function createLoadVisualization(containerElement: HTMLElement, op: OpRecord, viewState: unknown = null) {
    return createTensorVisualization(containerElement, op, {
        type: 'Load',
        colors: {
            GLOBAL: new THREE.Color(0.2, 0.2, 0.2),
            HIGHLIGHT: new THREE.Color(0.0, 0.7, 1.0)
        },
        viewState
    });
}
