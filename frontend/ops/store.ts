import { createTensorVisualization } from '../components/tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import type { OpRecord } from '../types/types.js';
import type { ViewState } from '../components/tensor_view.js';

export function createStoreVisualization(containerElement: HTMLElement, op: OpRecord, viewState: ViewState | null = null): (() => void) | void {
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
