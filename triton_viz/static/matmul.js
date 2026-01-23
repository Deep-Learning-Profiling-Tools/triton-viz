import { createTensorVisualization } from './tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';

export function createMatMulVisualization(containerElement, op, viewState = null) {
    const { input_shape, other_shape, output_shape } = op;
    const CUBE_SIZE = 0.2;
    const GAP = 0.05;
    const spacing = CUBE_SIZE + GAP;

    function matrixSize(dimensions) {
        const rows = dimensions[0];
        const cols = dimensions[1];
        const width = (cols - 1) * spacing + CUBE_SIZE;
        const height = (rows - 1) * spacing + CUBE_SIZE;
        return { rows, cols, width, height };
    }

    const gap = 5 * spacing;
    const sizeA = matrixSize(input_shape);
    const sizeB = matrixSize(other_shape);
    const posC = [0, 0, 0];
    const posA = [-(sizeA.width + gap), 0, 0];
    const posB = [0, sizeB.height + gap, 0];

    const AXIS_COLORS = {
        x: '#f87171', // Red -> K
        y: '#4ade80', // Green -> M
        z: '#60a5fa'  // Blue -> N
    };

    const cleanup = createTensorVisualization(containerElement, op, {
        type: 'Dot',
        tensorConfigs: [
            { name: 'A', shape: input_shape, color: '#3575ff', position: posA, endpoint: 'getMatmulA' },
            { name: 'B', shape: other_shape, color: '#ffeb3b', position: posB, endpoint: 'getMatmulB' },
            { name: 'C', shape: output_shape, color: '#4caf50', position: posC, endpoint: 'getMatmulC' }
        ],
        showDimLines: true,
        dimColors: {
            A: [AXIS_COLORS.y, AXIS_COLORS.x], // M, K
            B: [AXIS_COLORS.x, AXIS_COLORS.z], // K, N
            C: [AXIS_COLORS.y, AXIS_COLORS.z]  // M, N
        }
    });

    return cleanup;
}
