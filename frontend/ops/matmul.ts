import { createTensorVisualization } from '../components/tensor_view.js';
import * as THREE from 'https://esm.sh/three@0.155.0';
import type { OpRecord } from '../types/types.js';
import type { ViewState } from '../components/tensor_view.js';

type MatrixSize = { rows: number; cols: number; width: number; height: number };
type Vec3 = [number, number, number];

export function createMatMulVisualization(containerElement: HTMLElement, op: OpRecord, viewState: ViewState | null = null): (() => void) | void {
    const { input_shape, other_shape, output_shape } = op;
    const CUBE_SIZE = 0.2;
    const GAP = 0.05;
    const spacing = CUBE_SIZE + GAP;

    function matrixSize(dimensions: number[]): MatrixSize {
        const rows = dimensions[0] ?? 1;
        const cols = dimensions[1] ?? 1;
        const width = (cols - 1) * spacing + CUBE_SIZE;
        const height = (rows - 1) * spacing + CUBE_SIZE;
        return { rows, cols, width, height };
    }

    const gap = -4 * spacing;
    const sizeA = matrixSize(input_shape || []);
    const sizeB = matrixSize(other_shape || []);
    const shapeA = input_shape || [];
    const shapeB = other_shape || [];
    const shapeC = output_shape || [];
    const posC: Vec3 = [0, 0, 0];
    const posA: Vec3 = [-(sizeA.width + gap), 0, 0];
    const posB: Vec3 = [0, sizeB.height + gap, 0];

    const AXIS_COLORS = {
        x: '#f87171', // Red -> K
        y: '#4ade80', // Green -> M
        z: '#60a5fa'  // Blue -> N
    };

    const cleanup = createTensorVisualization(containerElement, op, {
        type: 'Dot',
        tensorConfigs: [
            { name: 'A', shape: shapeA, color: '#a1d9fc', position: posA, endpoint: 'getMatmulA' },
            { name: 'B', shape: shapeB, color: '#fcf8b0', position: posB, endpoint: 'getMatmulB' },
            { name: 'C', shape: shapeC, color: '#adfca9', position: posC, endpoint: 'getMatmulC' }
        ],
        showDimLines: true,
        dimColors: {
            A: [AXIS_COLORS.y, AXIS_COLORS.x], // M, K
            B: [AXIS_COLORS.x, AXIS_COLORS.z], // K, N
            C: [AXIS_COLORS.y, AXIS_COLORS.z]  // M, N
        },
        viewState
    });

    return cleanup;
}
