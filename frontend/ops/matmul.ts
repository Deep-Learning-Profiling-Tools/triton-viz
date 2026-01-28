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

    const gapX = 0 * spacing;
    const gapY = 0 * spacing;
    const sizeA = matrixSize(input_shape || []);
    const sizeB = matrixSize(other_shape || []);
    const shapeA = input_shape || [];
    const shapeB = other_shape || [];
    const shapeC = output_shape || [];
    const sizeC = matrixSize(shapeC);
    const layoutCols = Math.max(sizeA.cols, sizeB.cols, sizeC.cols);
    const layoutRows = Math.max(sizeA.rows, sizeB.rows, sizeC.rows);
    const layoutWidth = (layoutCols - 1) * spacing + CUBE_SIZE;
    const layoutHeight = (layoutRows - 1) * spacing + CUBE_SIZE;
    const dimExtension = 0.8 + 0.1;
    const posC: Vec3 = [0, 0, 0];
    const posA: Vec3 = [-(sizeC.width / 2 + dimExtension + sizeA.width / 2 + gapX), 0, 0];
    const posB: Vec3 = [0, sizeC.height / 2 + dimExtension + sizeB.height / 2 + gapY, 0];

    const AXIS_COLORS = {
        x: '#f87171', // Red -> K
        y: '#4ade80', // Green -> M
        z: '#60a5fa'  // Blue -> N
    };

    const dimLinePos = (CUBE_SIZE + GAP) * 1.5 + 0.1;
    const boundsWidth = sizeC.width + dimLinePos + sizeA.width + gapX;
    const boundsHeight = sizeC.height + dimLinePos + sizeB.height + gapY;
    const boundsCenterX = -boundsWidth / 2;
    const boundsCenterY = -boundsHeight / 2;
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
        layoutBounds: { width: boundsWidth, height: boundsHeight, depth: CUBE_SIZE, center: [boundsCenterX, boundsCenterY, 0] },
        fitToTensors: true,
        cameraPadding: 0.75,
        viewState
    });

    return cleanup;
}
