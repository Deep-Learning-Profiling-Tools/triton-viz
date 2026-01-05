import { GridBlock } from './gridblock.js';

let globalData;
let maxX = 0;
let maxY = 0;
let maxZ = 0;
let programIdConfig;
let currentBlockData = null;
let containerElement;

function openRecordViewerAt(x, y, z) {
    if (!programIdConfig || !containerElement) return;
    const clamp = (v, maxV) => Math.max(0, Math.min(maxV, v));
    const nx = clamp(x, programIdConfig.max.x);
    const ny = clamp(y, programIdConfig.max.y);
    const nz = clamp(z, programIdConfig.max.z);
    programIdConfig.values = { x: nx, y: ny, z: nz };
    const blockData = programIdConfig.getBlockData(nx, ny, nz);
    currentBlockData = new GridBlock(
        0,
        0,
        0,
        0,
        nx,
        ny,
        nz,
        blockData,
        switchToMainView,
        containerElement,
        null,
        null,
        programIdConfig
    );
    containerElement.style.pointerEvents = 'auto';
    containerElement.style.display = 'block';
    currentBlockData.showDetailedView();
}

function switchToMainView() {
    currentBlockData = null;
    if (containerElement) {
        containerElement.style.pointerEvents = 'none';
        containerElement.style.display = 'none';
    }
}

function initializeApp() {
    containerElement = document.getElementById('visualization-container');
    if (!containerElement) {
        console.error('Visualization container element not found');
        return;
    }
    containerElement.style.pointerEvents = 'none';
    containerElement.style.display = 'none';
}

function determineMaxValues(visualizationData) {
    maxX = 0;
    maxY = 0;
    maxZ = 0;
    const keys = Object.keys(visualizationData);
    keys.forEach(key => {
        const [x, y, z] = key
            .replace(/[()]/g, '')
            .split(',')
            .map(s => Number(String(s).trim()));
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    });
}

function initializeUIElements() {
    programIdConfig = {
        values: { x: 0, y: 0, z: 0 },
        max: { x: maxX, y: maxY, z: maxZ },
        getBlockData: (x, y, z) => {
            const viz = (globalData && globalData.ops && globalData.ops.visualization_data) || {};
            const key1 = `(${x}, ${y}, ${z})`;
            const key2 = `(${x},${y},${z})`;
            return viz[key1] || viz[key2] || [];
        }
    };
}

async function fetchData() {
    try {
        const API_BASE = window.__TRITON_VIZ_API__ || '';
        const response = await fetch(`${API_BASE}/api/data`);
        globalData = await response.json();

        determineMaxValues(globalData.ops.visualization_data);
        initializeUIElements();
        openRecordViewerAt(0, 0, 0);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeApp();
        fetchData();
    });
} else {
    initializeApp();
    fetchData();
}
