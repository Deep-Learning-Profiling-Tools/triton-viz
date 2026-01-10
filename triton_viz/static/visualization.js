import { GridBlock } from './gridblock.js';
import { createInfoPopup, showInfoPopup } from './infoPopup.js';
import { createLoadOverallVisualization } from './load.js';
import { createStoreOverallVisualization } from './store.js';
import { enableDrag } from './ui_helpers.js';

let globalData;
let currentView = 'main';
let canvas;
let ctx;
let canvasWrapper;
let canvasLogicalWidth = 0;
let canvasLogicalHeight = 0;
let kernelGrid;
let currentBlockData = null;
let containerElement;
let infoPopup;
let isInitialized = false;
let overallCleanup = null;
let maxX = 0;
let maxY = 0;
let maxZ = 0;
let zSlice = 0;
const overallCache = {};
const filterValues = [0, 0, 0];
const THEME_STORAGE_KEY = 'triton-viz-theme';

const controls = {
    panel: null,
    pidContainer: null,
    zSlider: null,
    zValueLabel: null,
    resetBtn: null,
    overallBtn: null,
    precomputeBtn: null,
    infoBtn: null,
    themeToggle: null,
    resizer: null,
    opColorizeBtn: null,
    opHistogramBtn: null,
    opAllProgramsBtn: null,
};

let controlToastEl = null;
let controlToastTimer = null;

const opControls = {
    handlers: null,
    state: {
        colorize: false,
        showCode: false,
        histogram: false,
        allPrograms: false,
    },
};

function setOpControlState(nextState = {}) {
    opControls.state = {
        ...opControls.state,
        ...nextState,
    };
    updateOpControls();
}

function setOpControlHandlers(handlers = null) {
    opControls.handlers = handlers;
    updateOpControls();
}

function resetOpControls() {
    opControls.handlers = null;
    opControls.state = {
        colorize: false,
        showCode: false,
        histogram: false,
        allPrograms: false,
    };
    if (window.__tritonVizCodeHide) {
        window.__tritonVizCodeHide();
    }
    updateOpControls();
}

function updateToggleLabel(button, label, isOn) {
    if (!button) return;
    button.textContent = `${label}: ${isOn ? 'ON' : 'OFF'}`;
    button.classList.toggle('active', isOn);
}

function updateOpControls() {
    const { handlers, state } = opControls;
    if (controls.opColorizeBtn) {
        controls.opColorizeBtn.disabled = !handlers || !handlers.toggleColorize;
        updateToggleLabel(controls.opColorizeBtn, 'Color by Value', !!state.colorize);
    }
    if (controls.opHistogramBtn) {
        controls.opHistogramBtn.disabled = !handlers || !handlers.toggleHistogram;
        updateToggleLabel(controls.opHistogramBtn, 'Value Histogram', !!state.histogram);
    }
    if (controls.opAllProgramsBtn) {
        controls.opAllProgramsBtn.disabled = !handlers || !handlers.toggleAllPrograms;
        updateToggleLabel(controls.opAllProgramsBtn, 'All Program IDs', !!state.allPrograms);
    }
}

function applyToggleResult(result, key) {
    if (result && typeof result.then === 'function') {
        result.then((value) => {
            setOpControlState({ [key]: !!value });
        });
    } else {
        setOpControlState({ [key]: !!result });
    }
}

try {
    window.setOpControlHandlers = setOpControlHandlers;
    window.setOpControlState = setOpControlState;
    window.resetOpControls = resetOpControls;
} catch (err) {
    console.warn('Unable to expose op control helpers', err);
}

function closeOverallOverlay() {
    if (overallCleanup) {
        overallCleanup();
        overallCleanup = null;
    }
    if (containerElement) {
        containerElement.innerHTML = '';
        containerElement.style.display = 'none';
        containerElement.style.pointerEvents = 'none';
    }
    if (canvas) {
        canvas.style.display = 'block';
    }
    currentView = 'main';
    resetOpControls();
}

function switchToMainView() {
    closeOverallOverlay();
    resetOpControls();

    if (currentBlockData) {
        currentBlockData.hideDetailedView();
        currentBlockData = null;
    }

    if (canvas) {
        canvas.style.display = 'block';
    }

    draw();
}

function switchToTensorView(clickedBlock) {
    currentView = 'tensor';
    resetOpControls();
    currentBlockData = clickedBlock;
    if (currentBlockData) {
        const coords = [
            currentBlockData.gridPosition.x,
            currentBlockData.gridPosition.y,
            currentBlockData.gridPosition.z,
        ];
        coords.forEach((val, idx) => {
            filterValues[idx] = val;
            const slider = document.querySelector(`input[data-filter-index="${idx}"]`);
            if (slider) {
                slider.value = val;
            }
            const pill = document.getElementById(`pid-value-${idx}`);
            if (pill) pill.textContent = String(val);
            if (kernelGrid) {
                kernelGrid.updateFilter(idx, val);
            }
        });
        setZSlice(currentBlockData.gridPosition.z);
        if (kernelGrid) {
            kernelGrid.updateZ(currentBlockData.gridPosition.z);
        }
    }

    if (containerElement) {
        containerElement.style.pointerEvents = 'auto';
        containerElement.style.display = 'block';
    }

    if (canvas) {
        canvas.style.display = 'none';
    }

    clickedBlock.showDetailedView();
}

function initializeApp() {
    canvas = document.getElementById('canvas');
    canvasWrapper = document.getElementById('canvas-wrapper');
    containerElement = document.getElementById('visualization-container');
    controls.panel = document.getElementById('control-panel');
    controls.resizer = document.getElementById('sidebar-resizer');
    controls.pidContainer = document.getElementById('pid-controls');
    controls.zSlider = document.getElementById('z-slider');
    controls.zValueLabel = document.getElementById('z-value');
    controls.resetBtn = document.getElementById('reset-filters');
    controls.overallBtn = document.getElementById('btn-overall');
    controls.precomputeBtn = document.getElementById('btn-precompute');
    controls.infoBtn = document.getElementById('btn-info');
    controls.themeToggle = document.getElementById('theme-toggle');
    controls.opColorizeBtn = document.getElementById('btn-op-colorize');
    controls.opHistogramBtn = document.getElementById('btn-op-histogram');
    controls.opAllProgramsBtn = document.getElementById('btn-op-all-programs');

    if (!canvas || !canvasWrapper || !containerElement) {
        console.error('Essential visualization elements are missing.');
        return;
    }

    ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Unable to obtain 2D context');
        return;
    }

    containerElement.style.pointerEvents = 'none';
    containerElement.style.display = 'none';

    canvas.addEventListener('mousedown', handleMouseEvent);
    canvas.addEventListener('mouseup', handleMouseEvent);
    canvas.addEventListener('mousemove', handleMouseEvent);
    window.addEventListener('resize', resizeCanvas);

    setupThemeToggle();
    setupControlEvents();
    setupSidebarResizer();
    updateOpControls();
    resizeCanvas();
    fetchData();
}

function setupThemeToggle() {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    const defaultTheme = stored || document.documentElement.dataset.theme || 'light';
    applyTheme(defaultTheme);
    if (controls.themeToggle) {
        controls.themeToggle.addEventListener('click', () => {
            const nextTheme = (document.documentElement.dataset.theme || 'light') === 'light' ? 'dark' : 'light';
            applyTheme(nextTheme);
        });
    }
}

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    const icon = document.getElementById('theme-toggle-icon');
    if (icon) {
        icon.textContent = theme === 'light' ? '☀' : '🌙';
    }
    draw();
}

function setupControlEvents() {
    if (controls.resetBtn) {
        controls.resetBtn.addEventListener('click', () => {
            for (let i = 0; i < filterValues.length; i += 1) {
                filterValues[i] = -1;
                const slider = document.querySelector(`input[data-filter-index="${i}"]`);
                if (slider) {
                    slider.value = -1;
                }
                const valuePill = document.getElementById(`pid-value-${i}`);
                if (valuePill) valuePill.textContent = 'All';
                if (kernelGrid) {
                    kernelGrid.updateFilter(i, -1);
                }
            }
            setZSlice(0);
            if (kernelGrid) {
                kernelGrid.updateZ(0);
            }
            draw();
            showControlToast('Filters reset');
        });
    }

    if (controls.overallBtn) {
        controls.overallBtn.addEventListener('click', showOverallOverlay);
    }

    if (controls.precomputeBtn) {
        controls.precomputeBtn.addEventListener('click', () => {
            showControlToast('Precompute mode is coming soon. This build previews the layout.');
        });
    }

    if (controls.infoBtn) {
        controls.infoBtn.addEventListener('click', () => {
            if (!infoPopup) {
                infoPopup = createInfoPopup();
            }
            showInfoPopup(infoPopup);
        });
    }

    if (controls.zSlider) {
        controls.zSlider.addEventListener('input', (event) => {
            const next = Number(event.target.value);
            setZSlice(next);
            if (kernelGrid) {
                kernelGrid.updateZ(next);
                draw();
            }
        });
    }

    if (controls.opColorizeBtn) {
        controls.opColorizeBtn.addEventListener('click', () => {
            const handler = opControls.handlers?.toggleColorize;
            if (!handler) return;
            applyToggleResult(handler(), 'colorize');
        });
    }

    if (controls.opHistogramBtn) {
        controls.opHistogramBtn.addEventListener('click', () => {
            const handler = opControls.handlers?.toggleHistogram;
            if (!handler) return;
            applyToggleResult(handler(), 'histogram');
        });
    }

    if (controls.opAllProgramsBtn) {
        controls.opAllProgramsBtn.addEventListener('click', () => {
            const handler = opControls.handlers?.toggleAllPrograms;
            if (!handler) return;
            applyToggleResult(handler(), 'allPrograms');
        });
    }
}

function setupSidebarResizer() {
    if (!controls.resizer || !controls.panel) return;
    const root = document.documentElement;
    const minWidth = 0;
    let startX = 0;
    let startWidth = 0;

    const onPointerMove = (event) => {
        const delta = event.clientX - startX;
        const resizerWidth = controls.resizer.getBoundingClientRect().width || 0;
        const maxWidth = Math.max(0, window.innerWidth - resizerWidth);
        const next = Math.min(maxWidth, Math.max(minWidth, startWidth + delta));
        root.style.setProperty('--sidebar-width', `${next}px`);
        resizeCanvas();
    };

    const onPointerUp = (event) => {
        controls.resizer.releasePointerCapture(event.pointerId);
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
    };

    controls.resizer.addEventListener('pointerdown', (event) => {
        startX = event.clientX;
        startWidth = controls.panel.getBoundingClientRect().width;
        controls.resizer.setPointerCapture(event.pointerId);
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
    });
}

function showControlToast(message) {
    if (!controls.panel) {
        console.info(message);
        return;
    }
    if (!controlToastEl) {
        controlToastEl = document.createElement('div');
        controlToastEl.id = 'control-panel-toast';
        controlToastEl.className = 'info-card';
        controls.panel.appendChild(controlToastEl);
    }
    controlToastEl.textContent = message;
    controlToastEl.style.opacity = '1';
    clearTimeout(controlToastTimer);
    controlToastTimer = setTimeout(() => {
        if (controlToastEl) {
            controlToastEl.style.opacity = '0';
        }
    }, 3200);
}

function resizeCanvas() {
    if (!canvas || !canvasWrapper || !ctx) return;
    const rect = canvasWrapper.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasLogicalWidth = rect.width;
    canvasLogicalHeight = rect.height;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    if (kernelGrid) {
        const gridRect = getKernelGridRect();
        kernelGrid.resize(gridRect.x, gridRect.y, gridRect.width, gridRect.height);
    }

    draw();
}

function getKernelGridRect() {
    const PADDING = 32;
    const BOTTOM_PADDING = 80;
    const width = Math.max(320, canvasLogicalWidth - PADDING * 2);
    const height = Math.max(320, canvasLogicalHeight - PADDING - BOTTOM_PADDING);
    return { x: PADDING, y: PADDING, width, height };
}

function setZSlice(value) {
    const clamped = Math.max(0, Math.min(value, maxZ));
    zSlice = clamped;
    if (controls.zSlider) {
        controls.zSlider.value = clamped;
    }
    if (controls.zValueLabel) {
        controls.zValueLabel.textContent = clamped;
    }
}

function getThemeColors() {
    const styles = getComputedStyle(document.documentElement);
    const pick = (name, fallback) => {
        const value = styles.getPropertyValue(name);
        return value ? value.trim() : fallback;
    };
    return {
        canvasBg: pick('--canvas-bg', '#1e1e28'),
        gridBg: pick('--grid-bg', '#f0f0f0'),
        gridBorder: pick('--grid-border', '#d4d4d8'),
        blockBg: pick('--block-bg', '#3c3c46'),
        blockHoverBg: pick('--block-hover-bg', '#50505a'),
        blockBorder: pick('--block-border', 'rgba(0,0,0,0.4)'),
        textPrimary: pick('--text-primary', '#0f172a'),
        textSecondary: pick('--text-secondary', '#475569'),
    };
}

class KernelGrid {
    constructor(x, y, width, height, gridSize, visualizationData) {
        this.rect = { x, y, width, height };
        this.gridSize = gridSize;
        this.visualizationData = visualizationData;
        this.currentZ = 0;
        this.blocks = [];
        this.filterValues = [0, 0, 0];
        this.selectedBlock = null;
        this.calculateBlockSize();
        this.createBlocks();
    }

    resize(x, y, width, height) {
        this.rect = { x, y, width, height };
        this.calculateBlockSize();
        this.createBlocks();
    }

    calculateBlockSize() {
        const [gridX, gridY] = this.gridSize;
        const safeX = Math.max(1, gridX);
        const safeY = Math.max(1, gridY);
        this.blockWidth = Math.floor(this.rect.width / safeX) - 1;
        this.blockHeight = Math.floor(this.rect.height / safeY) - 1;
    }

    createBlocks() {
        this.blocks = [];
        const [gridX, gridY] = this.gridSize;
        for (let y = 0; y < gridY; y += 1) {
            for (let x = 0; x < gridX; x += 1) {
                const blockX = this.rect.x + x * (this.blockWidth + 1);
                const blockY = this.rect.y + y * (this.blockHeight + 1);
                const gridKey1 = `(${x}, ${y}, ${this.currentZ})`;
                const gridKey2 = `(${x},${y},${this.currentZ})`;
                const blockData = this.visualizationData[gridKey1] || this.visualizationData[gridKey2] || [];
                const block = new GridBlock(
                    blockX,
                    blockY,
                    this.blockWidth,
                    this.blockHeight,
                    x,
                    y,
                    this.currentZ,
                    blockData,
                    switchToMainView,
                    containerElement,
                    canvas,
                    draw
                );
                this.blocks.push(block);
            }
        }
    }

    draw(ctxRef, palette) {
        ctxRef.fillStyle = palette.gridBg;
        ctxRef.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        ctxRef.strokeStyle = palette.gridBorder;
        ctxRef.lineWidth = 1;
        ctxRef.strokeRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        this.blocks.forEach((block) => {
            if (this.shouldDrawBlock(block)) {
                block.draw(ctxRef, palette);
            }
        });
    }

    shouldDrawBlock(block) {
        return (
            (this.filterValues[0] === -1 || block.gridPosition.x === this.filterValues[0]) &&
            (this.filterValues[1] === -1 || block.gridPosition.y === this.filterValues[1]) &&
            (this.filterValues[2] === -1 || block.gridPosition.z === this.filterValues[2])
        );
    }

    updateFilter(dimension, value) {
        this.filterValues[dimension] = value;
    }

    updateZ(z) {
        this.currentZ = z;
        this.filterValues[2] = z;
        this.blocks.forEach((block) => {
            block.gridPosition.z = z;
            const gridKey = `(${block.gridPosition.x}, ${block.gridPosition.y}, ${z})`;
            block.blockData = this.visualizationData[gridKey] || [];
        });
    }

    handleMouseMove(x, y) {
        this.blocks.forEach((block) => {
            if (this.shouldDrawBlock(block)) {
                block.handleMouseMove(x, y);
            } else {
                block.isHovered = false;
            }
        });
    }

    handleClick(x, y) {
        const target = this.blocks.find(
            (block) => block.isPointInside(x, y) && this.shouldDrawBlock(block)
        );
        if (target) {
            if (this.selectedBlock && this.selectedBlock !== target) {
                this.selectedBlock.hideDetailedView();
            }
            this.selectedBlock = target;
            target.showDetailedView();
            return target;
        }
        return null;
    }
}

function determineMaxValues(visualizationData) {
    maxX = 0;
    maxY = 0;
    maxZ = 0;
    Object.keys(visualizationData || {}).forEach((key) => {
        const [x, y, z] = key
            .replace(/[()]/g, '')
            .split(',')
            .map((s) => Number(String(s).trim()));
        if (Number.isFinite(x)) maxX = Math.max(maxX, x);
        if (Number.isFinite(y)) maxY = Math.max(maxY, y);
        if (Number.isFinite(z)) maxZ = Math.max(maxZ, z);
    });
}

function initializeUIElements() {
    if (!globalData || !globalData.ops) return;
    const vizData = globalData.ops.visualization_data || {};
    const gridRect = getKernelGridRect();
    kernelGrid = new KernelGrid(
        gridRect.x,
        gridRect.y,
        gridRect.width,
        gridRect.height,
        [maxX + 1, maxY + 1, maxZ + 1],
        vizData
    );
    kernelGrid.updateZ(zSlice);
    kernelGrid.updateFilter(0, filterValues[0]);
    kernelGrid.updateFilter(1, filterValues[1]);
    kernelGrid.updateFilter(2, filterValues[2]);
    createProgramIdControls();
    updateZSliderState();

    if (!infoPopup) {
        infoPopup = createInfoPopup();
    }

    isInitialized = true;
    draw();
}

function createProgramIdControls() {
    if (!controls.pidContainer) return;
    controls.pidContainer.innerHTML = '';
    const labels = ['X', 'Y', 'Z'];
    const maxValues = [maxX, maxY, maxZ];
    labels.forEach((label, index) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'control-field is-inline';
        const nameSpan = document.createElement('span');
        nameSpan.className = 'control-label';
        nameSpan.textContent = label;
        const valueSpan = document.createElement('span');
        valueSpan.className = 'value-pill';
        valueSpan.id = `pid-value-${index}`;
        const isLocked = maxValues[index] <= 0;
        if (isLocked) {
            filterValues[index] = 0;
        }
        const displayValue = isLocked ? 0 : filterValues[index];
        valueSpan.textContent = displayValue < 0 ? 'All' : String(displayValue);
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = isLocked ? 0 : -1;
        slider.max = maxValues[index];
        slider.value = displayValue;
        slider.dataset.filterIndex = index;
        slider.disabled = isLocked;
        slider.addEventListener('input', handleProgramFilterChange);
        wrapper.appendChild(nameSpan);
        wrapper.appendChild(slider);
        wrapper.appendChild(valueSpan);
        controls.pidContainer.appendChild(wrapper);
        if (isLocked && kernelGrid) {
            kernelGrid.updateFilter(index, 0);
        }
    });
}

function handleProgramFilterChange(event) {
    const index = Number(event.target.dataset.filterIndex);
    const value = Number(event.target.value);
    filterValues[index] = value;
    const pill = document.getElementById(`pid-value-${index}`);
    if (pill) {
        pill.textContent = value < 0 ? 'All' : String(value);
    }
    if (kernelGrid) {
        kernelGrid.updateFilter(index, value);
        draw();
    }
    if (currentBlockData && typeof currentBlockData.applyProgramIdSelection === 'function') {
        const fallback = currentBlockData.gridPosition || { x: 0, y: 0, z: 0 };
        const nextX = filterValues[0] >= 0 ? filterValues[0] : fallback.x;
        const nextY = filterValues[1] >= 0 ? filterValues[1] : fallback.y;
        const nextZ = filterValues[2] >= 0 ? filterValues[2] : fallback.z;
        currentBlockData.applyProgramIdSelection(nextX, nextY, nextZ);
    }
}

function updateZSliderState() {
    if (!controls.zSlider) return;
    controls.zSlider.min = 0;
    controls.zSlider.max = maxZ;
    controls.zSlider.disabled = maxZ <= 0;
    setZSlice(Math.min(zSlice, maxZ));
}

function handleMouseEvent(event) {
    if (!isInitialized || currentView !== 'main') return;
    if (!kernelGrid) return;
    const { offsetX, offsetY } = event;
    if (event.type === 'mousemove') {
        kernelGrid.handleMouseMove(offsetX, offsetY);
    }
    if (event.type === 'mousedown') {
        const block = kernelGrid.handleClick(offsetX, offsetY);
        if (block) {
            switchToTensorView(block);
            return;
        }
    }
    draw();
}

function draw() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvasLogicalWidth, canvasLogicalHeight);
    const palette = getThemeColors();
    ctx.fillStyle = palette.canvasBg;
    ctx.fillRect(0, 0, canvasLogicalWidth, canvasLogicalHeight);
    if (currentView === 'main' && kernelGrid) {
        kernelGrid.draw(ctx, palette);
    }
}

function collectOpsByType(kind = 'any') {
    if (!globalData || !globalData.ops || !globalData.ops.visualization_data) return [];
    const vizData = globalData.ops.visualization_data;
    const keys = Object.keys(vizData).sort((a, b) => {
        const parse = (key) =>
            key
                .replace(/[()]/g, '')
                .split(',')
                .map((s) => Number(String(s).trim()));
        const [ax, ay, az] = parse(a);
        const [bx, by, bz] = parse(b);
        if (ax !== bx) return ax - bx;
        if (ay !== by) return ay - by;
        return az - bz;
    });
    const ops = [];
    keys.forEach((key) => {
        const list = vizData[key] || [];
        list.forEach((op) => {
            if (op && op.overall_key && (kind === 'any' || op.type === kind)) {
                ops.push(op);
            }
        });
    });
    return ops;
}

async function fetchOverallData(keys, kind) {
    const unique = Array.from(new Set((keys || []).filter(Boolean)));
    if (!unique.length) {
        throw new Error('No overall data available');
    }
    const API_BASE = window.__TRITON_VIZ_API__ || '';
    const endpoint = kind === 'store' ? 'store_overall' : 'load_overall';
    const results = await Promise.all(
        unique.map(async (key) => {
            const resp = await fetch(`${API_BASE}/api/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key }),
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data && data.error ? data.error : 'Request failed');
            }
            return data;
        })
    );
    const merged = {
        shape: results[0]?.shape || [],
        slice_shape: results[0]?.slice_shape || [],
        tiles: [],
    };
    results.forEach((entry) => {
        (entry.tiles || []).forEach((tile) => merged.tiles.push(tile));
    });
    return merged;
}

async function showOverallOverlay() {
    const ops = collectOpsByType('any');
    if (!ops.length) {
        showControlToast('No Load/Store ops available to aggregate.');
        return;
    }
    resetOpControls();
    currentView = 'overall';
    if (canvas) canvas.style.display = 'none';
    if (!containerElement) return;
    containerElement.style.pointerEvents = 'auto';
    containerElement.style.display = 'block';
    containerElement.innerHTML = '';

    const overlay = document.createElement('div');
    overlay.className = 'overall-overlay';
    const shell = document.createElement('div');
    shell.className = 'overall-shell';
    overlay.appendChild(shell);

    const titleRow = document.createElement('div');
    titleRow.className = 'overlay-title-row';
    const title = document.createElement('h2');
    title.textContent = 'Load / Store Overview';
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `${ops.length} ops`;
    titleRow.appendChild(title);
    titleRow.appendChild(badge);
    const showCodeBtn = document.createElement('button');
    showCodeBtn.className = 'viz-button ghost';
    showCodeBtn.textContent = 'Show Code: OFF';
    titleRow.appendChild(showCodeBtn);

    // Kernel Summary toggle + panel (op counts + load/store bytes).
    let summaryPanel = null;
    const destroySummaryPanel = () => {
        if (summaryPanel && summaryPanel.remove) summaryPanel.remove();
        summaryPanel = null;
    };

    const summaryBtn = document.createElement('button');
    summaryBtn.className = 'viz-button ghost';
    summaryBtn.textContent = 'Summary: OFF';
    titleRow.appendChild(summaryBtn);

    const openSummaryPanel = () => {
        destroySummaryPanel();
        try {
            const panel = document.createElement('div');
            panel.className = 'info-card';
            panel.style.position = 'fixed';
            panel.style.right = '24px';
            panel.style.top = '96px';
            panel.style.maxWidth = '320px';
            panel.style.zIndex = '2200';

            const header = document.createElement('div');
            header.className = 'panel-header drag-handle';
            header.style.marginBottom = '6px';
            const titleSpan = document.createElement('span');
            titleSpan.textContent = 'Kernel Summary';
            header.appendChild(titleSpan);
            const grip = document.createElement('span');
            grip.className = 'drag-grip';
            grip.setAttribute('aria-hidden', 'true');
            grip.textContent = '⠿';
            header.appendChild(grip);
            const closeBtn = document.createElement('button');
            closeBtn.className = 'viz-button ghost';
            closeBtn.textContent = 'Close';
            closeBtn.style.marginLeft = 'auto';
            closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
            closeBtn.addEventListener('click', () => {
                destroySummaryPanel();
                summaryBtn.textContent = 'Summary: OFF';
            });
            header.appendChild(closeBtn);
            panel.appendChild(header);

            const body = document.createElement('div');
            body.style.fontSize = '11px';
            body.style.display = 'flex';
            body.style.flexDirection = 'column';
            body.style.gap = '6px';

            // 1) Op counts from visualization_data.
            const vizData = (globalData && globalData.ops && globalData.ops.visualization_data) || {};
            const opCounts = {};
            Object.values(vizData).forEach((list) => {
                (list || []).forEach((op) => {
                    const t = (op && op.type) || 'Unknown';
                    opCounts[t] = (opCounts[t] || 0) + 1;
                });
            });
            const countsKeys = Object.keys(opCounts).sort();
            const countsBlock = document.createElement('div');
            const countsTitle = document.createElement('div');
            countsTitle.textContent = 'Op counts';
            countsTitle.style.fontWeight = '600';
            countsTitle.style.marginBottom = '2px';
            countsBlock.appendChild(countsTitle);
            if (countsKeys.length) {
                const list = document.createElement('ul');
                list.style.margin = '0';
                list.style.paddingLeft = '16px';
                countsKeys.forEach((name) => {
                    const li = document.createElement('li');
                    li.textContent = `${name}: ${opCounts[name]}`;
                    list.appendChild(li);
                });
                countsBlock.appendChild(list);
            } else {
                const empty = document.createElement('div');
                empty.textContent = 'No ops recorded.';
                countsBlock.appendChild(empty);
            }
            body.appendChild(countsBlock);

            // 2) Load/Store bytes aggregated from per-op metadata, plus optional extra stats.
            const analysis = globalData && globalData.analysis;
            const metrics = analysis && Array.isArray(analysis.Metric) ? analysis.Metric : null;
            const values = analysis && Array.isArray(analysis.Value) ? analysis.Value : null;
            const bytesBlock = document.createElement('div');
            const bytesTitle = document.createElement('div');
            bytesTitle.textContent = 'Load / Store bytes';
            bytesTitle.style.fontWeight = '600';
            bytesTitle.style.margin = '6px 0 2px';
            bytesBlock.appendChild(bytesTitle);

            const vizBytes = { Load: 0, Store: 0 };
            Object.values(vizData).forEach((list) => {
                (list || []).forEach((op) => {
                    if (!op || typeof op.bytes !== 'number') return;
                    if (op.type === 'Load') {
                        vizBytes.Load += Math.max(0, op.bytes);
                    } else if (op.type === 'Store') {
                        vizBytes.Store += Math.max(0, op.bytes);
                    }
                });
            });

            const table = document.createElement('table');
            table.style.width = '100%';
            table.style.borderCollapse = 'collapse';

            const addRow = (label, value) => {
                const row = document.createElement('tr');
                const k = document.createElement('td');
                const v = document.createElement('td');
                k.textContent = label;
                k.style.paddingRight = '4px';
                k.style.verticalAlign = 'top';
                v.textContent = String(value);
                v.style.textAlign = 'right';
                v.style.whiteSpace = 'nowrap';
                row.appendChild(k);
                row.appendChild(v);
                table.appendChild(row);
            };

            addRow('Total load bytes', vizBytes.Load);
            addRow('Total store bytes', vizBytes.Store);

            // Number of grids that actually perform Load/Store.
            const activeGridKeys = Object.entries(vizData).filter(([, list]) =>
                (list || []).some((op) => op && (op.type === 'Load' || op.type === 'Store'))
            );
            addRow('Grids with Load/Store', activeGridKeys.length);

            // If analysis metrics are available, append them as extra rows (excluding raw "Grid Size").
            if (metrics && values && metrics.length === values.length && metrics.length > 0) {
                metrics.forEach((name, idx) => {
                    const label = String(name);
                    if (String(label) === 'Grid Size') return;
                    const value = values[idx];
                    addRow(label, value);
                });
            }

            bytesBlock.appendChild(table);
            body.appendChild(bytesBlock);

            panel.appendChild(body);
            document.body.appendChild(panel);
            enableDrag(panel, { handle: header, bounds: window, initialLeft: window.innerWidth - 360, initialTop: 96 });
            summaryPanel = panel;
        } catch (e) {
            console.warn('Kernel summary panel failed:', e);
        }
    };

    summaryBtn.addEventListener('click', () => {
        const turnOn = summaryBtn.textContent.endsWith('OFF');
        summaryBtn.textContent = `Summary: ${turnOn ? 'ON' : 'OFF'}`;
        if (turnOn) {
            openSummaryPanel();
        } else {
            destroySummaryPanel();
        }
    });

    shell.appendChild(titleRow);

    const tabs = document.createElement('div');
    tabs.className = 'overall-tabs';
    shell.appendChild(tabs);

    const contentArea = document.createElement('div');
    contentArea.className = 'overall-content';
    shell.appendChild(contentArea);

    const footerNote = document.createElement('div');
    footerNote.className = 'info-card';
    footerNote.textContent = 'Tip: switch themes or filters before capturing paper-ready shots.';
    shell.appendChild(footerNote);

    const backBtn = document.createElement('button');
    backBtn.className = 'viz-button ghost overall-back';
    backBtn.textContent = 'Back to Canvas';
    backBtn.addEventListener('click', () => {
        closeOverallOverlay();
        switchToMainView();
        destroyOverallCodePanel();
        // Ensure kernel summary panel is cleaned up when leaving overall view
        if (typeof destroySummaryPanel === 'function') {
            destroySummaryPanel();
            if (summaryBtn) summaryBtn.textContent = 'Summary: OFF';
        }
    });
    shell.appendChild(backBtn);

    containerElement.appendChild(overlay);

    let currentTab = null;
    let currentOverallOp = null;
    let overallCodePanel = null;

    const destroyOverallCodePanel = () => {
        if (overallCodePanel && overallCodePanel.remove) overallCodePanel.remove();
        overallCodePanel = null;
    };

    const openOverallCodePanel = async (op) => {
        destroyOverallCodePanel();
        if (!op || !op.uuid) {
            showControlToast('No code available for this operation.');
            return;
        }
        const wrapper = document.createElement('div');
        wrapper.className = 'show-code-panel';
        const header = document.createElement('div');
        header.className = 'panel-header drag-handle';
        header.innerHTML = '<span>Operation Code & Context</span><span class="drag-grip" aria-hidden="true">⠿</span>';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'viz-button ghost';
        closeBtn.textContent = 'Close';
        closeBtn.style.marginLeft = 'auto';
        closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
        closeBtn.addEventListener('click', () => {
            destroyOverallCodePanel();
            showCodeBtn.textContent = 'Show Code: OFF';
        });
        header.appendChild(closeBtn);
        wrapper.appendChild(header);
        const meta = document.createElement('div');
        meta.style.marginBottom = '6px';
        meta.style.fontSize = '12px';
        wrapper.appendChild(meta);
        const pre = document.createElement('pre');
        pre.style.margin = '0';
        wrapper.appendChild(pre);
        document.body.appendChild(wrapper);
        enableDrag(wrapper, { handle: header, bounds: window, initialLeft: window.innerWidth - 520, initialTop: 120 });
        overallCodePanel = wrapper;
        try {
            const API_BASE = window.__TRITON_VIZ_API__ || '';
            const res = await fetch(`${API_BASE}/api/op_code`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uuid: op.uuid, frame_idx: 0, context: 8 })
            });
            const data = await res.json();
            meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
            const lines = (data.lines || []).map((line) => {
                const mark = (data.highlight === line.no) ? '▶ ' : '  ';
                return `${mark}${String(line.no).padStart(6, ' ')} | ${line.text || ''}`;
            }).join('\n');
            pre.textContent = lines || '(no code available)';
        } catch (err) {
            pre.textContent = `Failed to load code: ${err}`;
        }
    };

    showCodeBtn.addEventListener('click', async () => {
        const turnOn = showCodeBtn.textContent.endsWith('OFF');
        showCodeBtn.textContent = `Show Code: ${turnOn ? 'ON' : 'OFF'}`;
        if (!turnOn) {
            destroyOverallCodePanel();
            return;
        }
        if (currentOverallOp) {
            await openOverallCodePanel(currentOverallOp);
        } else {
            showControlToast('Select an operation first.');
            showCodeBtn.textContent = 'Show Code: OFF';
        }
    });

    const selectOp = async (op, tabElement) => {
        currentOverallOp = op;
        if (currentTab) currentTab.classList.remove('active');
        currentTab = tabElement;
        currentTab.classList.add('active');
        if (overallCleanup) {
            overallCleanup();
            overallCleanup = null;
        }
        contentArea.innerHTML = '<div class="overall-empty">Loading…</div>';
        try {
            const payload = await getOverallPayload(op);
            contentArea.innerHTML = '';
            const renderer = op.type === 'Store' ? createStoreOverallVisualization : createLoadOverallVisualization;
            overallCleanup = renderer(contentArea, payload);
            if (showCodeBtn.textContent.endsWith('ON')) {
                await openOverallCodePanel(op);
            }
        } catch (err) {
            contentArea.innerHTML = `<div class="overall-empty">${err}</div>`;
        }
    };

    ops.forEach((op, idx) => {
        const tab = document.createElement('button');
        tab.textContent = `${idx + 1}. ${op.type} (${(op.global_shape || []).join('×') || 'shape'})`;
        if (idx === 0) tab.classList.add('active');
        tab.addEventListener('click', () => selectOp(op, tab));
        tabs.appendChild(tab);
        if (idx === 0) {
            selectOp(op, tab);
        }
    });
}

async function getOverallPayload(op) {
    if (!op.overall_key) {
        throw new Error('Overall data unavailable for this operation.');
    }
    const cacheKey = `${op.type}:${op.overall_key}`;
    if (!overallCache[cacheKey]) {
        const endpoint = op.type === 'Store' ? 'store_overall' : 'load_overall';
        const API_BASE = window.__TRITON_VIZ_API__ || '';
        const resp = await fetch(`${API_BASE}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: op.overall_key }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            throw new Error(data && data.error ? data.error : 'Request failed');
        }
        overallCache[cacheKey] = data;
    }
    const data = overallCache[cacheKey];
    return {
        ...op,
        overall_mode: true,
        overall_tiles: data.tiles || [],
        overall_shape: data.shape || op.global_shape,
        overall_slice_shape: data.slice_shape || op.slice_shape,
    };
}

async function fetchData() {
    try {
        const API_BASE = window.__TRITON_VIZ_API__ || '';
        const response = await fetch(`${API_BASE}/api/data`);
        globalData = await response.json();
        determineMaxValues(globalData?.ops?.visualization_data || {});
        initializeUIElements();
    } catch (error) {
        console.error('Error fetching data:', error);
        showControlToast('Failed to load data. Please ensure the backend is running.');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
