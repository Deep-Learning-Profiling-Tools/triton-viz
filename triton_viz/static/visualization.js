import { GridBlock } from './gridblock.js';

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
let isInitialized = false;
let maxX = 0;
let maxY = 0;
let maxZ = 0;
let zSlice = 0;
const filterValues = [0, 0, 0];
const THEME_STORAGE_KEY = 'triton-viz-theme';

const controls = {
    panel: null,
    pidContainer: null,
    zSlider: null,
    zValueLabel: null,
    resetBtn: null,
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
    try { window.__tritonVizOpState = { ...opControls.state }; } catch (err) {}
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
    try { window.__tritonVizOpState = { ...opControls.state }; } catch (err) {}
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
        updateToggleLabel(controls.opColorizeBtn, 'Heatmap', !!state.colorize);
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

function switchToMainView() {
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

function openProgramZeroBlock() {
    if (!kernelGrid || kernelGrid.blocks.length === 0) return;
    const target = kernelGrid.blocks.find(
        (block) =>
            block.gridPosition.x === 0 &&
            block.gridPosition.y === 0 &&
            block.gridPosition.z === 0 &&
            Array.isArray(block.blockData) &&
            block.blockData.length > 0
    );
    if (target) {
        switchToTensorView(target);
    }
}

function initializeApp() {
    if (document.body) {
        document.body.classList.add('tensor-only');
    }
    canvas = document.getElementById('canvas');
    canvasWrapper = document.getElementById('canvas-wrapper');
    containerElement = document.getElementById('visualization-container');
    controls.panel = document.getElementById('control-panel');
    controls.resizer = document.getElementById('sidebar-resizer');
    controls.pidContainer = document.getElementById('pid-controls');
    controls.zSlider = document.getElementById('z-slider');
    controls.zValueLabel = document.getElementById('z-value');
    controls.resetBtn = document.getElementById('reset-filters');
    controls.precomputeBtn = document.getElementById('btn-precompute');
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
                filterValues[i] = 0;
                const slider = document.querySelector(`input[data-filter-index="${i}"]`);
                if (slider) {
                    slider.value = 0;
                }
                const valuePill = document.getElementById(`pid-value-${i}`);
                if (valuePill) valuePill.textContent = '0';
                if (kernelGrid) {
                    kernelGrid.updateFilter(i, 0);
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


    if (controls.precomputeBtn) {
        controls.precomputeBtn.addEventListener('click', () => {
            showControlToast('Precompute mode is coming soon. This build previews the layout.');
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
        this.gridDepth = Math.max(1, (gridSize[2] || 1));
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
                const programIdConfig = {
                    max: {
                        x: Math.max(0, gridX - 1),
                        y: Math.max(0, gridY - 1),
                        z: Math.max(0, this.gridDepth - 1),
                    },
                    values: {
                        x,
                        y,
                        z: this.currentZ,
                    },
                    getBlockData: (nx, ny, nz) => this.getBlockDataAt(nx, ny, nz),
                };
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
                    , programIdConfig
                );
                this.blocks.push(block);
            }
        }
    }

    getBlockDataAt(nx, ny, nz) {
        const withSpaces = `(${nx}, ${ny}, ${nz})`;
        const withoutSpaces = `(${nx},${ny},${nz})`;
        return (
            this.visualizationData[withSpaces] ||
            this.visualizationData[withoutSpaces] ||
            []
        );
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
            block.gridPosition.x === this.filterValues[0] &&
            block.gridPosition.y === this.filterValues[1] &&
            block.gridPosition.z === this.filterValues[2]
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

    isInitialized = true;
    draw();
    openProgramZeroBlock();
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
        const displayValue = isLocked ? 0 : Math.max(0, filterValues[index]);
        filterValues[index] = displayValue;
        valueSpan.textContent = String(displayValue);
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = 0;
        slider.max = maxValues[index];
        slider.value = displayValue;
        slider.dataset.filterIndex = index;
        slider.disabled = isLocked;
        const tickId = `pid-ticks-${index}`;
        const ticks = document.createElement('datalist');
        ticks.id = tickId;
        for (let i = 0; i <= maxValues[index]; i += 1) {
            const opt = document.createElement('option');
            opt.value = String(i);
            ticks.appendChild(opt);
        }
        slider.setAttribute('list', tickId);
        slider.addEventListener('input', handleProgramFilterChange);
        wrapper.appendChild(nameSpan);
        wrapper.appendChild(slider);
        wrapper.appendChild(valueSpan);
        wrapper.appendChild(ticks);
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
        pill.textContent = String(value);
    }
    if (kernelGrid) {
        kernelGrid.updateFilter(index, value);
        draw();
    }
    if (currentBlockData && typeof currentBlockData.applyProgramIdSelection === 'function') {
        const nextX = filterValues[0];
        const nextY = filterValues[1];
        const nextZ = filterValues[2];
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
