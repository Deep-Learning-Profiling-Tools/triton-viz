import { OpWorkspace } from './op_workspace.js';
import { getJson } from './api.js';
import { getState, resetToggles, setActiveProgram as setActiveProgramState, setToggles } from './state.js';
import { logAction } from './logger.js';
import { createDisposer } from './utils/dispose.js';
let globalData;
let visualizationData;
let containerElement;
let opWorkspace;
let maxX = 0;
let maxY = 0;
let maxZ = 0;
const PROGRAM_AXES = ['x', 'y', 'z'];
const appDisposer = createDisposer();
let pidDisposer = createDisposer();
const THEME_STORAGE_KEY = 'triton-viz-theme';
const controls = {
    panel: null,
    pidContainer: null,
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
};
function setOpControlState(nextState = {}) {
    const toggles = {
        ...getState().toggles,
        ...nextState,
    };
    setToggles(toggles);
    try {
        window.__tritonVizOpState = { ...toggles };
    }
    catch (err) { }
    updateOpControls(toggles);
}
function setOpControlHandlers(handlers = null) {
    opControls.handlers = handlers;
    updateOpControls();
}
function resetOpControls() {
    opControls.handlers = null;
    const nextState = resetToggles();
    try {
        window.__tritonVizOpState = { ...nextState.toggles };
    }
    catch (err) { }
    if (window.__tritonVizCodeHide) {
        window.__tritonVizCodeHide();
    }
    updateOpControls(nextState.toggles);
}
function updateToggleLabel(button, label, isOn) {
    if (!button)
        return;
    button.textContent = `${label}: ${isOn ? 'ON' : 'OFF'}`;
    button.classList.toggle('active', isOn);
}
function updateOpControls(state = null) {
    const { handlers } = opControls;
    const nextState = state || getState().toggles;
    if (controls.opColorizeBtn) {
        controls.opColorizeBtn.disabled = !handlers || !handlers.toggleColorize;
        updateToggleLabel(controls.opColorizeBtn, 'Heatmap', !!nextState.colorize);
    }
    if (controls.opHistogramBtn) {
        controls.opHistogramBtn.disabled = !handlers || !handlers.toggleHistogram;
        updateToggleLabel(controls.opHistogramBtn, 'Value Histogram', !!nextState.histogram);
    }
    if (controls.opAllProgramsBtn) {
        controls.opAllProgramsBtn.disabled = !handlers || !handlers.toggleAllPrograms;
        updateToggleLabel(controls.opAllProgramsBtn, 'All Program IDs', !!nextState.allPrograms);
    }
}
function applyToggleResult(result, key) {
    if (result && typeof result.then === 'function') {
        result.then((value) => {
            setOpControlState({ [key]: !!value });
        });
    }
    else {
        setOpControlState({ [key]: !!result });
    }
}
try {
    window.setOpControlHandlers = setOpControlHandlers;
    window.setOpControlState = setOpControlState;
    window.resetOpControls = resetOpControls;
}
catch (err) {
    console.warn('Unable to expose op control helpers', err);
}
function initializeApp() {
    appDisposer.dispose();
    pidDisposer.dispose();
    pidDisposer = createDisposer();
    containerElement = document.getElementById('visualization-container');
    controls.panel = document.getElementById('control-panel');
    controls.resizer = document.getElementById('sidebar-resizer');
    controls.pidContainer = document.getElementById('pid-controls');
    controls.resetBtn = document.getElementById('reset-filters');
    controls.precomputeBtn = document.getElementById('btn-precompute');
    controls.themeToggle = document.getElementById('theme-toggle');
    controls.opColorizeBtn = document.getElementById('btn-op-colorize');
    controls.opHistogramBtn = document.getElementById('btn-op-histogram');
    controls.opAllProgramsBtn = document.getElementById('btn-op-all-programs');
    if (!containerElement) {
        console.error('Essential visualization elements are missing.');
        return;
    }
    setupThemeToggle();
    setupControlEvents();
    setupSidebarResizer();
    updateOpControls();
    fetchData();
}
function setupThemeToggle() {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    const defaultTheme = stored || document.documentElement.dataset.theme || 'light';
    applyTheme(defaultTheme);
    if (controls.themeToggle) {
        appDisposer.listen(controls.themeToggle, 'click', () => {
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
}
function setupControlEvents() {
    if (controls.resetBtn) {
        appDisposer.listen(controls.resetBtn, 'click', () => {
            setActiveProgram(0, 0, 0, { syncControls: true, force: true });
            showControlToast('Filters reset');
            logAction('program_reset', { program: { ...getState().activeProgram } });
        });
    }
    if (controls.precomputeBtn) {
        appDisposer.listen(controls.precomputeBtn, 'click', () => {
            showControlToast('Precompute mode is coming soon. This build previews the layout.');
        });
    }
    if (controls.opColorizeBtn) {
        appDisposer.listen(controls.opColorizeBtn, 'click', () => {
            const handler = opControls.handlers?.toggleColorize;
            if (!handler)
                return;
            logAction('toggle_colorize', { next: !getState().toggles.colorize });
            applyToggleResult(handler(), 'colorize');
        });
    }
    if (controls.opHistogramBtn) {
        appDisposer.listen(controls.opHistogramBtn, 'click', () => {
            const handler = opControls.handlers?.toggleHistogram;
            if (!handler)
                return;
            logAction('toggle_histogram', { next: !getState().toggles.histogram });
            applyToggleResult(handler(), 'histogram');
        });
    }
    if (controls.opAllProgramsBtn) {
        appDisposer.listen(controls.opAllProgramsBtn, 'click', () => {
            const handler = opControls.handlers?.toggleAllPrograms;
            if (!handler)
                return;
            logAction('toggle_all_programs', { next: !getState().toggles.allPrograms });
            applyToggleResult(handler(), 'allPrograms');
        });
    }
}
function setupSidebarResizer() {
    if (!controls.resizer || !controls.panel)
        return;
    const root = document.documentElement;
    const minWidth = 0;
    let startX = 0;
    let startWidth = 0;
    const dragDisposer = createDisposer();
    const onPointerMove = (event) => {
        const delta = event.clientX - startX;
        const resizerWidth = controls.resizer.getBoundingClientRect().width || 0;
        const maxWidth = Math.max(0, window.innerWidth - resizerWidth);
        const next = Math.min(maxWidth, Math.max(minWidth, startWidth + delta));
        root.style.setProperty('--sidebar-width', `${next}px`);
    };
    const onPointerUp = (event) => {
        controls.resizer.releasePointerCapture(event.pointerId);
        dragDisposer.dispose();
    };
    appDisposer.listen(controls.resizer, 'pointerdown', (event) => {
        startX = event.clientX;
        startWidth = controls.panel.getBoundingClientRect().width;
        controls.resizer.setPointerCapture(event.pointerId);
        dragDisposer.listen(window, 'pointermove', onPointerMove);
        dragDisposer.listen(window, 'pointerup', onPointerUp);
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
function determineMaxValues(opsData) {
    maxX = 0;
    maxY = 0;
    maxZ = 0;
    Object.keys(opsData || {}).forEach((key) => {
        const [x, y, z] = key
            .replace(/[()]/g, '')
            .split(',')
            .map((s) => Number(String(s).trim()));
        if (Number.isFinite(x))
            maxX = Math.max(maxX, x);
        if (Number.isFinite(y))
            maxY = Math.max(maxY, y);
        if (Number.isFinite(z))
            maxZ = Math.max(maxZ, z);
    });
}
function getBlockDataAt(nx, ny, nz) {
    if (!visualizationData)
        return [];
    const withSpaces = `(${nx}, ${ny}, ${nz})`;
    const withoutSpaces = `(${nx},${ny},${nz})`;
    return (visualizationData[withSpaces] ||
        visualizationData[withoutSpaces] ||
        []);
}
function syncProgramControls() {
    const values = getState().activeProgram;
    for (let i = 0; i < PROGRAM_AXES.length; i += 1) {
        const slider = document.querySelector(`input[data-filter-index="${i}"]`);
        if (slider) {
            slider.value = String(values[PROGRAM_AXES[i]] ?? 0);
        }
        const valuePill = document.getElementById(`pid-value-${i}`);
        if (valuePill)
            valuePill.textContent = String(values[PROGRAM_AXES[i]] ?? 0);
    }
}
function setActiveProgram(x, y, z, { syncControls = false, force = false } = {}) {
    const nextX = Math.max(0, Math.min(maxX, x));
    const nextY = Math.max(0, Math.min(maxY, y));
    const nextZ = Math.max(0, Math.min(maxZ, z));
    setActiveProgramState({ x: nextX, y: nextY, z: nextZ });
    if (syncControls) {
        syncProgramControls();
    }
    if (opWorkspace) {
        opWorkspace.setProgram(nextX, nextY, nextZ, { force });
    }
}
function createProgramIdControls() {
    if (!controls.pidContainer)
        return;
    controls.pidContainer.innerHTML = '';
    pidDisposer.dispose();
    pidDisposer = createDisposer();
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
        const axis = PROGRAM_AXES[index];
        const currentValues = getState().activeProgram;
        const nextValue = isLocked ? 0 : Math.max(0, currentValues[axis] ?? 0);
        setActiveProgramState({ ...currentValues, [axis]: nextValue });
        valueSpan.textContent = String(nextValue);
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = String(maxValues[index]);
        slider.value = String(nextValue);
        slider.dataset.filterIndex = String(index);
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
        pidDisposer.listen(slider, 'input', handleProgramFilterChange);
        wrapper.appendChild(nameSpan);
        wrapper.appendChild(slider);
        wrapper.appendChild(valueSpan);
        wrapper.appendChild(ticks);
        controls.pidContainer.appendChild(wrapper);
    });
}
function handleProgramFilterChange(event) {
    const index = Number(event.target.dataset.filterIndex);
    const value = Number(event.target.value);
    const axis = PROGRAM_AXES[index];
    const current = getState().activeProgram;
    const next = { ...current, [axis]: value };
    logAction('program_slider', { axis, value });
    setActiveProgram(next.x ?? 0, next.y ?? 0, next.z ?? 0, { syncControls: true });
}
function initializeUIElements() {
    if (!globalData || !globalData.ops)
        return;
    visualizationData = globalData.ops.visualization_data || {};
    opWorkspace = new OpWorkspace(containerElement, {
        getBlockData: getBlockDataAt,
        maxValues: { x: maxX, y: maxY, z: maxZ },
    });
    createProgramIdControls();
    setActiveProgram(0, 0, 0, { syncControls: true, force: true });
}
async function fetchData() {
    try {
        globalData = await getJson('/api/data');
        determineMaxValues(globalData?.ops?.visualization_data || {});
        initializeUIElements();
    }
    catch (error) {
        console.error('Error fetching data:', error);
        showControlToast('Failed to load data. Please ensure the backend is running.');
    }
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
}
else {
    initializeApp();
}
