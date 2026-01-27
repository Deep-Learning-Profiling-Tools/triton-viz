import { OpWorkspace } from '../components/op_workspace.js';
import { getJson } from './api.js';
import { getState, resetToggles, setActiveProgram as setActiveProgramState, setToggles } from './state.js';
import { logAction } from './logger.js';
import { createDisposer } from '../utils/dispose.js';
import type { OpRecord, ProgramAxes } from '../types/types.js';

type VisualizationData = Record<string, OpRecord[]>;
type OpControlHandlers = {
    toggleColorize?: () => boolean | Promise<boolean>;
    toggleHistogram?: () => boolean | Promise<boolean>;
    toggleAllPrograms?: (() => boolean | Promise<boolean>) | null;
};
type ToggleKey = 'colorize' | 'histogram' | 'allPrograms' | 'showCode';

let globalData: { ops?: { visualization_data?: VisualizationData } } | null = null;
let visualizationData: VisualizationData | null = null;
let containerElement: HTMLElement | null = null;
let opWorkspace: OpWorkspace | null = null;
let maxX = 0;
let maxY = 0;
let maxZ = 0;
const PROGRAM_AXES: Array<keyof ProgramAxes> = ['x', 'y', 'z'];
const appDisposer = createDisposer();
let pidDisposer = createDisposer();
const THEME_STORAGE_KEY = 'triton-viz-theme';
const DEV_OVERLAY_QUERY_KEY = 'dev';
const DEV_OVERLAY_TOGGLE = 'KeyD';
let devOverlayEnabled = false;
let devOverlayRoot: HTMLDivElement | null = null;

const controls = {
    panel: null as HTMLElement | null,
    pidContainer: null as HTMLElement | null,
    resetBtn: null as HTMLButtonElement | null,
    precomputeBtn: null as HTMLButtonElement | null,
    infoBtn: null as HTMLButtonElement | null,
    themeToggle: null as HTMLButtonElement | null,
    resizer: null as HTMLElement | null,
    opColorizeBtn: null as HTMLButtonElement | null,
    opHistogramBtn: null as HTMLButtonElement | null,
    opAllProgramsBtn: null as HTMLButtonElement | null,
};

let controlToastEl: HTMLDivElement | null = null;
let controlToastTimer: number | null = null;

const opControls = {
    handlers: null as OpControlHandlers | null,
};

function setOpControlState(nextState: Partial<Record<ToggleKey, boolean>> = {}): void {
    const toggles = {
        ...getState().toggles,
        ...nextState,
    };
    setToggles(toggles);
    try { window.__tritonVizOpState = { ...toggles }; } catch (err) {}
    updateOpControls(toggles);
}

function setOpControlHandlers(handlers: OpControlHandlers | null): void {
    opControls.handlers = handlers;
    updateOpControls();
}

function resetOpControls(): void {
    opControls.handlers = null;
    const nextState = resetToggles();
    try { window.__tritonVizOpState = { ...nextState.toggles }; } catch (err) {}
    if (window.__tritonVizCodeHide) {
        window.__tritonVizCodeHide();
    }
    updateOpControls(nextState.toggles);
}

function updateToggleLabel(button: HTMLButtonElement | null, label: string, isOn: boolean): void {
    if (!button) return;
    button.textContent = `${label}: ${isOn ? 'ON' : 'OFF'}`;
    button.classList.toggle('active', isOn);
}

function updateOpControls(state: ReturnType<typeof getState>['toggles'] | null = null): void {
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

function applyToggleResult(result: boolean | Promise<boolean>, key: ToggleKey): void {
    if (result && typeof (result as Promise<boolean>).then === 'function') {
        (result as Promise<boolean>).then((value) => {
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

function initializeApp(): void {
    appDisposer.dispose();
    pidDisposer.dispose();
    pidDisposer = createDisposer();
    containerElement = document.getElementById('visualization-container');
    controls.panel = document.getElementById('control-panel');
    controls.resizer = document.getElementById('sidebar-resizer');
    controls.pidContainer = document.getElementById('pid-controls');
    controls.resetBtn = document.getElementById('reset-filters') as HTMLButtonElement | null;
    controls.precomputeBtn = document.getElementById('btn-precompute') as HTMLButtonElement | null;
    controls.themeToggle = document.getElementById('theme-toggle') as HTMLButtonElement | null;
    controls.opColorizeBtn = document.getElementById('btn-op-colorize') as HTMLButtonElement | null;
    controls.opHistogramBtn = document.getElementById('btn-op-histogram') as HTMLButtonElement | null;
    controls.opAllProgramsBtn = document.getElementById('btn-op-all-programs') as HTMLButtonElement | null;

    if (!containerElement) {
        console.error('Essential visualization elements are missing.');
        return;
    }

    setupThemeToggle();
    setupControlEvents();
    setupSidebarResizer();
    setupDevOverlay();
    updateOpControls();
    fetchData();
}

function setupThemeToggle(): void {
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

function applyTheme(theme: string): void {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    const icon = document.getElementById('theme-toggle-icon');
    if (icon) {
        icon.textContent = theme === 'light' ? '☀' : '🌙';
    }
}

function setupDevOverlay(): void {
    const params = new URLSearchParams(window.location.search);
    const enabledFromQuery = params.get(DEV_OVERLAY_QUERY_KEY) === '1';
    setDevOverlay(enabledFromQuery, 'query');
    appDisposer.listen(window, 'resize', () => {
        if (devOverlayEnabled) updateDevOverlay();
    });
    appDisposer.listen(window, 'scroll', () => {
        if (devOverlayEnabled) updateDevOverlay();
    }, true);
    appDisposer.listen(document, 'keydown', (event) => {
        const keyEvent = event as KeyboardEvent;
        if (!keyEvent.ctrlKey || !keyEvent.shiftKey) return;
        if (keyEvent.code !== DEV_OVERLAY_TOGGLE) return;
        const target = keyEvent.target as HTMLElement | null;
        if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
        keyEvent.preventDefault();
        setDevOverlay(!devOverlayEnabled, 'keyboard');
    });
    const observer = new MutationObserver(() => {
        if (devOverlayEnabled) updateDevOverlay();
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-component'] });
    appDisposer.add(() => observer.disconnect());
}

function setDevOverlay(enabled: boolean, source: string): void {
    if (enabled === devOverlayEnabled) return;
    devOverlayEnabled = enabled;
    if (!enabled) {
        if (devOverlayRoot) devOverlayRoot.remove();
        devOverlayRoot = null;
        logAction('dev_overlay_toggle', { enabled, source });
        return;
    }
    if (!devOverlayRoot) {
        devOverlayRoot = document.createElement('div');
        devOverlayRoot.className = 'dev-overlay';
        document.body.appendChild(devOverlayRoot);
    }
    updateDevOverlay();
    logAction('dev_overlay_toggle', { enabled, source });
}

function updateDevOverlay(): void {
    const root = devOverlayRoot;
    if (!root) return;
    root.innerHTML = '';
    const nodes = document.querySelectorAll('[data-component]');
    nodes.forEach((node) => {
        const label = node.getAttribute('data-component');
        if (!label) return;
        const rect = node.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;
        const badge = document.createElement('div');
        badge.className = 'dev-overlay-badge';
        badge.textContent = label;
        badge.style.left = `${Math.max(0, rect.left + 6)}px`;
        badge.style.top = `${Math.max(0, rect.top + 6)}px`;
        root.appendChild(badge);
    });
}

function setupControlEvents(): void {
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
            if (!handler) return;
            logAction('toggle_colorize', { next: !getState().toggles.colorize });
            applyToggleResult(handler(), 'colorize');
        });
    }

    if (controls.opHistogramBtn) {
        appDisposer.listen(controls.opHistogramBtn, 'click', () => {
            const handler = opControls.handlers?.toggleHistogram;
            if (!handler) return;
            logAction('toggle_histogram', { next: !getState().toggles.histogram });
            applyToggleResult(handler(), 'histogram');
        });
    }

    if (controls.opAllProgramsBtn) {
        appDisposer.listen(controls.opAllProgramsBtn, 'click', () => {
            const handler = opControls.handlers?.toggleAllPrograms;
            if (!handler) return;
            logAction('toggle_all_programs', { next: !getState().toggles.allPrograms });
            applyToggleResult(handler(), 'allPrograms');
        });
    }
}

function setupSidebarResizer(): void {
    if (!controls.resizer || !controls.panel) return;
    const resizer = controls.resizer;
    const panel = controls.panel;
    const root = document.documentElement;
    const minWidth = 0;
    let startX = 0;
    let startWidth = 0;
    const dragDisposer = createDisposer();

    const onPointerMove = (event: Event): void => {
        const pointerEvent = event as PointerEvent;
        const delta = pointerEvent.clientX - startX;
        const resizerWidth = resizer.getBoundingClientRect().width || 0;
        const maxWidth = Math.max(0, window.innerWidth - resizerWidth);
        const next = Math.min(maxWidth, Math.max(minWidth, startWidth + delta));
        root.style.setProperty('--sidebar-width', `${next}px`);
    };

    const onPointerUp = (event: Event): void => {
        const pointerEvent = event as PointerEvent;
        resizer.releasePointerCapture(pointerEvent.pointerId);
        dragDisposer.dispose();
    };

    appDisposer.listen(resizer, 'pointerdown', (event: Event) => {
        const pointerEvent = event as PointerEvent;
        startX = pointerEvent.clientX;
        startWidth = panel.getBoundingClientRect().width;
        resizer.setPointerCapture(pointerEvent.pointerId);
        dragDisposer.listen(window, 'pointermove', onPointerMove);
        dragDisposer.listen(window, 'pointerup', onPointerUp);
    });
}

function showControlToast(message: string): void {
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
    if (controlToastTimer) clearTimeout(controlToastTimer);
    controlToastTimer = window.setTimeout(() => {
        if (controlToastEl) {
            controlToastEl.style.opacity = '0';
        }
    }, 3200);
}

function determineMaxValues(opsData: VisualizationData | null): void {
    maxX = 0;
    maxY = 0;
    maxZ = 0;
    Object.keys(opsData || {}).forEach((key) => {
        const parts = key
            .replace(/[()]/g, '')
            .split(',')
            .map((s) => Number(String(s).trim()));
        const x = parts[0] ?? NaN;
        const y = parts[1] ?? NaN;
        const z = parts[2] ?? NaN;
        if (Number.isFinite(x)) maxX = Math.max(maxX, x);
        if (Number.isFinite(y)) maxY = Math.max(maxY, y);
        if (Number.isFinite(z)) maxZ = Math.max(maxZ, z);
    });
}

function getBlockDataAt(nx: number, ny: number, nz: number): OpRecord[] {
    if (!visualizationData) return [];
    const withSpaces = `(${nx}, ${ny}, ${nz})`;
    const withoutSpaces = `(${nx},${ny},${nz})`;
    return (
        visualizationData[withSpaces] ||
        visualizationData[withoutSpaces] ||
        []
    );
}

function syncProgramControls(): void {
    const values = getState().activeProgram;
    for (let i = 0; i < PROGRAM_AXES.length; i += 1) {
        const slider = document.querySelector(`input[data-filter-index="${i}"]`) as HTMLInputElement | null;
        if (slider) {
            const axis = PROGRAM_AXES[i];
            if (axis) slider.value = String(values[axis] ?? 0);
        }
        const valuePill = document.getElementById(`pid-value-${i}`);
        if (valuePill) {
            const axis = PROGRAM_AXES[i];
            valuePill.textContent = String(axis ? values[axis] ?? 0 : 0);
        }
    }
}

function setActiveProgram(
    x: number,
    y: number,
    z: number,
    { syncControls = false, force = false }: { syncControls?: boolean; force?: boolean } = {},
): void {
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

function createProgramIdControls(): void {
    const pidContainer = controls.pidContainer;
    if (!pidContainer) return;
    pidContainer.innerHTML = '';
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
        const maxValue = maxValues[index] ?? 0;
        const isLocked = maxValue <= 0;
        const axis = PROGRAM_AXES[index];
        if (!axis) return;
        const currentValues = getState().activeProgram;
        const nextValue = isLocked ? 0 : Math.max(0, currentValues[axis] ?? 0);
        setActiveProgramState({ ...currentValues, [axis]: nextValue });
        valueSpan.textContent = String(nextValue);
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = String(maxValue);
        slider.value = String(nextValue);
        slider.dataset.filterIndex = String(index);
        slider.disabled = isLocked;
        const tickId = `pid-ticks-${index}`;
        const ticks = document.createElement('datalist');
        ticks.id = tickId;
        for (let i = 0; i <= maxValue; i += 1) {
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
        pidContainer.appendChild(wrapper);
    });
}

function handleProgramFilterChange(event: Event): void {
    const target = event.target as HTMLInputElement | null;
    if (!target) return;
    const index = Number(target.dataset.filterIndex);
    const value = Number(target.value);
    const axis = PROGRAM_AXES[index];
    if (!axis) return;
    const current = getState().activeProgram;
    const next = { ...current, [axis]: value };
    logAction('program_slider', { axis, value });
    setActiveProgram(next.x ?? 0, next.y ?? 0, next.z ?? 0, { syncControls: true });
}

function initializeUIElements(): void {
    if (!globalData || !globalData.ops) return;
    visualizationData = globalData.ops.visualization_data || {};
    opWorkspace = new OpWorkspace(containerElement, {
        getBlockData: getBlockDataAt,
        maxValues: { x: maxX, y: maxY, z: maxZ },
    });
    createProgramIdControls();
    setActiveProgram(0, 0, 0, { syncControls: true, force: true });
}

async function fetchData(): Promise<void> {
    try {
        globalData = await getJson('/api/data');
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
