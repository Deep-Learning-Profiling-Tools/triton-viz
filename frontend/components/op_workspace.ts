import { createFlowDiagram } from '../ops/nki.js';
import { enableDrag } from '../utils/ui_helpers.js';
import { postJson } from '../core/api.js';
import { logAction, logInfo } from '../core/logger.js';
import { setActiveOp } from '../core/state.js';
import { getVisualizer } from '../ops/registry.js';
import '../ops/defaults.js';
import type { OpCodePayload, OpRecord } from '../types/types.js';

export class OpWorkspace {
    containerElement: HTMLElement | null;
    getBlockData: ((x: number, y: number, z: number) => OpRecord[]) | null;
    maxValues: { x: number; y: number; z: number };
    gridPosition: { x: number; y: number; z: number };
    blockData: OpRecord[];
    visualizationContainer: HTMLElement | null;
    visualizationCleanupFunction: (() => void) | null;
    contentArea: (HTMLElement & { __vizGetState?: () => unknown }) | null;
    activeTab: HTMLElement | null;
    activeTabIndex: number;
    lastOpType: string | null;
    titleEl: HTMLElement | null;
    badgeEl: HTMLElement | null;
    cardEl: HTMLElement | null;
    headerBar: HTMLElement | null;

    constructor(
        containerElement: HTMLElement | null,
        { getBlockData = null, maxValues = { x: 0, y: 0, z: 0 } }: {
            getBlockData?: ((x: number, y: number, z: number) => OpRecord[]) | null;
            maxValues?: { x: number; y: number; z: number };
        } = {},
    ) {
        this.containerElement = containerElement;
        this.getBlockData = getBlockData;
        this.maxValues = maxValues;
        this.gridPosition = { x: 0, y: 0, z: 0 };
        this.blockData = [];
        this.visualizationContainer = null;
        this.visualizationCleanupFunction = null;
        this.contentArea = null;
        this.activeTab = null;
        this.activeTabIndex = 0;
        this.lastOpType = null;
        this.titleEl = null;
        this.badgeEl = null;
        this.cardEl = null;
        this.headerBar = null;
        this.initialize();
    }

    initialize(): void {
        if (!this.containerElement) return;
        this.visualizationContainer = this.createVisualizationContainer();
        const card = document.createElement('div');
        card.className = 'detail-card';
        this.cardEl = card;

        const titleRow = this.createTitleRow();
        this.headerBar = this.createHeaderBar();
        this.contentArea = this.createContentArea();

        card.appendChild(titleRow);
        card.appendChild(this.headerBar);
        card.appendChild(this.contentArea);
        this.visualizationContainer.appendChild(card);

        this.containerElement.innerHTML = '';
        this.containerElement.appendChild(this.visualizationContainer);
        this.containerElement.style.display = 'block';
        this.containerElement.style.pointerEvents = 'auto';

        this.ensureGlobalCodeToggle();
        try {
            window.__tritonVizActiveBlock = this;
        } catch (error) {}
    }

    setProgram(x: number, y: number, z: number, { force = false }: { force?: boolean } = {}): void {
        const max = this.maxValues || { x: 0, y: 0, z: 0 };
        const nx = Math.max(0, Math.min(max.x ?? 0, x));
        const ny = Math.max(0, Math.min(max.y ?? 0, y));
        const nz = Math.max(0, Math.min(max.z ?? 0, z));
        const changed = nx !== this.gridPosition.x || ny !== this.gridPosition.y || nz !== this.gridPosition.z;
        if (!changed && !force) return;
        this.gridPosition = { x: nx, y: ny, z: nz };
        if (this.getBlockData) {
            this.blockData = this.getBlockData(nx, ny, nz);
        } else {
            this.blockData = [];
        }
        if (changed) {
            logInfo('active program changed', { x: nx, y: ny, z: nz });
            if (window.resetOpControls) window.resetOpControls();
        }
        if (this.titleEl) {
            this.titleEl.textContent = `Program (${nx}, ${ny}, ${nz})`;
        }
        if (this.badgeEl) {
            this.badgeEl.textContent = `${this.blockData.length} ops`;
        }
        if (this.headerBar && this.cardEl) {
            this.headerBar.remove();
            this.headerBar = this.createHeaderBar();
            this.cardEl.insertBefore(this.headerBar, this.contentArea);
        }
        if (this.blockData.length > 0) {
            const nextIndex = Math.min(this.activeTabIndex || 0, this.blockData.length - 1);
            this.activeTabIndex = nextIndex;
            const nextOp = this.blockData[nextIndex];
            if (nextOp) {
                this.displayOpVisualization(nextOp, {
                    preserveViewState: true,
                    logTabChange: true,
                });
            } else {
                this.renderEmptyState();
            }
        } else {
            this.renderEmptyState();
        }
    }

    createVisualizationContainer(): HTMLElement {
        const container = document.createElement('div');
        container.className = 'detail-overlay';
        return container;
    }

    createTitleRow(): HTMLElement {
        const row = document.createElement('div');
        row.className = 'overlay-title-row';
        const title = document.createElement('h2');
        title.className = 'detail-title';
        title.textContent = `Program (${this.gridPosition.x}, ${this.gridPosition.y}, ${this.gridPosition.z})`;
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = `${this.blockData.length} ops`;
        row.appendChild(title);
        row.appendChild(badge);
        this.titleEl = title;
        this.badgeEl = badge;
        return row;
    }

    createHeaderBar(): HTMLElement {
        const headerBar = document.createElement('div');
        headerBar.className = 'detail-tabs';
        this.activeTab = null;

        let defaultTab = null;
        let preferredTab = null;
        this.blockData.forEach((op, index) => {
            const opTab = this.createOperationTab(op);
            opTab.addEventListener('click', () => this.handleTabClick(opTab, op, index));
            headerBar.appendChild(opTab);
            if (index === 0) {
                defaultTab = opTab;
            }
            if (this.activeTabIndex === index) {
                preferredTab = opTab;
            }
        });
        if (preferredTab) {
            this.setActiveTab(preferredTab);
        } else if (defaultTab) {
            this.setActiveTab(defaultTab);
        }

        const flowTab = this.createTabButton('Flow');
        flowTab.addEventListener('click', () => {
            this.setActiveTab(flowTab);
            this.displayFlowDiagram({ logTabChange: true });
            logAction('flow_tab_click', { program: { ...this.gridPosition } });
        });
        headerBar.appendChild(flowTab);

        return headerBar;
    }

    createOperationTab(op: OpRecord): HTMLButtonElement {
        const pieces = [
            op.type,
            (op.global_shape || []).join('x'),
            typeof op.overall_key === 'string' ? op.overall_key : '',
        ].filter(Boolean);
        const tab = this.createTabButton(pieces.join(' - '));
        return tab;
    }

    createTabButton(label: string): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.className = 'detail-tab';
        btn.textContent = label;
        return btn;
    }

    setActiveTab(tab: HTMLElement): void {
        if (this.activeTab === tab) return;
        if (this.activeTab) {
            this.activeTab.classList.remove('is-active');
        }
        this.activeTab = tab;
        if (this.activeTab) {
            this.activeTab.classList.add('is-active');
        }
    }

    handleTabClick(clickedTab: HTMLElement, op: OpRecord, index: number | null = null): void {
        this.setActiveTab(clickedTab);
        if (typeof index === 'number') {
            this.activeTabIndex = index;
        }
        logAction('op_tab_click', {
            program: { ...this.gridPosition },
            opType: op.type,
            index,
        });
        this.displayOpVisualization(op, { preserveViewState: true, logTabChange: true });
    }

    createContentArea(): HTMLElement {
        const contentArea = document.createElement('div');
        contentArea.className = 'detail-content';
        if (this.blockData.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'overall-empty';
            empty.textContent = 'No operation data available.';
            contentArea.appendChild(empty);
        }
        return contentArea;
    }

    renderEmptyState(): void {
        if (!this.contentArea) return;
        this.contentArea.innerHTML = '';
        const empty = document.createElement('div');
        empty.className = 'overall-empty';
        empty.textContent = 'No operation data available.';
        this.contentArea.appendChild(empty);
        if (window.resetOpControls) window.resetOpControls();
    }

    logTabChange(opType: string): void {
        logInfo('op tab changed', {
            program: {
                x: this.gridPosition.x,
                y: this.gridPosition.y,
                z: this.gridPosition.z,
            },
            opType,
        });
    }

    displayOpVisualization(
        op: OpRecord,
        options: {
            viewState?: unknown;
            preserveViewState?: boolean;
            logTabChange?: boolean;
            refreshCodePanel?: boolean;
        } = {},
    ): void {
        if (!this.contentArea) {
            console.error('Content area is not initialized');
            return;
        }
        let viewState = options.viewState || null;
        const canReuseViz = options.preserveViewState && this.contentArea.__vizGetState;
        if (!viewState && canReuseViz) {
            viewState = this.contentArea.__vizGetState?.() ?? null;
        }
        const preserveCodePanel = options.refreshCodePanel === false;
        if (preserveCodePanel) {
            try { window.__tritonVizPreserveCodePanel = true; } catch (error) {}
        }
        const isTensorOp = op.type === 'Dot' || op.type === 'Load' || op.type === 'Store';
        const reuseTensorView = canReuseViz && isTensorOp && this.lastOpType === op.type;
        try {
            if (!reuseTensorView && this.visualizationCleanupFunction) {
                this.visualizationCleanupFunction();
                this.visualizationCleanupFunction = null;
            }
        } finally {
            if (preserveCodePanel) {
                try { window.__tritonVizPreserveCodePanel = false; } catch (error) {}
            }
        }
        if (!reuseTensorView) {
            this.contentArea.innerHTML = '';
        }
        try {
            window.last_op = op;
            if (op.global_shape) window.last_op_global_shape = op.global_shape;
            else if (window.last_op_global_shape) delete window.last_op_global_shape;
            if (op.global_coords) window.last_global_coords = op.global_coords;
            else if (window.last_global_coords) delete window.last_global_coords;
            if (op.slice_shape) window.last_slice_shape = op.slice_shape;
            else if (window.last_slice_shape) delete window.last_slice_shape;
            if (op.slice_coords) window.last_slice_coords = op.slice_coords;
            else if (window.last_slice_coords) delete window.last_slice_coords;
        } catch (error) { /* noop */ }

        const visualizer = getVisualizer(op.type);
        if (visualizer) {
            const cleanup = visualizer(this.contentArea, op, viewState);
            this.visualizationCleanupFunction = typeof cleanup === 'function' ? cleanup : null;
        } else {
            this.contentArea.textContent = `Visualization not supported for ${op.type} operations.`;
            this.visualizationCleanupFunction = null;
        }
        this.lastOpType = op.type;
        setActiveOp({ type: op.type, uuid: op.uuid });
        if (options.logTabChange) {
            this.logTabChange(op.type);
        }
        if (options.refreshCodePanel !== false) {
            this.syncCodePanel(true);
        }
    }

    displayFlowDiagram({ logTabChange = false }: { logTabChange?: boolean } = {}): void {
        if (!this.contentArea) return;
        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }
        this.contentArea.innerHTML = '';
        this.visualizationCleanupFunction = createFlowDiagram(this.contentArea, this.blockData || []);
        setActiveOp({ type: 'Flow', uuid: null });
        if (logTabChange) {
            this.logTabChange('Flow');
        }
    }

    ensureGlobalCodeToggle(): void {
        if (window.__tritonVizCodeToggle) return;
        let panel: HTMLDivElement | null = null;
        let visible = false;

        const destroyPanel = (): void => {
            if (panel && panel.remove) panel.remove();
            const host = document.getElementById('op-code-panel');
            if (host && host.contains(panel)) {
                host.innerHTML = '';
            }
            panel = null;
        };

        const getActiveUuid = (): string | null => {
            const activeBlock = window.__tritonVizActiveBlock;
            const candidate = window.current_op_uuid ?? activeBlock?.blockData?.[0]?.uuid;
            return candidate ?? null;
        };

        const createPanel = async (uuid: string, frameIdx = 0, context = 8): Promise<void> => {
            destroyPanel();
            const wrapper = document.createElement('div');
            wrapper.className = 'show-code-panel';
            const header = document.createElement('div');
            header.className = 'panel-header drag-handle';
            header.style.fontWeight = '600';
            header.style.marginBottom = '8px';
            header.innerHTML = '<span>Code View</span>';
            wrapper.appendChild(header);
            const blurb = document.createElement('div');
            blurb.style.fontSize = '12px';
            blurb.style.color = 'var(--text-secondary)';
            blurb.style.marginBottom = '8px';
            blurb.textContent = 'Arrow points to the line being visualized.';
            wrapper.appendChild(blurb);
            try {
                const data = await postJson<OpCodePayload>('/api/op_code', { uuid, frame_idx: frameIdx, context });
                const meta = document.createElement('div');
                meta.style.marginBottom = '6px';
                meta.style.fontSize = '12px';
                meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
                wrapper.appendChild(meta);
                const pre = document.createElement('pre');
                pre.style.margin = '0';
                const lines = (data.lines || []).map((line) => {
                    const mark = (data.highlight === line.no) ? '> ' : '  ';
                    return `${mark}${String(line.no).padStart(6, ' ')} | ${line.text || ''}`;
                }).join('\n');
                pre.textContent = lines || '(no code available)';
                wrapper.appendChild(pre);
            } catch (error) {
                const err = document.createElement('div');
                err.textContent = 'Failed to load code context.';
                wrapper.appendChild(err);
            }
            const host = document.getElementById('op-code-panel');
            if (host) {
                host.innerHTML = '';
                wrapper.classList.add('is-sidebar');
                host.appendChild(wrapper);
            } else {
                document.body.appendChild(wrapper);
                enableDrag(wrapper, { handle: header, bounds: window });
            }
            panel = wrapper;
        };

        const togglePanel = async (force?: boolean): Promise<boolean> => {
            const next = typeof force === 'boolean' ? force : !visible;
            if (!next) {
                visible = false;
                destroyPanel();
                logAction('code_peek_toggle', { visible, uuid: null, source: 'toggle' });
                return visible;
            }
            const activeUuid = getActiveUuid();
            if (!activeUuid) {
                visible = false;
                destroyPanel();
                logAction('code_peek_toggle', { visible, uuid: null, source: 'toggle' });
                return visible;
            }
            visible = true;
            await createPanel(activeUuid, 0, 8);
            logAction('code_peek_toggle', { visible, uuid: activeUuid, source: typeof force === 'boolean' ? 'sync' : 'toggle' });
            return visible;
        };

        const hidePanel = (): boolean => {
            visible = false;
            destroyPanel();
            logAction('code_peek_toggle', { visible, uuid: null, source: 'toggle' });
            return visible;
        };

        window.__tritonVizCodeToggle = togglePanel;
        window.__tritonVizCodeHide = hidePanel;
        window.__tritonVizCodeVisible = () => visible;
    }

    syncCodePanel(forceShow = false): void {
        if (!window.__tritonVizCodeToggle) return;
        const isVisible = window.__tritonVizCodeVisible ? window.__tritonVizCodeVisible() : false;
        if (!forceShow && !isVisible) return;
        const result = window.__tritonVizCodeToggle(true);
        const setOpControlState = window.setOpControlState;
        if (!setOpControlState) return;
        if (result && typeof (result as any).then === 'function') {
            (result as Promise<boolean>).then((visible) => {
                setOpControlState({ showCode: !!visible });
            });
        } else {
            setOpControlState({ showCode: !!result });
        }
    }
}
