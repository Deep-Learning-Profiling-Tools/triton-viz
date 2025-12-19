import { createMatMulVisualization } from './matmul.js';
import { createFlipVisualization } from './flip.js';
import { createLoadVisualization, createLoadOverallVisualization } from './load.js';
import { createStoreVisualization, createStoreOverallVisualization } from './store.js';
import { createFlowDiagram } from './nki.js';
import { enableDrag } from './ui_helpers.js';

const DEFAULT_PALETTE = {
    blockBg: '#323232',
    blockHoverBg: '#4a4a4a',
    blockBorder: 'rgba(0, 0, 0, 0.5)',
    textPrimary: '#f8fafc',
    textSecondary: '#cbd5f5',
};

export class GridBlock {
    constructor(x, y, width, height, gridX, gridY, gridZ, blockData, onClose, containerElement, canvas, drawFunction) {
        this.rect = { x, y, width, height };
        this.gridPosition = { x: gridX, y: gridY, z: gridZ };
        this.blockData = blockData;
        this.isHovered = false;
        this.visualizationContainer = null;
        this.visualizationCleanupFunction = null;
        this.onClose = onClose;
        this.containerElement = containerElement;
        this.canvas = canvas;
        this.drawFunction = drawFunction;
        this.isDetailedViewVisible = false;
        this.contentArea = null;
        this.activeTab = null;
    }

    draw(ctx, palette = DEFAULT_PALETTE) {
        const theme = { ...DEFAULT_PALETTE, ...(palette || {}) };
        ctx.fillStyle = this.isHovered ? theme.blockHoverBg : theme.blockBg;
        ctx.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        ctx.strokeStyle = theme.blockBorder;
        ctx.lineWidth = 1;
        ctx.strokeRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);

        ctx.fillStyle = theme.textSecondary;
        ctx.font = '12px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const posText = `${this.gridPosition.x},${this.gridPosition.y},${this.gridPosition.z}`;
        ctx.fillText(posText, this.rect.x + this.rect.width / 2, this.rect.y + 4);

        ctx.fillStyle = theme.textPrimary;
        ctx.font = '10px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        this.blockData.forEach((op, index) => {
            ctx.fillText(op.type, this.rect.x + 6, this.rect.y + 22 + index * 14);
        });
    }

    isPointInside(x, y) {
        return x >= this.rect.x && x <= this.rect.x + this.rect.width &&
            y >= this.rect.y && y <= this.rect.y + this.rect.height;
    }

    handleMouseMove(x, y) {
        const wasHovered = this.isHovered;
        this.isHovered = this.isPointInside(x, y);
        return wasHovered !== this.isHovered;
    }

    showDetailedView() {
        if (this.isDetailedViewVisible) return;

        this.visualizationContainer = this.createVisualizationContainer();
        const card = document.createElement('div');
        card.className = 'detail-card';

        const titleRow = this.createTitleRow();
        const headerBar = this.createHeaderBar();
        this.contentArea = this.createContentArea();

        card.appendChild(titleRow);
        card.appendChild(headerBar);
        card.appendChild(this.contentArea);
        this.visualizationContainer.appendChild(card);

        const closeButton = this.createCloseButton();
        const backButton = this.createBackButton();
        this.visualizationContainer.appendChild(closeButton);
        this.visualizationContainer.appendChild(backButton);

        if (this.containerElement) {
            this.containerElement.innerHTML = '';
            this.containerElement.appendChild(this.visualizationContainer);
            this.containerElement.style.display = 'block';
            this.containerElement.style.pointerEvents = 'auto';
        } else {
            document.body.appendChild(this.visualizationContainer);
        }

        this.isDetailedViewVisible = true;
        if (this.canvas) this.canvas.style.display = 'none';

        if (this.blockData.length > 0) {
            this.displayOpVisualization(this.blockData[0]);
        }

        this.ensureGlobalCodeToggle();
        if (window.setGlobalCodeButtonVisible) {
            window.setGlobalCodeButtonVisible(true);
        }
    }

    ensureGlobalCodeToggle() {
        const codeBtnId = 'global-code-toggle-btn';
        if (document.getElementById(codeBtnId)) {
            document.getElementById(codeBtnId).style.display = 'block';
            return;
        }
        const btn = document.createElement('button');
        btn.id = codeBtnId;
        btn.textContent = 'Show Code: OFF';
        btn.className = 'show-code-btn';
        document.body.appendChild(btn);

        let panel = null;
        const destroyPanel = () => {
            if (panel && panel.remove) panel.remove();
            panel = null;
        };

        const createPanel = async (uuid, frameIdx = 0, context = 8) => {
            destroyPanel();
            const wrapper = document.createElement('div');
            wrapper.className = 'show-code-panel';
            const header = document.createElement('div');
            header.className = 'panel-header drag-handle';
            header.style.fontWeight = '600';
            header.style.marginBottom = '8px';
            header.innerHTML = '<span>Operation Code & Context</span><span class="drag-grip" aria-hidden="true">⠿</span>';
            const closeBtn = document.createElement('button');
            closeBtn.className = 'viz-button ghost';
            closeBtn.textContent = 'Close';
            closeBtn.style.marginLeft = 'auto';
            closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
            closeBtn.addEventListener('click', () => {
                destroyPanel();
                btn.textContent = 'Show Code: OFF';
            });
            header.appendChild(closeBtn);
            wrapper.appendChild(header);
            try {
                const API_BASE = window.__TRITON_VIZ_API__ || '';
                const res = await fetch(`${API_BASE}/api/op_code`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ uuid, frame_idx: frameIdx, context })
                });
                const data = await res.json();
                const meta = document.createElement('div');
                meta.style.marginBottom = '6px';
                meta.style.fontSize = '12px';
                meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
                wrapper.appendChild(meta);
                const pre = document.createElement('pre');
                pre.style.margin = '0';
                const lines = (data.lines || []).map((line) => {
                    const mark = (data.highlight === line.no) ? '▶ ' : '  ';
                    return `${mark}${String(line.no).padStart(6, ' ')} | ${line.text || ''}`;
                }).join('\n');
                pre.textContent = lines || '(no code available)';
                wrapper.appendChild(pre);
            } catch (error) {
                const err = document.createElement('div');
                err.textContent = 'Failed to load code context.';
                wrapper.appendChild(err);
            }
            document.body.appendChild(wrapper);
            panel = wrapper;
            enableDrag(wrapper, { handle: header, bounds: window });
        };

        btn.addEventListener('click', async () => {
            const turnOn = btn.textContent.endsWith('OFF');
            btn.textContent = `Show Code: ${turnOn ? 'ON' : 'OFF'}`;
            if (!this.blockData || this.blockData.length === 0) {
                if (!turnOn) destroyPanel();
                return;
            }
            const activeUuid = window.current_op_uuid || (this.blockData[0] && this.blockData[0].uuid);
            if (turnOn && activeUuid) {
                await createPanel(activeUuid, 0, 8);
            } else {
                destroyPanel();
            }
        });
    }

    createVisualizationContainer() {
        const container = document.createElement('div');
        container.className = 'detail-overlay';
        return container;
    }

    createTitleRow() {
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
        return row;
    }

    createHeaderBar() {
        const headerBar = document.createElement('div');
        headerBar.className = 'detail-tabs';
        this.activeTab = null;

        this.blockData.forEach((op, index) => {
            const opTab = this.createOperationTab(op);
            opTab.addEventListener('click', () => this.handleTabClick(opTab, op));
            headerBar.appendChild(opTab);
            if (index === 0) {
                this.setActiveTab(opTab);
            }
        });

        const flowTab = this.createTabButton('Flow');
        flowTab.addEventListener('click', () => {
            this.setActiveTab(flowTab);
            this.displayFlowDiagram();
        });
        headerBar.appendChild(flowTab);

        const loadOps = this.blockData.filter((op) => op.type === 'Load' && op.overall_key);
        if (loadOps.length) {
            const loadTab = this.createTabButton('Load Overall');
            loadTab.addEventListener('click', async () => {
                this.setActiveTab(loadTab);
                await this.displayLoadOverallView(loadOps);
            });
            headerBar.appendChild(loadTab);
        }

        const storeOps = this.blockData.filter((op) => op.type === 'Store' && op.overall_key);
        if (storeOps.length) {
            const storeTab = this.createTabButton('Store Overall');
            storeTab.addEventListener('click', async () => {
                this.setActiveTab(storeTab);
                await this.displayStoreOverallView(storeOps);
            });
            headerBar.appendChild(storeTab);
        }

        return headerBar;
    }

    createOperationTab(op) {
        const pieces = [
            op.type,
            (op.global_shape || []).join('×'),
            typeof op.overall_key === 'string' ? op.overall_key : ''
        ].filter(Boolean);
        const tab = this.createTabButton(pieces.join(' • '));
        return tab;
    }

    createTabButton(label) {
        const btn = document.createElement('button');
        btn.className = 'detail-tab';
        btn.textContent = label;
        return btn;
    }

    setActiveTab(tab) {
        if (this.activeTab === tab) return;
        if (this.activeTab) {
            this.activeTab.classList.remove('is-active');
        }
        this.activeTab = tab;
        if (this.activeTab) {
            this.activeTab.classList.add('is-active');
        }
    }

    handleTabClick(clickedTab, op) {
        this.setActiveTab(clickedTab);
        this.displayOpVisualization(op);
    }

    createContentArea() {
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

    displayOpVisualization(op) {
        if (!this.contentArea) {
            console.error('Content area is not initialized');
            return;
        }
        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }
        this.contentArea.innerHTML = '';
        try {
            window.last_op = op;
            window.last_op_global_shape = op.global_shape;
            window.last_global_coords = op.global_coords;
            window.last_slice_shape = op.slice_shape;
            window.last_slice_coords = op.slice_coords;
        } catch (error) { /* noop */ }

        switch (op.type) {
            case 'Dot':
                this.visualizationCleanupFunction = createMatMulVisualization(this.contentArea, op);
                break;
            case 'Load':
                this.visualizationCleanupFunction = createLoadVisualization(this.contentArea, op);
                break;
            case 'Store':
                this.visualizationCleanupFunction = createStoreVisualization(this.contentArea, op);
                break;
            case 'Flip':
                this.visualizationCleanupFunction = createFlipVisualization(this.contentArea, op);
                break;
            default:
                this.contentArea.textContent = `Visualization not supported for ${op.type} operations.`;
        }
    }

    displayFlowDiagram() {
        if (!this.contentArea) return;
        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }
        this.contentArea.innerHTML = '';
        this.visualizationCleanupFunction = createFlowDiagram(this.contentArea, this.blockData || []);
    }

    async displayLoadOverallView(loadOps) {
        if (!this.contentArea) return;
        this.contentArea.innerHTML = '<div class="overall-empty">Loading load overview…</div>';
        try {
            const data = await this.fetchOverallData(loadOps.map((op) => op.overall_key), 'load');
            if (this.visualizationCleanupFunction) {
                this.visualizationCleanupFunction();
                this.visualizationCleanupFunction = null;
            }
            const base = loadOps[0] || {};
            const opPayload = {
                ...base,
                overall_mode: true,
                overall_tiles: data.tiles || [],
                overall_shape: data.shape || base.global_shape,
                overall_slice_shape: data.slice_shape || base.slice_shape,
            };
            this.visualizationCleanupFunction = createLoadOverallVisualization(this.contentArea, opPayload);
        } catch (err) {
            this.contentArea.innerHTML = `<div class="overall-empty">Failed to load overall view: ${err}</div>`;
        }
    }

    async displayStoreOverallView(storeOps) {
        if (!this.contentArea) return;
        this.contentArea.innerHTML = '<div class="overall-empty">Loading store overview…</div>';
        try {
            const data = await this.fetchOverallData(storeOps.map((op) => op.overall_key), 'store');
            if (this.visualizationCleanupFunction) {
                this.visualizationCleanupFunction();
                this.visualizationCleanupFunction = null;
            }
            const base = storeOps[0] || {};
            const opPayload = {
                ...base,
                overall_mode: true,
                overall_tiles: data.tiles || [],
                overall_shape: data.shape || base.global_shape,
                overall_slice_shape: data.slice_shape || base.slice_shape,
            };
            this.visualizationCleanupFunction = createStoreOverallVisualization(this.contentArea, opPayload);
        } catch (err) {
            this.contentArea.innerHTML = `<div class="overall-empty">Failed to load overall view: ${err}</div>`;
        }
    }

    async fetchOverallData(keys, kind) {
        const uniqueKeys = Array.from(new Set((keys || []).filter(Boolean)));
        if (!uniqueKeys.length) {
            throw new Error('No overall data available');
        }
        const API_BASE = window.__TRITON_VIZ_API__ || '';
        const endpoint = kind === 'store' ? 'store_overall' : 'load_overall';
        const results = await Promise.all(uniqueKeys.map(async (key) => {
            const resp = await fetch(`${API_BASE}/api/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key })
            });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                throw new Error(data && data.error ? data.error : 'Request failed');
            }
            return data;
        }));
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

    createCloseButton() {
        const closeButton = document.createElement('button');
        closeButton.className = 'detail-close';
        closeButton.textContent = 'Close';
        closeButton.addEventListener('click', () => this.hideDetailedView());
        return closeButton;
    }

    createBackButton() {
        const backButton = document.createElement('button');
        backButton.className = 'detail-back';
        backButton.textContent = 'Back to grid';
        backButton.addEventListener('click', () => this.hideDetailedView());
        return backButton;
    }

    hideDetailedView() {
        if (!this.isDetailedViewVisible) return;

        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }

        if (this.containerElement && this.visualizationContainer) {
            this.containerElement.innerHTML = '';
            this.containerElement.style.display = 'none';
            this.containerElement.style.pointerEvents = 'none';
        }
        if (!this.containerElement && this.visualizationContainer && this.visualizationContainer.parentNode) {
            this.visualizationContainer.parentNode.removeChild(this.visualizationContainer);
        }
        this.visualizationContainer = null;

        this.isDetailedViewVisible = false;

        if (this.onClose) {
            this.onClose();
        }

        if (this.canvas) this.canvas.style.display = 'block';
        this.drawFunction();
        if (window.setGlobalCodeButtonVisible) {
            window.setGlobalCodeButtonVisible(false);
        }
    }
}
