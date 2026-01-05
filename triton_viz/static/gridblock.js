import { createMatMulVisualization } from './matmul.js';
import { createFlipVisualization } from './flip.js';
import { createLoadVisualization } from './load.js';
import { createStoreVisualization } from './store.js';
import { createFlowDiagram } from './nki.js';

export class GridBlock {
    constructor(x, y, width, height, gridX, gridY, gridZ, blockData, onClose, containerElement, canvas, drawFunction, programIdConfig = null) {
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
        this.currentSelectedTab = null;
        this.programIdConfig = programIdConfig;
        this.programIdControls = null;
        this.programIdInputs = null;
        this.programIdValueEls = null;
        this.titleEl = null;
        this.bodyArea = null;
        this.headerBar = null;
        this.mainArea = null;
        this.codeSidebar = null;
        this.codeSidebarHeader = null;
        this.codeSidebarMeta = null;
        this.codeSidebarPre = null;
        this.codeSidebarVisible = true;
        this.codeFetchToken = 0;
        this.sidebarWidth = 420;
        this.splitterEl = null;
        this.activeOpType = null;
        this.activeOpIndex = null;
        this.viewState = { camera: null };
    }


    draw(ctx) {
        // Draw background
        ctx.fillStyle = this.isHovered ? '#4a4a4a' : '#323232';
        ctx.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);

        // Draw border
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.strokeRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);

        // Draw grid position
        ctx.fillStyle = '#c8c8c8';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const posText = `${this.gridPosition.x},${this.gridPosition.y},${this.gridPosition.z}`;
        ctx.fillText(posText, this.rect.x + this.rect.width / 2, this.rect.y + 2);

        // Draw operation types
        ctx.font = '10px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        this.blockData.forEach((op, index) => {
            ctx.fillText(op.type, this.rect.x + 5, this.rect.y + 20 + index * 15);
        });
    }

    isPointInside(x, y) {
        return x >= this.rect.x && x <= this.rect.x + this.rect.width &&
               y >= this.rect.y && y <= this.rect.y + this.rect.height;
    }

    handleMouseMove(x, y) {
        const wasHovered = this.isHovered;
        this.isHovered = this.isPointInside(x, y);
        return wasHovered !== this.isHovered; // Return true if hover state changed
    }


    showDetailedView() {
        if (this.isDetailedViewVisible) return;

        this.visualizationContainer = this.createVisualizationContainer();
        document.body.appendChild(this.visualizationContainer);

        this.programIdControls = this.createProgramIdControls();
        this.bodyArea = document.createElement('div');
        Object.assign(this.bodyArea.style, {
            display: 'flex',
            flexDirection: 'column',
            flex: '1',
            minWidth: '0',
            minHeight: '0'
        });
        this.mainArea = document.createElement('div');
        Object.assign(this.mainArea.style, {
            display: 'flex',
            flex: '1',
            minHeight: '0',
            minWidth: '0'
        });
        this.codeSidebar = this.createCodeSidebar();
        if (this.codeSidebar) {
            this.titleEl = this.createTitle();
            if (this.titleEl) {
                this.codeSidebar.insertBefore(this.titleEl, this.codeSidebar.firstChild);
            }
            if (this.programIdControls) this.codeSidebar.appendChild(this.programIdControls);
            this.mainArea.appendChild(this.codeSidebar);
            this.setSidebarWidth(this.sidebarWidth);
            this.splitterEl = this.createSplitter();
            this.mainArea.appendChild(this.splitterEl);
        }
        this.mainArea.appendChild(this.bodyArea);

        this.visualizationContainer.appendChild(this.mainArea);

        this.isDetailedViewVisible = true;
        if (this.canvas) this.canvas.style.display = 'none';
        this.containerElement.style.display = 'block';

        this.setCodeSidebarVisible(true);
        this.refreshBody();
    }

    applyProgramIdSelection(x, y, z) {
        if (!this.programIdConfig) return;
        const { max } = this.programIdConfig;
        const nx = Math.max(0, Math.min(max.x, x));
        const ny = Math.max(0, Math.min(max.y, y));
        const nz = Math.max(0, Math.min(max.z, z));
        if (nx === this.gridPosition.x && ny === this.gridPosition.y && nz === this.gridPosition.z) return;
        this.gridPosition = { x: nx, y: ny, z: nz };
        this.programIdConfig.values = { x: nx, y: ny, z: nz };
        if (this.programIdConfig.getBlockData) {
            this.blockData = this.programIdConfig.getBlockData(nx, ny, nz);
        }
        if (this.titleEl) {
            this.titleEl.textContent = 'Triton Visualizer';
        }
        if (this.programIdInputs && this.programIdValueEls) {
            this.programIdInputs.x.value = String(nx);
            this.programIdInputs.y.value = String(ny);
            this.programIdInputs.z.value = String(nz);
            this.programIdValueEls.x.textContent = String(nx);
            this.programIdValueEls.y.textContent = String(ny);
            this.programIdValueEls.z.textContent = String(nz);
        }
        this.refreshBody(false);
    }

    refreshBody(updateCode = true) {
        if (!this.bodyArea) return;
        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }
        this.bodyArea.innerHTML = '';
        this.headerBar = this.createHeaderBar();
        this.contentArea = this.createContentArea();
        this.bodyArea.appendChild(this.headerBar);
        this.bodyArea.appendChild(this.contentArea);
        if (this.blockData.length > 0) {
            let nextOp = null;
            if (this.activeOpIndex !== null && this.activeOpIndex < this.blockData.length) {
                nextOp = this.blockData[this.activeOpIndex];
            }
            if (!nextOp && this.activeOpType) {
                nextOp = this.blockData.find(op => op.type === this.activeOpType);
            }
            if (!nextOp) nextOp = this.blockData[0];
            if (nextOp) {
                const nextIndex = this.blockData.indexOf(nextOp);
                const tab = this.headerBar.querySelector(`button[data-op-index="${nextIndex}"]`);
                if (tab) this.setActiveTab(tab);
                this.activeOpType = nextOp.type;
                this.activeOpIndex = nextIndex;
                this.displayOpVisualization(nextOp, updateCode);
            }
            return;
        }
        if (updateCode) this.updateCodeSidebar(null);
    }

    setCodeSidebarVisible(visible) {
        this.codeSidebarVisible = visible;
        if (this.codeSidebar) {
            this.codeSidebar.style.display = visible ? 'flex' : 'none';
        }
        if (this.splitterEl) {
            this.splitterEl.style.display = visible ? 'block' : 'none';
        }
        window.triton_viz_code_sidebar_visible = visible;
    }

    setSidebarWidth(width) {
        if (!this.codeSidebar) return;
        const px = Math.round(width);
        this.sidebarWidth = px;
        this.codeSidebar.style.width = `${px}px`;
        this.codeSidebar.style.flex = `0 0 ${px}px`;
    }

    createSplitter() {
        const splitter = document.createElement('div');
        Object.assign(splitter.style, {
            flex: '0 0 6px',
            width: '6px',
            cursor: 'col-resize',
            background: 'rgba(255, 255, 255, 0.06)'
        });
        splitter.addEventListener('mousedown', (event) => {
            if (!this.codeSidebar) return;
            event.preventDefault();
            const startX = event.clientX;
            const startWidth = this.codeSidebar.getBoundingClientRect().width;
            const minWidth = 240;
            const maxWidth = Math.min(720, window.innerWidth * 0.6);
            document.body.style.userSelect = 'none';
            const onMove = (moveEvent) => {
                const next = startWidth + (moveEvent.clientX - startX);
                this.setSidebarWidth(Math.max(minWidth, Math.min(maxWidth, next)));
            };
            const onUp = () => {
                document.body.style.userSelect = '';
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
            };
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        });
        return splitter;
    }

    createCodeSidebar() {
        const wrapper = document.createElement('div');
        Object.assign(wrapper.style, {
            flex: '0 0 420px',
            width: '420px',
            background: '#101019',
            color: '#fff',
            padding: '10px 12px',
            borderRight: '1px solid #2f2f38',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            overflow: 'auto',
            height: '100%'
        });
        const header = document.createElement('div');
        header.textContent = 'Operation Code & Context';
        header.style.opacity = '0.9';
        wrapper.appendChild(header);
        const meta = document.createElement('div');
        meta.style.opacity = '0.7';
        wrapper.appendChild(meta);
        const pre = document.createElement('pre');
        pre.style.margin = '0';
        pre.style.whiteSpace = 'pre';
        pre.style.font = '12px Menlo, Consolas, monospace';
        pre.style.lineHeight = '1.35';
        pre.style.overflow = 'auto';
        wrapper.appendChild(pre);
        this.codeSidebarHeader = header;
        this.codeSidebarMeta = meta;
        this.codeSidebarPre = pre;
        return wrapper;
    }

    async updateCodeSidebar(op) {
        if (!this.codeSidebarMeta || !this.codeSidebarPre) return;
        if (!op || !op.uuid) {
            this.codeSidebarMeta.textContent = '';
            this.codeSidebarPre.textContent = 'No operation selected.';
            return;
        }
        const token = ++this.codeFetchToken;
        this.codeSidebarMeta.textContent = '';
        this.codeSidebarPre.textContent = 'Loading...';
        try {
            const API_BASE = window.__TRITON_VIZ_API__ || '';
            const res = await fetch(`${API_BASE}/api/op_code`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uuid: op.uuid, frame_idx: 0, context: 8 })
            });
            const data = await res.json();
            if (token !== this.codeFetchToken) return;
            this.codeSidebarMeta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
            const lines = (data.lines || []).map(l => {
                const mark = (data.highlight === l.no) ? '▶ ' : '  ';
                return `${mark}${String(l.no).padStart(6, ' ')} | ${l.text || ''}`;
            }).join('\n');
            this.codeSidebarPre.textContent = lines || '(no code available)';
        } catch (e) {
            if (token !== this.codeFetchToken) return;
            this.codeSidebarMeta.textContent = '';
            this.codeSidebarPre.textContent = 'Failed to load code context.';
        }
    }

    createVisualizationContainer() {
        const container = document.createElement('div');
        Object.assign(container.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            width: '100vw',
            height: '100vh',
            backgroundColor: '#1e1e28',
            zIndex: '1000',
            display: 'flex',
            flexDirection: 'column',
            color: '#ffffff'
        });
        return container;
    }

    createTitle() {
        const title = document.createElement('h2');
        title.textContent = 'Triton Visualizer';
        title.style.textAlign = 'left';
        title.style.margin = '4px 0 6px 0';
        return title;
    }

    createProgramIdControls() {
        if (!this.programIdConfig) return null;
        const { max, values } = this.programIdConfig;
        const wrapper = document.createElement('div');
        Object.assign(wrapper.style, {
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            padding: '12px',
            background: '#242430',
            borderTop: '1px solid #3a3a44',
            flex: '0 0 auto',
            width: '100%',
            overflow: 'auto',
            height: 'auto',
            boxSizing: 'border-box'
        });

        this.programIdInputs = {};
        this.programIdValueEls = {};

        const title = document.createElement('div');
        title.textContent = 'Program ID';
        title.style.fontWeight = '600';
        wrapper.appendChild(title);

        const createRow = (label, axis, maxValue, initialValue) => {
            const row = document.createElement('div');
            Object.assign(row.style, {
                display: 'grid',
                gridTemplateColumns: '28px 1fr 36px',
                alignItems: 'center',
                columnGap: '8px',
                width: '100%'
            });
            const text = document.createElement('span');
            text.textContent = label;
            text.style.textAlign = 'right';
            const input = document.createElement('input');
            input.type = 'range';
            input.min = '0';
            input.max = String(maxValue);
            input.value = String(initialValue);
            input.style.width = '100%';
            input.style.minWidth = '0';
            const isFixed = maxValue === 0;
            if (isFixed) {
                input.disabled = true;
                input.style.opacity = '0.6';
                input.style.cursor = 'not-allowed';
            }
            const valueEl = document.createElement('span');
            valueEl.textContent = String(initialValue);
            valueEl.style.textAlign = 'left';
            input.addEventListener('input', () => {
                valueEl.textContent = input.value;
                const nextX = Number(this.programIdInputs.x.value);
                const nextY = Number(this.programIdInputs.y.value);
                const nextZ = Number(this.programIdInputs.z.value);
                this.applyProgramIdSelection(nextX, nextY, nextZ);
            });
            row.appendChild(text);
            row.appendChild(input);
            row.appendChild(valueEl);
            wrapper.appendChild(row);
            this.programIdInputs[axis] = input;
            this.programIdValueEls[axis] = valueEl;
        };

        const initial = values || this.gridPosition;
        createRow('X:', 'x', max.x, initial.x);
        createRow('Y:', 'y', max.y, initial.y);
        createRow('Z:', 'z', max.z, initial.z);

        return wrapper;
    }

    createHeaderBar() {
        const headerBar = document.createElement('div');
        Object.assign(headerBar.style, {
            display: 'flex',
            flexDirection: 'row',
            backgroundColor: '#333',
            padding: '5px',
            overflowX: 'auto'
        });

        this.currentSelectedTab = null;
        // tabs for each op
        this.blockData.forEach((op, index) => {
            const opTab = this.createOperationTab(op, index === 0, index);
            opTab.addEventListener('click', () => this.handleTabClick(opTab, op));
            headerBar.appendChild(opTab);
            if (index === 0) this.currentSelectedTab = opTab;
        });
        // Extra: NKI view tab (aggregates all ops)
        const nkiTab = document.createElement('button');
        nkiTab.textContent = 'Flow';
        Object.assign(nkiTab.style, { flex:'0 0 auto', marginRight:'5px', background:'#333', color:'#fff', border:'none', padding:'10px', cursor:'pointer' });
        nkiTab.addEventListener('click', () => {
            this.setActiveTab(nkiTab);
            this.displayFlowDiagram();
        });
        headerBar.appendChild(nkiTab);

        return headerBar;
    }

    createOperationTab(op, isFirst, index) {
        const opTab = document.createElement('button');
        opTab.textContent = op.type;
        opTab.dataset.opType = op.type;
        opTab.dataset.opIndex = String(index);
        Object.assign(opTab.style, {
            flex: '0 0 auto',
            marginRight: '5px',
            backgroundColor: isFirst ? '#555' : '#333',
            color: '#fff',
            border: 'none',
            padding: '10px',
            cursor: 'pointer'
        });
        return opTab;
    }

    setActiveTab(tab) {
        if (this.currentSelectedTab) this.currentSelectedTab.style.backgroundColor = '#333';
        this.currentSelectedTab = tab;
        if (tab) tab.style.backgroundColor = '#555';
    }

    handleTabClick(clickedTab, op) {
        this.activeOpType = op.type;
        this.activeOpIndex = clickedTab && clickedTab.dataset ? Number(clickedTab.dataset.opIndex) : null;
        this.setActiveTab(clickedTab);
        this.displayOpVisualization(op);
    }

    createContentArea() {
        const contentArea = document.createElement('div');
        Object.assign(contentArea.style, {
            flex: '1',
            padding: '10px',
            overflow: 'hidden',
            position: 'relative',
            minHeight: '0'
        });

        if (this.blockData.length === 0) {
            const noDataMsg = document.createElement('p');
            noDataMsg.textContent = 'No operation data available';
            noDataMsg.style.textAlign = 'center';
            contentArea.appendChild(noDataMsg);
        }

        return contentArea;
    }
    displayOpVisualization(op, updateCode = true) {
        if (!this.contentArea) {
            console.error('Content area is not initialized');
            return;
        }
        this.activeOpType = op.type;

        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }

        this.contentArea.innerHTML = '';

        // expose current op for debugging
        try {
            window.last_op = op;
            window.last_op_global_shape = op.global_shape;
            window.last_global_coords = op.global_coords;
            window.last_slice_shape = op.slice_shape;
            window.last_slice_coords = op.slice_coords;
        } catch (e) {}

        switch (op.type) {
            case 'Dot':
                this.visualizationCleanupFunction = createMatMulVisualization(this.contentArea, op, this.viewState);
                break;
            case 'Load':
                this.visualizationCleanupFunction = createLoadVisualization(this.contentArea, op, this.viewState);
                break;
            case 'Store':
                this.visualizationCleanupFunction = createStoreVisualization(this.contentArea, op, this.viewState);
                break;
            case 'Flip':
                this.visualizationCleanupFunction = createFlipVisualization(this.contentArea, op, this.viewState);
                break;
            default:
                const unsupportedMsg = document.createElement('p');
                unsupportedMsg.textContent = `Visualization not supported for ${op.type} operation`;
                unsupportedMsg.style.textAlign = 'center';
                this.contentArea.appendChild(unsupportedMsg);
        }
        if (updateCode) this.updateCodeSidebar(op);
    }

    displayFlowDiagram() {
        if (!this.contentArea) return;
        if (this.visualizationCleanupFunction) { this.visualizationCleanupFunction(); this.visualizationCleanupFunction = null; }
        this.contentArea.innerHTML = '';
        // Pass the entire block data (ops in this program) to Flow view
        this.visualizationCleanupFunction = createFlowDiagram(this.contentArea, this.blockData || []);
        this.updateCodeSidebar(null);
    }


    hideDetailedView() {
        if (!this.isDetailedViewVisible) return;

        if (this.visualizationCleanupFunction) {
            this.visualizationCleanupFunction();
            this.visualizationCleanupFunction = null;
        }

        if (this.visualizationContainer) {
            document.body.removeChild(this.visualizationContainer);
            this.visualizationContainer = null;
        }

        this.isDetailedViewVisible = false;

        if (this.onClose) {
            this.onClose();
        }
    }
}
