import { createMatMulVisualization } from './matmul.js';
import { createFlipVisualization } from './flip.js';
import { createLoadVisualization } from './load.js';
import { createStoreVisualization } from './store.js';
import { createFlowDiagram } from './nki.js';

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

        const title = this.createTitle();
        const headerBar = this.createHeaderBar();
        this.contentArea = this.createContentArea();

        this.visualizationContainer.appendChild(title);
        this.visualizationContainer.appendChild(headerBar);
        this.visualizationContainer.appendChild(this.contentArea);

        const closeButton = this.createCloseButton();
        this.visualizationContainer.appendChild(closeButton);
        // Add an explicit Back button (top-left) to return to main canvas view
        const backButton = this.createBackButton();
        this.visualizationContainer.appendChild(backButton);

        // Ensure buttons panel sits above the canvas and accepts clicks
        const buttonsPanel = this.visualizationContainer.querySelector('div');
        if (buttonsPanel) {
            buttonsPanel.style.pointerEvents = 'auto';
            buttonsPanel.style.zIndex = '1002';
        }

        this.isDetailedViewVisible = true;
        this.canvas.style.display = 'none';
        this.containerElement.style.display = 'block';

        // Display the first operation visualization after the content area is added to the DOM
        if (this.blockData.length > 0) {
            this.displayOpVisualization(this.blockData[0]);
        }

        // Add a global "Show Code" toggle once (applies to any op type)
        const codeBtnId = 'global-code-toggle-btn';
        if (!document.getElementById(codeBtnId)) {
            const btn = document.createElement('button');
            btn.id = codeBtnId;
            btn.textContent = 'Show Code: OFF';
            Object.assign(btn.style, {
                position: 'fixed',
                right: '10px',
                top: '10px',
                zIndex: '2001'
            });
            document.body.appendChild(btn);

            let panel = null;
            const destroyPanel = () => { if (panel && panel.remove) panel.remove(); panel = null; };
            const createPanel = async (uuid, frameIdx = 0, context = 8) => {
                destroyPanel();
                const wrapper = document.createElement('div');
                Object.assign(wrapper.style, {
                    position: 'fixed', right: '10px', top: '50px', width: '520px', maxHeight: '60vh', overflow: 'auto',
                    padding: '8px 10px', background: 'rgba(0,0,0,0.65)', color: '#fff', font: '12px Menlo, Consolas, monospace',
                    borderRadius: '6px', zIndex: '2000'
                });
                const header = document.createElement('div');
                header.textContent = 'Operation Code & Context';
                header.style.marginBottom = '6px';
                header.style.opacity = '0.9';
                wrapper.appendChild(header);
                try {
                    const API_BASE = window.__TRITON_VIZ_API__ || '';
                    const res = await fetch(`${API_BASE}/api/op_code`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ uuid, frame_idx: frameIdx, context }) });
                    const data = await res.json();
                    const meta = document.createElement('div');
                    meta.style.marginBottom = '4px';
                    meta.textContent = `${data.filename || ''}:${data.lineno || ''}`;
                    wrapper.appendChild(meta);
                    const pre = document.createElement('pre');
                    pre.style.margin = '0';
                    pre.style.whiteSpace = 'pre';
                    const lines = (data.lines || []).map(l => {
                        const mark = (data.highlight === l.no) ? '▶ ' : '  ';
                        return `${mark}${String(l.no).padStart(6,' ')} | ${l.text||''}`;
                    }).join('\n');
                    pre.textContent = lines || '(no code available)';
                    wrapper.appendChild(pre);
                } catch (e) {
                    const err = document.createElement('div');
                    err.textContent = 'Failed to load code context.';
                    wrapper.appendChild(err);
                }
                document.body.appendChild(wrapper);
                panel = wrapper;
            };

            btn.addEventListener('click', async () => {
                const turnOn = btn.textContent.endsWith('OFF');
                btn.textContent = `Show Code: ${turnOn ? 'ON' : 'OFF'}`;
                if (!this.blockData || this.blockData.length === 0) { if (!turnOn) destroyPanel(); return; }
                // Prefer latest clicked/active uuid; fallback to first in current grid block
                const activeUuid = window.current_op_uuid || (this.blockData[0] && this.blockData[0].uuid);
                if (turnOn && activeUuid) await createPanel(activeUuid, 0, 8); else destroyPanel();
            });
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
        title.textContent = `Operations at (${this.gridPosition.x}, ${this.gridPosition.y}, ${this.gridPosition.z})`;
        title.style.textAlign = 'center';
        title.style.margin = '10px 0';
        return title;
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

        let currentSelectedTab = null;
        // Tabs for each op
        this.blockData.forEach((op, index) => {
            const opTab = this.createOperationTab(op, index === 0);
            opTab.addEventListener('click', () => this.handleTabClick(opTab, op, currentSelectedTab));
            headerBar.appendChild(opTab);
            if (index === 0) currentSelectedTab = opTab;
        });
        // Extra: NKI view tab (aggregates all ops)
        const nkiTab = document.createElement('button');
        nkiTab.textContent = 'Flow';
        Object.assign(nkiTab.style, { flex:'0 0 auto', marginRight:'5px', background:'#333', color:'#fff', border:'none', padding:'10px', cursor:'pointer' });
        nkiTab.addEventListener('click', () => {
            if (currentSelectedTab) currentSelectedTab.style.backgroundColor = '#333';
            nkiTab.style.backgroundColor = '#555';
            this.displayFlowDiagram();
        });
        headerBar.appendChild(nkiTab);

        return headerBar;
    }

    createOperationTab(op, isFirst) {
        const opTab = document.createElement('button');
        opTab.textContent = op.type;
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

    handleTabClick(clickedTab, op, currentSelectedTab) {
        if (currentSelectedTab) currentSelectedTab.style.backgroundColor = '#333';
        clickedTab.style.backgroundColor = '#555';
        this.displayOpVisualization(op);
    }

    createContentArea() {
        const contentArea = document.createElement('div');
        Object.assign(contentArea.style, {
            flex: '1',
            padding: '10px',
            overflow: 'hidden',
            position: 'relative'
        });

        if (this.blockData.length === 0) {
            const noDataMsg = document.createElement('p');
            noDataMsg.textContent = 'No operation data available';
            noDataMsg.style.textAlign = 'center';
            contentArea.appendChild(noDataMsg);
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
                const unsupportedMsg = document.createElement('p');
                unsupportedMsg.textContent = `Visualization not supported for ${op.type} operation`;
                unsupportedMsg.style.textAlign = 'center';
                this.contentArea.appendChild(unsupportedMsg);
        }
    }

    displayFlowDiagram() {
        if (!this.contentArea) return;
        if (this.visualizationCleanupFunction) { this.visualizationCleanupFunction(); this.visualizationCleanupFunction = null; }
        this.contentArea.innerHTML = '';
        // Pass the entire block data (ops in this program) to Flow view
        this.visualizationCleanupFunction = createFlowDiagram(this.contentArea, this.blockData || []);
    }


    createCloseButton() {
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        Object.assign(closeButton.style, {
            position: 'fixed',
            top: '10px',
            right: '10px',
            zIndex: '1001'
        });
        closeButton.addEventListener('click', () => this.hideDetailedView());
        return closeButton;
    }

    createBackButton() {
        const backButton = document.createElement('button');
        backButton.textContent = 'Back';
        Object.assign(backButton.style, {
            position: 'fixed',
            left: '10px',
            bottom: '10px',
            zIndex: '2002',
            background: 'rgba(0,0,0,0.65)',
            color: '#fff',
            border: '1px solid #666',
            padding: '6px 10px',
            borderRadius: '6px',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.5)'
        });
        backButton.addEventListener('click', () => this.hideDetailedView());
        return backButton;
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

        this.canvas.style.display = 'block';
        this.containerElement.style.display = 'none';
        this.drawFunction();
    }
}
