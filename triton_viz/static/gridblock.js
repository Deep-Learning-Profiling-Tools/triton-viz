import { createMatMulVisualization } from './matmul.js';
import { createLoadVisualization } from './load.js';
import { createStoreVisualization } from './store.js';

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

        this.isDetailedViewVisible = true;
        this.canvas.style.display = 'none';
        this.containerElement.style.display = 'block';

        // Display the first operation visualization after the content area is added to the DOM
        if (this.blockData.length > 0) {
            this.displayOpVisualization(this.blockData[0]);
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
        this.blockData.forEach((op, index) => {
            const opTab = this.createOperationTab(op, index === 0);
            opTab.addEventListener('click', () => this.handleTabClick(opTab, op, currentSelectedTab));
            headerBar.appendChild(opTab);
            if (index === 0) currentSelectedTab = opTab;
        });

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
            default:
                const unsupportedMsg = document.createElement('p');
                unsupportedMsg.textContent = `Visualization not supported for ${op.type} operation`;
                unsupportedMsg.style.textAlign = 'center';
                this.contentArea.appendChild(unsupportedMsg);
        }
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
