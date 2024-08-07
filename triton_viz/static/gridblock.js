import {createMatMulVisualization} from './matmul.js'
import {createLoadVisualization} from './load.js';
import {createStoreVisualization} from './store.js';

export class GridBlock {
    constructor(x, y, width, height, gridX, gridY, gridZ, blockData) {
        this.rect = { x, y, width, height };
        this.gridPosition = { x: gridX, y: gridY, z: gridZ };
        this.blockData = blockData;
        this.isHovered = false;
        this.visualizationContainer = null;
        this.cleanupFunction = null;

        // Three.js properties
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.matMulVisualization = null;

        // 2D canvas for additional information
        this.canvas2D = null;
        this.ctx2D = null;

        // Animation frame ID
        this.animationFrameId = null;
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

    showDetailedView(containerElement) {
        if (this.isDetailedViewVisible) return;
    
        this.visualizationContainer = containerElement;
        this.visualizationContainer.innerHTML = '';
        this.visualizationContainer.style.pointerEvents = 'auto';
    
        const wrapper = document.createElement('div');
        wrapper.style.padding = '10px';
        wrapper.style.backgroundColor = '#1e1e28';
        wrapper.style.color = '#ffffff';
        wrapper.style.height = '100%';
        wrapper.style.overflowY = 'auto';
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.alignItems = 'stretch';
        wrapper.style.width = '100%';
    
        const title = document.createElement('h2');
        title.textContent = `Operations at (${this.gridPosition.x}, ${this.gridPosition.y}, ${this.gridPosition.z})`;
        title.style.textAlign = 'center';
        title.style.margin = '10px 0';
        wrapper.appendChild(title);
    
        const columnsContainer = document.createElement('div');
        columnsContainer.style.display = 'flex';
        columnsContainer.style.justifyContent = 'space-between';
        columnsContainer.style.gap = '10px';
        columnsContainer.style.flex = '1';
    
        const leftColumn = document.createElement('div');
        leftColumn.style.flex = '1';
        leftColumn.style.display = 'flex';
        leftColumn.style.flexDirection = 'column';
        leftColumn.style.gap = '10px';
    
        const rightColumn = document.createElement('div');
        rightColumn.style.flex = '1';
        rightColumn.style.display = 'flex';
        rightColumn.style.flexDirection = 'column';
        rightColumn.style.gap = '10px';
    
        this.blockData.forEach((op, index) => {
            const opContainer = document.createElement('div');
            opContainer.style.padding = '10px';
            opContainer.style.border = '1px solid #444';
            opContainer.style.borderRadius = '5px';
            opContainer.style.display = 'flex';
            opContainer.style.flexDirection = 'column';
            opContainer.style.flex = '1';
    
            const opTitle = document.createElement('h3');
            opTitle.textContent = `Operation ${index + 1}: ${op.type}`;
            opTitle.style.margin = '0 0 10px 0';
            opContainer.appendChild(opTitle);
    
            const visualizationDiv = document.createElement('div');
            visualizationDiv.style.flex = '1';
            visualizationDiv.style.minHeight = '300px';
            opContainer.appendChild(visualizationDiv);
    
            switch(op.type) {
                case 'Dot':
                    createMatMulVisualization(visualizationDiv, op);
                    break;
                case 'Load':
                    createLoadVisualization(visualizationDiv, op);
                    break;
                case 'Store':
                    createStoreVisualization(visualizationDiv, op);
                    break;
                default:
                    const unsupportedMsg = document.createElement('p');
                    unsupportedMsg.textContent = `Unsupported operation type: ${op.type}`;
                    visualizationDiv.appendChild(unsupportedMsg);
            }
    
            if (index % 2 === 0) {
                leftColumn.appendChild(opContainer);
            } else {
                rightColumn.appendChild(opContainer);
            }
        });
    
        columnsContainer.appendChild(leftColumn);
        columnsContainer.appendChild(rightColumn);
        wrapper.appendChild(columnsContainer);
    
        if (this.blockData.length === 0) {
            const noDataMsg = document.createElement('p');
            noDataMsg.textContent = 'No operation data available';
            noDataMsg.style.textAlign = 'center';
            wrapper.appendChild(noDataMsg);
        }
    
        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.style.position = 'fixed';
        closeButton.style.top = '10px';
        closeButton.style.right = '10px';
        closeButton.addEventListener('click', () => this.hideDetailedView());
    
        this.visualizationContainer.appendChild(wrapper);
        this.visualizationContainer.appendChild(closeButton);
    
        this.visualizationContainer.tabIndex = 0;
        this.visualizationContainer.focus();
    
        this.isDetailedViewVisible = true;
        this.cleanupFunction = this.hideDetailedView.bind(this);
    
        // Trigger a resize event to ensure visualizations render correctly
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 100);
    }

    hideDetailedView() {
        if (!this.isDetailedViewVisible) return;

        if (this.visualizationContainer) {
            this.visualizationContainer.innerHTML = ''; // Clear the container
            this.visualizationContainer.style.pointerEvents = 'none'; // Disable interaction
        }

        this.isDetailedViewVisible = false;
        this.cleanupFunction = null;
    }

    
    animate() {
        this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
        
        if (this.matMulVisualization && this.matMulVisualization.update) {
            this.matMulVisualization.update();
        }
        
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
        
        this.drawTensorView(this.ctx2D, this.canvas2D);
    }

    onWindowResize() {
        if (this.camera && this.renderer && this.visualizationContainer) {
            const width = this.visualizationContainer.clientWidth;
            const height = this.visualizationContainer.clientHeight;
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
            if (this.canvas2D) {
                this.canvas2D.width = width;
                this.canvas2D.height = height;
            }
        }
    }

    drawTensorView(ctx, canvas) {
        if (!ctx || !canvas) return;

        // Clear the 2D canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw tensor view content
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('Matrix Multiplication Visualization', canvas.width / 2, 10);

        // Display block information
        ctx.font = '18px Arial';
        ctx.fillText(`Block Position: (${this.gridPosition.x}, ${this.gridPosition.y}, ${this.gridPosition.z})`, canvas.width / 2, 40);

        const dotOperation = this.blockData.find(op => op.type === 'Dot');
        if (dotOperation) {
            // Display matrix dimensions
            ctx.font = '16px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(`Matrix A: ${dotOperation.input_shape.join('x')}`, 10, 70);
            ctx.fillText(`Matrix B: ${dotOperation.other_shape.join('x')}`, 10, 90);
            ctx.fillText(`Result Matrix: ${dotOperation.output_shape.join('x')}`, 10, 110);
        }

        // Add a back button
        ctx.fillStyle = 'rgba(200, 200, 200, 0.8)';
        ctx.fillRect(10, 10, 60, 30);
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Back', 40, 25);
    }

    hideDetailedView() {

        if (this.matMulVisualization && this.matMulVisualization.cleanup) {
            this.matMulVisualization.cleanup();
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        if (this.visualizationContainer && this.visualizationContainer.parentNode) {
            this.visualizationContainer.parentNode.removeChild(this.visualizationContainer);
        }
        window.removeEventListener('resize', this.onWindowResize.bind(this));

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.matMulVisualization = null;
        this.canvas2D = null;
        this.ctx2D = null;
        this.visualizationContainer = null;
        this.animationFrameId = null;
        this.cleanupFunction = null;

        
    }
}