import { GridBlock } from './gridblock.js';
import { createInfoPopup, showInfoPopup } from './infoPopup.js';
let globalData;
let currentView = 'main';
let canvas, ctx;
let maxX = 0, maxY = 0, maxZ = 0;
let sliders = [], zSlider, precomputeButton, kernelGrid;
let backButton;
let currentBlockData = null;
let isInitialized = false;
let containerElement;
let infoPopup;
let infoButton;
function switchToMainView() {
    currentView = 'main';
    if (currentBlockData) {
        currentBlockData.hideDetailedView();
        currentBlockData = null;
    }
    containerElement.style.pointerEvents = 'none';
    containerElement.style.display = 'none';
    containerElement.innerHTML = '';

    canvas.style.display = 'block';
    draw();
}

function initializeApp() {
    canvas = document.getElementById('canvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }
    ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Unable to get 2D context from canvas');
        return;
    }
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    canvas.addEventListener('mousedown', handleMouseEvent);
    canvas.addEventListener('mouseup', handleMouseEvent);
    canvas.addEventListener('mousemove', handleMouseEvent);

    containerElement = document.getElementById('visualization-container');
    if (!containerElement) {
        console.error('Visualization container element not found');
        return;
    }

    containerElement.style.pointerEvents = 'none';
    containerElement.style.display = 'none';

    fetchData();
}

class Slider {
    constructor(x, y, width, height, label, min_value = -1, max_value = 100) {
        this.rect = { x, y, width, height };
        this.label = label;
        this.min = min_value;
        this.max = max_value;
        this.value = min_value;
        this.grabbed = false;
        this.enabled = true;
    }

    draw(ctx) {
        if (!this.enabled) return;
        ctx.fillStyle = '#3c3c46';
        ctx.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        const buttonX = this.rect.x + (this.value - this.min) / (this.max - this.min) * this.rect.width;
        ctx.fillStyle = '#c8c8c8';
        ctx.fillRect(buttonX - 5, this.rect.y - 2, 10, this.rect.height + 4);

        ctx.fillStyle = '#c8c8c8';
        ctx.font = '18px Arial';
        ctx.fillText(this.label, this.rect.x, this.rect.y - 10);
        ctx.fillText(this.value.toString(), this.rect.x + this.rect.width + 10, this.rect.y + this.rect.height / 2 + 5);
    }

    handleEvent(event) {
        if (!this.enabled) return;
        if (event.type === 'mousedown') {
            if (this.isPointInside(event.offsetX, event.offsetY)) {
                this.grabbed = true;
            }
        } else if (event.type === 'mouseup') {
            this.grabbed = false;
        } else if (event.type === 'mousemove' && this.grabbed) {
            const mouseX = event.offsetX;
            this.value = Math.round((mouseX - this.rect.x) / this.rect.width * (this.max - this.min) + this.min);
            this.value = Math.max(this.min, Math.min(this.max, this.value));
        }
    }

    isPointInside(x, y) {
        return x >= this.rect.x && x <= this.rect.x + this.rect.width &&
               y >= this.rect.y && y <= this.rect.y + this.rect.height;
    }
}

class Button {
    constructor(x, y, width, height, text, isIcon = false) {
        this.rect = { x, y, width, height };
        this.text = text;
        this.isIcon = isIcon;
        this.color = '#3c3c46';
        this.hoverColor = '#50505a';
        this.clickColor = '#64646e';
        this.isHovered = false;
        this.isClicked = false;
        this.clickTime = 0;
    }

    draw(ctx) {
        let color = this.color;
        if (this.isClicked && Date.now() - this.clickTime < 100) {
            color = this.clickColor;
        } else if (this.isHovered) {
            color = this.hoverColor;
        }

        ctx.fillStyle = color;
        ctx.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        ctx.strokeStyle = '#c8c8c8';
        ctx.strokeRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);

        ctx.fillStyle = '#c8c8c8';
        ctx.font = this.isIcon ? 'bold 24px Arial' : '18px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.text, this.rect.x + this.rect.width / 2, this.rect.y + this.rect.height / 2);
    }

    handleEvent(event) {
        const { offsetX, offsetY } = event;
        this.isHovered = this.isPointInside(offsetX, offsetY);
        if (event.type === 'mousedown' && this.isHovered) {
            this.isClicked = true;
            this.clickTime = Date.now();
            console.log(`Button '${this.text}' clicked!`);
        } else if (event.type === 'mouseup') {
            this.isClicked = false;
        }
    }

    isPointInside(x, y) {
        return x >= this.rect.x && x <= this.rect.x + this.rect.width &&
               y >= this.rect.y && y <= this.rect.y + this.rect.height;
    }
}

class KernelGrid {
    constructor(x, y, width, height, gridSize, visualizationData) {
        this.rect = { x, y, width, height };
        this.gridSize = gridSize;
        this.visualizationData = visualizationData;
        this.currentZ = 0;
        this.blocks = [];
        this.calculateBlockSize();
        this.createBlocks();
        this.selectedBlock = null;
        this.filterValues = [-1, -1, -1]; // Default filter values for x, y, z
    }

    calculateBlockSize() {
        this.blockWidth = Math.floor(this.rect.width / this.gridSize[0]) - 1;
        this.blockHeight = Math.floor(this.rect.height / this.gridSize[1]) - 1;
    }

    createBlocks() {
        this.blocks = [];
        for (let y = 0; y < this.gridSize[1]; y++) {
            for (let x = 0; x < this.gridSize[0]; x++) {
                const blockX = this.rect.x + x * (this.blockWidth + 1);
                const blockY = this.rect.y + y * (this.blockHeight + 1);
                const gridKey = `(${x}, ${y}, ${this.currentZ})`;
                const blockData = this.visualizationData[gridKey] || [];
                const block = new GridBlock(
                    blockX, blockY, this.blockWidth, this.blockHeight,
                    x, y, this.currentZ, blockData,
                    switchToMainView,
                    containerElement,
                    canvas,
                    draw
                );
                this.blocks.push(block);
            }
        }
    }

    draw(ctx) {
        ctx.fillStyle = '#F0F0F0';
        ctx.fillRect(this.rect.x, this.rect.y, this.rect.width, this.rect.height);
        this.blocks.forEach(block => {
            if (this.shouldDrawBlock(block)) {
                block.draw(ctx);
            }
        });
    }

    shouldDrawBlock(block) {
        return (this.filterValues[0] === -1 || block.gridPosition.x === this.filterValues[0]) &&
               (this.filterValues[1] === -1 || block.gridPosition.y === this.filterValues[1]) &&
               (this.filterValues[2] === -1 || block.gridPosition.z === this.filterValues[2]);
    }

    updateZ(z) {
        this.currentZ = z;
        this.filterValues[2] = z;
        this.blocks.forEach(block => {
            block.gridPosition.z = z;
            const gridKey = `(${block.gridPosition.x}, ${block.gridPosition.y}, ${z})`;
            block.blockData = this.visualizationData[gridKey] || [];
        });
    }

    handleClick(x, y) {
        const clickedBlock = this.blocks.find(block =>
            block.isPointInside(x, y) && this.shouldDrawBlock(block)
        );
        if (clickedBlock) {
            console.log(`Clicked block at (${clickedBlock.gridPosition.x}, ${clickedBlock.gridPosition.y}, ${clickedBlock.gridPosition.z})`);
            if (this.selectedBlock) {
                this.selectedBlock.hideDetailedView();
            }
            this.selectedBlock = clickedBlock;
            clickedBlock.showDetailedView();
            return clickedBlock;
        }
        return null;
    }

    handleMouseMove(x, y) {
        this.blocks.forEach(block => {
            if (this.shouldDrawBlock(block)) {
                block.handleMouseMove(x, y);
            } else {
                block.isHovered = false;
            }
        });
    }

    updateFilter(dimension, value) {
        this.filterValues[dimension] = value;
    }
}

function determineMaxValues(visualizationData) {
    maxX = 0;
    maxY = 0;
    maxZ = 0;
    const keys = Object.keys(visualizationData);
    keys.forEach(key => {
        const [x, y, z] = key.replace(/[()]/g, '').split(', ').map(Number);
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    });
}

function initializeUIElements() {
    sliders = [
        new Slider(1300, 50, 250, 20, "Program Id 0", -1, maxX),
        new Slider(1300, 120, 250, 20, "Program Id 1", -1, maxY),
        new Slider(1300, 190, 250, 20, "Program Id 2", -1, maxZ)
    ];

    zSlider = new Slider(50, 860, 1200, 20, "Z-axis", 0, maxZ);
    zSlider.enabled = maxZ > 0;

    precomputeButton = new Button(1300, 260, 250, 40, "Precompute");
    kernelGrid = new KernelGrid(50, 50, 1200, 800, [maxX + 1, maxY + 1, maxZ + 1], globalData.ops.visualization_data);
    backButton = new Button(50, 50, 100, 40, "Back");
    const buttonSize = 40;
    const margin = 10;
    infoButton = new Button(
        canvas.width - buttonSize - margin,
        margin,
        buttonSize,
        buttonSize,
        "i",
        true
    );

    isInitialized = true;

    infoPopup = createInfoPopup();
}

function switchToTensorView(clickedBlock) {
    currentView = 'tensor';
    currentBlockData = clickedBlock;
    console.log("Switched to tensor view. Block data:", clickedBlock);

    containerElement.style.pointerEvents = 'auto';
    containerElement.style.display = 'block';
    clickedBlock.showDetailedView();

    canvas.style.display = 'none';
}

function handleMouseEvent(event) {
    if (!isInitialized) {
        console.warn('UI elements not initialized yet');
        return;
    }
    if (infoButton) {
        infoButton.handleEvent(event);
        if (event.type === 'mousedown' && infoButton.isHovered) {
            showInfoPopup(infoPopup);
        }
    }
    const { offsetX, offsetY } = event;
    if (currentView === 'main') {
        sliders.forEach((slider, index) => {
            slider.handleEvent(event);
            if (kernelGrid) {
                kernelGrid.updateFilter(index, slider.value);
            }
        });
        if (zSlider && zSlider.enabled) {
            zSlider.handleEvent(event);
            if (kernelGrid) {
                kernelGrid.updateZ(zSlider.value);
            }
        }
        if (precomputeButton) {
            precomputeButton.handleEvent(event);
        }
        if (kernelGrid) {
            kernelGrid.handleMouseMove(offsetX, offsetY);
            if (event.type === 'mousedown') {
                const clickedBlock = kernelGrid.handleClick(offsetX, offsetY);
                if (clickedBlock) {
                    switchToTensorView(clickedBlock);
                }
            }
        }
    } else if (currentView === 'tensor') {
        if (backButton) {
            backButton.handleEvent(event);
            if (event.type === 'mousedown' && backButton.isHovered) {
                switchToMainView();
            }
        }
    }
    draw();
}

function draw() {
    if (!ctx) {
        console.error('Canvas context not available');
        return;
    }

    ctx.fillStyle = '#1e1e28';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (currentView === 'main' || currentView === 'main') {
        if (kernelGrid) kernelGrid.draw(ctx);
        sliders.forEach(slider => slider.draw(ctx));
        if (zSlider && zSlider.enabled) {
            zSlider.draw(ctx);
        }
        if (precomputeButton) precomputeButton.draw(ctx);
        if (infoButton) {
            infoButton.draw(ctx);
        }
    }
}

async function fetchData() {
    try {
        const response = await fetch('/api/data');
        globalData = await response.json();
        console.log(globalData);

        determineMaxValues(globalData.ops.visualization_data);
        initializeUIElements();
        draw();
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Initialize app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeApp();
        fetchData();
    });
} else {
    // DOM is already loaded
    initializeApp();
    fetchData();
}
