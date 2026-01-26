import { registerVisualizer } from './registry.js';
import { createMatMulVisualization } from '../matmul.js';
import { createLoadVisualization } from '../load.js';
import { createStoreVisualization } from '../store.js';
registerVisualizer('Dot', createMatMulVisualization);
registerVisualizer('Load', createLoadVisualization);
registerVisualizer('Store', createStoreVisualization);
