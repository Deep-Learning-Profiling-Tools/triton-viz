import type { OpRecord } from '../types/types.js';
import type { ViewState } from '../components/tensor_view.js';

export type Visualizer = (container: HTMLElement, op: OpRecord, viewState?: ViewState | null) => (() => void) | void;

const visualizers = new Map<string, Visualizer>();

export function registerVisualizer(type: string, create: Visualizer): void {
    if (!type || typeof create !== 'function') return;
    visualizers.set(type, create);
}

export function getVisualizer(type: string): Visualizer | null {
    return visualizers.get(type) || null;
}

export function hasVisualizer(type: string): boolean {
    return visualizers.has(type);
}

export function listVisualizers(): string[] {
    return Array.from(visualizers.keys());
}
