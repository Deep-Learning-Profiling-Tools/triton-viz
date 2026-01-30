import type { OpRecord } from '../types/types.js';
import type { ViewState } from '../components/tensor_view.js';

/** Renderer factory signature for op visualizers. */
export type Visualizer = (container: HTMLElement, op: OpRecord, viewState?: ViewState | null) => (() => void) | void;

const visualizers = new Map<string, Visualizer>();

/**
 * Register an op visualizer by type name.
 * @param type - Op type string to register.
 * @param create - Visualizer factory.
 */
export function registerVisualizer(type: string, create: Visualizer): void {
    if (!type || typeof create !== 'function') return;
    visualizers.set(type, create);
}

/**
 * Look up a visualizer by op type.
 * @param type - Op type string to look up.
 * @returns Visualizer factory or null.
 */
export function getVisualizer(type: string): Visualizer | null {
    return visualizers.get(type) || null;
}

/**
 * Check whether a visualizer is registered.
 * @param type - Op type string to check.
 * @returns True if a visualizer exists.
 */
export function hasVisualizer(type: string): boolean {
    return visualizers.has(type);
}

/**
 * List all registered visualizer types.
 * @returns Array of op type strings.
 */
export function listVisualizers(): string[] {
    return Array.from(visualizers.keys());
}
