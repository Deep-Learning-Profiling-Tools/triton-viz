const visualizers = new Map();

export function registerVisualizer(type, create) {
    if (!type || typeof create !== 'function') return;
    visualizers.set(type, create);
}

export function getVisualizer(type) {
    return visualizers.get(type) || null;
}

export function hasVisualizer(type) {
    return visualizers.has(type);
}

export function listVisualizers() {
    return Array.from(visualizers.keys());
}
