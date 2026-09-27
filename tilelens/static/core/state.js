const DEFAULT_STATE = {
    activeProgram: { x: 0, y: 0, z: 0 },
    activeOp: null,
    toggles: {
        colorize: false,
        histogram: false,
        allPrograms: false,
        showCode: false,
        editTensorView: false,
    },
};
let currentState = {
    activeProgram: { ...DEFAULT_STATE.activeProgram },
    activeOp: null,
    toggles: { ...DEFAULT_STATE.toggles },
};
const listeners = new Set();
function notify(nextState) {
    listeners.forEach((listener) => {
        try {
            listener(nextState);
        }
        catch (err) { }
    });
}
function mergeState(patch = {}) {
    const next = {
        ...currentState,
        ...patch,
        activeProgram: {
            ...currentState.activeProgram,
            ...(patch.activeProgram || {}),
        },
        toggles: {
            ...currentState.toggles,
            ...(patch.toggles || {}),
        },
    };
    currentState = next;
    notify(next);
    return next;
}
export function getState() {
    return currentState;
}
export function subscribe(listener) {
    if (typeof listener !== 'function')
        return () => { };
    listeners.add(listener);
    listener(currentState);
    return () => listeners.delete(listener);
}
export function setActiveProgram(activeProgram) {
    return mergeState({ activeProgram: { ...activeProgram } });
}
export function setActiveOp(activeOp) {
    return mergeState({ activeOp: activeOp || null });
}
export function setToggles(toggles) {
    return mergeState({ toggles: { ...toggles } });
}
export function resetToggles() {
    return mergeState({ toggles: { ...DEFAULT_STATE.toggles } });
}
