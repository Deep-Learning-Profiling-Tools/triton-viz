type ProgramCoords = { x: number; y: number; z: number };
type Toggles = { colorize: boolean; histogram: boolean; allPrograms: boolean; showCode: boolean };
export type State = { activeProgram: ProgramCoords; activeOp: unknown | null; toggles: Toggles };
type StatePatch = Omit<Partial<State>, 'activeProgram' | 'toggles'> & {
    activeProgram?: Partial<ProgramCoords>;
    toggles?: Partial<Toggles>;
};
type Listener = (state: State) => void;

const DEFAULT_STATE: State = {
    activeProgram: { x: 0, y: 0, z: 0 },
    activeOp: null,
    toggles: {
        colorize: false,
        histogram: false,
        allPrograms: false,
        showCode: false,
    },
};

let currentState: State = {
    activeProgram: { ...DEFAULT_STATE.activeProgram },
    activeOp: null,
    toggles: { ...DEFAULT_STATE.toggles },
};

const listeners = new Set<Listener>();

function notify(nextState: State) {
    listeners.forEach((listener) => {
        try { listener(nextState); } catch (err) {}
    });
}

function mergeState(patch: StatePatch = {}): State {
    const next: State = {
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

export function subscribe(listener: Listener) {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    listener(currentState);
    return () => listeners.delete(listener);
}

export function setActiveProgram(activeProgram: ProgramCoords) {
    return mergeState({ activeProgram: { ...activeProgram } });
}

export function setActiveOp(activeOp: unknown | null) {
    return mergeState({ activeOp: activeOp || null });
}

export function setToggles(toggles: Partial<Toggles>) {
    return mergeState({ toggles: { ...toggles } });
}

export function resetToggles() {
    return mergeState({ toggles: { ...DEFAULT_STATE.toggles } });
}
