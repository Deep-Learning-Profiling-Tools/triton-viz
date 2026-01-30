type ProgramCoords = { x: number; y: number; z: number };
type Toggles = { colorize: boolean; histogram: boolean; allPrograms: boolean; showCode: boolean };
/** Shared UI state for the visualization shell. */
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

function notify(nextState: State): void {
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

/**
 * Read the current UI state snapshot.
 * @returns Current state object.
 */
export function getState(): State {
    return currentState;
}

/**
 * Subscribe to state changes and receive an unsubscribe function.
 * @param listener - Callback invoked on state updates.
 * @returns Unsubscribe function.
 */
export function subscribe(listener: Listener): () => boolean | void {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    listener(currentState);
    return () => listeners.delete(listener);
}

/**
 * Update the active program coordinates.
 * @param activeProgram - New program coordinates.
 * @returns Updated state.
 */
export function setActiveProgram(activeProgram: ProgramCoords): State {
    return mergeState({ activeProgram: { ...activeProgram } });
}

/**
 * Update the active op record.
 * @param activeOp - Active op payload or null.
 * @returns Updated state.
 */
export function setActiveOp(activeOp: unknown | null): State {
    return mergeState({ activeOp: activeOp || null });
}

/**
 * Update UI toggle flags.
 * @param toggles - Partial toggle updates.
 * @returns Updated state.
 */
export function setToggles(toggles: Partial<Toggles>): State {
    return mergeState({ toggles: { ...toggles } });
}

/**
 * Reset toggles back to defaults.
 * @returns Updated state.
 */
export function resetToggles(): State {
    return mergeState({ toggles: { ...DEFAULT_STATE.toggles } });
}
