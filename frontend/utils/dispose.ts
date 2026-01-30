type Disposer = () => void;
type DisposerHandle = Disposer | null;
type EventTargetLike = {
    addEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
    removeEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
};

/** Container for cleanup helpers that unregister listeners and timers. */
export type DisposerBag = {
    add: (disposer: Disposer) => DisposerHandle;
    listen: (
        target: EventTargetLike | null,
        event: string,
        handler: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions,
    ) => DisposerHandle;
    interval: (handler: () => void, delay: number) => DisposerHandle;
    timeout: (handler: () => void, delay: number) => DisposerHandle;
    dispose: () => void;
};

/**
 * Create a disposer bag for managing listeners and timers.
 * @returns Disposer helpers for cleanup.
 */
export function createDisposer(): DisposerBag {
    const disposers = new Set<Disposer>();

    const add = (disposer: Disposer): DisposerHandle => {
        if (typeof disposer !== 'function') return null;
        disposers.add(disposer);
        return disposer;
    };

    const listen = (
        target: EventTargetLike | null,
        event: string,
        handler: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions,
    ): DisposerHandle => {
        if (!target || !target.addEventListener) return null;
        target.addEventListener(event, handler, options);
        return add(() => target.removeEventListener(event, handler, options));
    };

    const interval = (handler: () => void, delay: number): DisposerHandle => {
        const id = setInterval(handler, delay);
        return add(() => clearInterval(id));
    };

    const timeout = (handler: () => void, delay: number): DisposerHandle => {
        const id = setTimeout(handler, delay);
        return add(() => clearTimeout(id));
    };

    const dispose = (): void => {
        disposers.forEach((fn) => {
            try { fn(); } catch (err) {}
        });
        disposers.clear();
    };

    return {
        add,
        listen,
        interval,
        timeout,
        dispose,
    };
}
