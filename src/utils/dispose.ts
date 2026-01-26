export function createDisposer() {
    type Disposer = () => void;
    const disposers = new Set<Disposer>();
    type EventTargetLike = {
        addEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
        removeEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
    };

    const add = (disposer: Disposer) => {
        if (typeof disposer !== 'function') return null;
        disposers.add(disposer);
        return disposer;
    };

    const listen = (
        target: EventTargetLike | null,
        event: string,
        handler: EventListenerOrEventListenerObject,
        options?: boolean | AddEventListenerOptions,
    ) => {
        if (!target || !target.addEventListener) return null;
        target.addEventListener(event, handler, options);
        return add(() => target.removeEventListener(event, handler, options));
    };

    const interval = (handler: () => void, delay: number) => {
        const id = setInterval(handler, delay);
        return add(() => clearInterval(id));
    };

    const timeout = (handler: () => void, delay: number) => {
        const id = setTimeout(handler, delay);
        return add(() => clearTimeout(id));
    };

    const dispose = () => {
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
