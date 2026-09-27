export function createDisposer() {
    const disposers = new Set();
    const add = (disposer) => {
        if (typeof disposer !== 'function')
            return null;
        disposers.add(disposer);
        return disposer;
    };
    const listen = (target, event, handler, options) => {
        if (!target || !target.addEventListener)
            return null;
        target.addEventListener(event, handler, options);
        return add(() => target.removeEventListener(event, handler, options));
    };
    const interval = (handler, delay) => {
        const id = setInterval(handler, delay);
        return add(() => clearInterval(id));
    };
    const timeout = (handler, delay) => {
        const id = setTimeout(handler, delay);
        return add(() => clearTimeout(id));
    };
    const dispose = () => {
        disposers.forEach((fn) => {
            try {
                fn();
            }
            catch (err) { }
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
