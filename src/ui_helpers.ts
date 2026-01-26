type DragOptions = {
    handle?: HTMLElement;
    bounds?: HTMLElement | Window | null;
    initialLeft?: number;
    initialTop?: number;
};

export function enableDrag(panel: HTMLElement | null, options: DragOptions = {}) {
    if (!panel) return;
    const {
        handle = panel,
        bounds = null,
        initialLeft = 16,
        initialTop = 16,
    } = options;

    const useViewport = bounds === window;
    const boundsElement = useViewport ? null : (bounds || panel.parentElement || document.body);
    const boundsEl = boundsElement as HTMLElement | null;

    if (!useViewport && boundsEl) {
        const computed = getComputedStyle(boundsEl);
        if (computed.position === 'static') {
            boundsEl.style.position = 'relative';
        }
    }

    panel.style.position = useViewport ? 'fixed' : 'absolute';
    if (!panel.style.left) {
        panel.style.left = `${initialLeft}px`;
    }
    if (!panel.style.top) {
        panel.style.top = `${initialTop}px`;
    }

    handle.classList.add('is-draggable');

    let pointerId: number | null = null;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

    const onPointerDown = (event: PointerEvent) => {
        pointerId = event.pointerId;
        if (handle.setPointerCapture) {
            handle.setPointerCapture(pointerId);
        }
        handle.classList.add('is-dragging');
        panel.classList.add('dragging');
        startX = event.clientX;
        startY = event.clientY;
        startLeft = parseFloat(panel.style.left) || 0;
        startTop = parseFloat(panel.style.top) || 0;
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
    };

    const onPointerMove = (event: PointerEvent) => {
        if (event.pointerId !== pointerId) return;
        const deltaX = event.clientX - startX;
        const deltaY = event.clientY - startY;
        let nextLeft = startLeft + deltaX;
        let nextTop = startTop + deltaY;

        if (useViewport) {
            const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth);
            const maxTop = Math.max(0, window.innerHeight - panel.offsetHeight);
            nextLeft = clamp(nextLeft, 0, maxLeft);
            nextTop = clamp(nextTop, 0, maxTop);
        } else if (boundsEl) {
            const maxLeft = Math.max(0, boundsEl.clientWidth - panel.offsetWidth);
            const maxTop = Math.max(0, boundsEl.clientHeight - panel.offsetHeight);
            nextLeft = clamp(nextLeft, 0, maxLeft);
            nextTop = clamp(nextTop, 0, maxTop);
        }

        panel.style.left = `${nextLeft}px`;
        panel.style.top = `${nextTop}px`;
    };

    const onPointerUp = (event: PointerEvent) => {
        if (event.pointerId !== pointerId) return;
        if (handle.hasPointerCapture && handle.hasPointerCapture(pointerId)) {
            handle.releasePointerCapture(pointerId);
        }
        handle.classList.remove('is-dragging');
        panel.classList.remove('dragging');
        pointerId = null;
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
    };

    handle.addEventListener('pointerdown', onPointerDown);
}
