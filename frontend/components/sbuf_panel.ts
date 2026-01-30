import { getApiBase, getJson } from '../core/api.js';
import type { SbufTimelinePayload } from '../types/types.js';

type DevicePreset = {
    value: string;
    label: string;
    limit: number | null;
};

const DEVICE_PRESETS: DevicePreset[] = [
    { value: 'TRN1_NC_V2', label: 'Trn1 NC-v2 (24 MiB)', limit: 24 * 1024 * 1024 },
    { value: 'TRN1_CHIP', label: 'Trn1 chip (48 MiB)', limit: 48 * 1024 * 1024 },
    { value: 'TRN1_2X', label: 'trn1.2xlarge (48 MiB)', limit: 48 * 1024 * 1024 },
    { value: 'TRN1_32X', label: 'trn1.32xlarge (768 MiB)', limit: 768 * 1024 * 1024 },
    { value: 'TRN2_NC_V3', label: 'Trn2 NC-v3 (28 MiB)', limit: 28 * 1024 * 1024 },
    { value: 'TRN2_CHIP', label: 'Trn2 chip (224 MiB)', limit: 224 * 1024 * 1024 },
    { value: 'TRN2_48X', label: 'trn2.48xlarge (≈3.5 GiB)', limit: 3584 * 1024 * 1024 },
    { value: 'CUSTOM', label: 'Custom', limit: null },
];

function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes)) return '0 B';
    const thresh = 1024;
    if (Math.abs(bytes) < thresh) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let u = -1;
    do {
        bytes /= thresh;
        ++u;
    } while (Math.abs(bytes) >= thresh && u < units.length - 1);
    return `${bytes.toFixed(1)} ${units[u]}`;
}
/**
 * Create the SBUF usage panel toggle button.
 * @returns A button that opens the SBUF overlay.
 */
export function renderSbufPanel(): HTMLButtonElement {
    const apiBase = getApiBase();
    const button = document.createElement('button');
    button.textContent = 'SBUF Usage';

    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed',
        top: '60px',
        left: '20px',
        width: '520px',
        padding: '12px',
        background: 'rgba(0,0,0,0.9)',
        borderRadius: '8px',
        border: '1px solid #444',
        color: '#fff',
        zIndex: 3200,
        display: 'none',
    });
    let dragMode = false;
    let dragOffsetX = 0;
    let dragOffsetY = 0;
    overlay.addEventListener('mousedown', (event: MouseEvent) => {
        dragMode = true;
        dragOffsetX = event.clientX - overlay.offsetLeft;
        dragOffsetY = event.clientY - overlay.offsetTop;
    });
    window.addEventListener('mouseup', () => (dragMode = false));
    window.addEventListener('mousemove', (event: MouseEvent) => {
        if (!dragMode) return;
        overlay.style.left = `${event.clientX - dragOffsetX}px`;
        overlay.style.top = `${event.clientY - dragOffsetY}px`;
    });

    const header = document.createElement('div');
    header.textContent = 'SBUF Occupancy';
    header.style.fontWeight = 'bold';
    header.style.marginBottom = '8px';
    overlay.appendChild(header);

    const controls = document.createElement('div');
    controls.style.display = 'flex';
    controls.style.gap = '8px';
    controls.style.alignItems = 'center';

    const deviceSelect = document.createElement('select');
    DEVICE_PRESETS.forEach((preset) => {
        const opt = document.createElement('option');
        opt.value = preset.value;
        opt.textContent = preset.label;
        deviceSelect.appendChild(opt);
    });
    controls.appendChild(deviceSelect);

    const customInput = document.createElement('input');
    customInput.type = 'number';
    customInput.placeholder = 'bytes';
    customInput.style.width = '120px';
    customInput.disabled = true;
    controls.appendChild(customInput);

    const refreshBtn = document.createElement('button');
    refreshBtn.textContent = 'Refresh';
    controls.appendChild(refreshBtn);

    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', () => (overlay.style.display = 'none'));
    controls.appendChild(closeBtn);

    overlay.appendChild(controls);

    const summary = document.createElement('div');
    summary.style.fontSize = '12px';
    summary.style.margin = '8px 0';
    overlay.appendChild(summary);

    const canvas = document.createElement('canvas');
    canvas.width = 480;
    canvas.height = 220;
    canvas.style.background = '#111';
    canvas.style.border = '1px solid #555';
    overlay.appendChild(canvas);

    button.addEventListener('click', () => {
        overlay.style.display = 'block';
        fetchAndRender();
    });
    refreshBtn.addEventListener('click', fetchAndRender);
    deviceSelect.addEventListener('change', () => {
        customInput.disabled = deviceSelect.value !== 'CUSTOM';
        fetchAndRender();
    });

    async function fetchAndRender(): Promise<void> {
        let url = `/api/sbuf?device=${encodeURIComponent(deviceSelect.value)}`;
        if (deviceSelect.value === 'CUSTOM') {
            const customVal = parseInt(customInput.value || '0', 10);
            if (customVal > 0) {
                url += `&limit_bytes=${customVal}`;
            }
        }
        summary.textContent = 'Loading...';
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        try {
            const data = await getJson<SbufTimelinePayload>(url, { base: apiBase });
            const timeline = data?.timeline || [];
            const limitBytes = data?.limit_bytes || 0;
            const maxUsage = data?.max_usage || 0;
            const overflowPoints = data?.overflow_points || [];
            if (ctx) drawTimeline(ctx, timeline, limitBytes);
            summary.textContent = `Limit: ${formatBytes(limitBytes)} | Max: ${formatBytes(maxUsage)} | Overflow events: ${overflowPoints.length}`;
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            summary.textContent = `Error: ${message}`;
        }
    }

    function drawTimeline(ctx: CanvasRenderingContext2D, timeline: SbufTimelinePayload['timeline'], limit: number): void {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!timeline || !timeline.length) {
            ctx.fillStyle = '#ccc';
            ctx.fillText('No SBUF metadata', 20, 30);
            return;
        }
        const margin = 30;
        const width = canvas.width - margin * 2;
        const height = canvas.height - margin * 2;
        const first = timeline[0];
        const last = timeline[timeline.length - 1];
        if (!first || !last) return;
        const minTime = first.time_idx;
        const maxTime = last.time_idx || minTime + 1;
        const maxUsage = Math.max(limit, ...timeline.map((p) => p.usage));

        ctx.strokeStyle = '#555';
        ctx.strokeRect(margin, margin, width, height);

        const scaleX = (t: number): number => margin + ((t - minTime) / (maxTime - minTime || 1)) * width;
        const scaleY = (u: number): number => margin + height - (u / (maxUsage || 1)) * height;

        ctx.strokeStyle = '#ff4444';
        ctx.beginPath();
        ctx.moveTo(margin, scaleY(limit));
        ctx.lineTo(margin + width, scaleY(limit));
        ctx.stroke();

        ctx.strokeStyle = '#00e0ff';
        ctx.beginPath();
        timeline.forEach((pt, idx) => {
            const x = scaleX(pt.time_idx);
            const y = scaleY(pt.usage);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    document.body.appendChild(overlay);
    return button;
}
