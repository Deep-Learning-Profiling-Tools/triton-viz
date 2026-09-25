import { renderSbufPanel } from '../components/sbuf_panel.js';
export function createFlowDiagram(containerElement, opsByProgram) {
    // Minimal NKI flow view: three lanes (HBM, SBUF, PSUM) and arrows per op time_idx
    // opsByProgram: array of op objects for a grid block (Load/Store/Dot/Copy) with mem_* fields
    const laneNames = ["HBM", "SBUF", "PSUM"];
    const laneY = { HBM: 60, SBUF: 160, PSUM: 260 };
    const containerWidth = containerElement.clientWidth || 1200;
    const height = 360;
    const wrapper = document.createElement('div');
    wrapper.style.width = '100%';
    wrapper.style.overflowX = 'auto';
    wrapper.style.border = '1px solid #333';
    wrapper.style.background = '#111';
    containerElement.appendChild(wrapper);
    const canvas = document.createElement('canvas');
    canvas.height = height;
    canvas.style.height = `${height}px`;
    wrapper.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return () => { containerElement.innerHTML = ''; };
    // Normalize time to X range
    const events = opsByProgram
        .filter((op) => op.time_idx !== undefined && op.time_idx >= 0)
        .map((op) => ({
        t: op.time_idx || 0,
        src: (op.mem_src || '').toUpperCase(),
        dst: (op.mem_dst || '').toUpperCase(),
        bytes: Number(op.bytes || 0),
        type: op.type,
        uuid: op.uuid ?? null
    }));
    if (events.length === 0) {
        ctx.fillStyle = '#aaa';
        ctx.fillText('No NKI metadata found. Run NKI example or add mem_src/mem_dst/bytes/time_idx to records.', 80, height / 2);
        return () => { containerElement.innerHTML = ''; };
    }
    events.sort((a, b) => a.t - b.t);
    const firstEvent = events[0];
    const lastEvent = events[events.length - 1];
    if (!firstEvent || !lastEvent) {
        return () => { containerElement.innerHTML = ''; };
    }
    const tMin = firstEvent.t;
    const tMax = lastEvent.t || (tMin + 1);
    // classify copy operations (PSUM -> SBUF)
    events.forEach(e => {
        if ((e.type || '').toLowerCase() === 'store' && e.src === 'PSUM' && e.dst === 'SBUF') {
            e.type = 'Copy';
        }
    });
    const baseSpacing = 100;
    const dynamicWidth = Math.max(containerWidth, 120 + baseSpacing * events.length);
    canvas.width = dynamicWidth;
    canvas.style.width = `${dynamicWidth}px`;
    // Draw lanes
    ctx.font = '14px Arial';
    ctx.fillStyle = '#ddd';
    ctx.strokeStyle = '#444';
    laneNames.forEach(name => {
        const y = laneY[name] ?? 0;
        ctx.beginPath();
        ctx.moveTo(80, y);
        ctx.lineTo(dynamicWidth - 20, y);
        ctx.stroke();
        ctx.fillText(name, 20, y + 5);
    });
    const toX = (t) => 80 + (dynamicWidth - 120) * (t - tMin) / Math.max(1, (tMax - tMin));
    // Color mapping per type
    const colorFor = (type) => {
        switch ((type || '').toLowerCase()) {
            case 'load': return '#00bcd4';
            case 'store': return '#ff9800';
            case 'dot': return '#8bc34a';
            case 'copy': return '#b0bec5';
            default: return '#9e9e9e';
        }
    };
    // Draw arrows for each event
    events.forEach((e, idx) => {
        const y1 = laneY[e.src] ?? laneY.SBUF ?? 0;
        const y2 = laneY[e.dst] ?? laneY.SBUF ?? 0;
        const x = toX(e.t);
        const col = colorFor(e.type);
        // Arrow thickness by bytes (log scaled)
        const w = Math.max(1, Math.log10((e.bytes || 1)));
        ctx.strokeStyle = col;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
        ctx.stroke();
        // Arrow head
        ctx.beginPath();
        const dir = Math.sign(y2 - y1) || 1;
        const hx = x;
        const hy = y2;
        ctx.moveTo(hx, hy);
        ctx.lineTo(hx - 6, hy - 6 * dir);
        ctx.lineTo(hx + 6, hy - 6 * dir);
        ctx.closePath();
        ctx.fillStyle = col;
        ctx.fill();
        // Label
        ctx.fillStyle = '#ccc';
        ctx.font = '11px Arial';
        const label = `${e.type}  ${e.bytes || 0}B`;
        const labelOffset = idx % 2 === 0 ? -6 : -18;
        ctx.fillText(label, x + 6, (y1 + y2) / 2 + labelOffset);
    });
    // Legend
    const legend = document.createElement('div');
    legend.style.position = 'relative';
    legend.style.marginTop = '6px';
    legend.style.color = '#ddd';
    legend.style.font = '12px Arial';
    legend.innerHTML = `<b>Flow Diagram</b> &nbsp; <span style="color:#00bcd4">■ Load</span>&nbsp;&nbsp;<span style="color:#8bc34a">■ Dot(PSUM)</span>&nbsp;&nbsp;<span style="color:#b0bec5">■ Copy (PSUM→SBUF)</span>&nbsp;&nbsp;<span style="color:#ff9800">■ Store</span>`;
    containerElement.appendChild(legend);
    const sbufButton = renderSbufPanel();
    if (sbufButton) {
        sbufButton.style.position = 'absolute';
        sbufButton.style.top = '10px';
        sbufButton.style.right = '10px';
        containerElement.appendChild(sbufButton);
    }
    return () => { containerElement.innerHTML = ''; };
}
// Backward compatibility alias
export const createNKIFlow = createFlowDiagram;
