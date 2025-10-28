export function createFlowDiagram(containerElement, opsByProgram) {
// Minimal NKI flow view: three lanes (HBM, SBUF, PSUM) and arrows per op time_idx
// opsByProgram: array of op objects for a grid block (Load/Store/Dot/Copy) with mem_* fields
  const laneNames = ["HBM", "SBUF", "PSUM"];
  const laneY = { HBM: 60, SBUF: 160, PSUM: 260 };
  const width = containerElement.clientWidth || 1200;
  const height = 360;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = '100%';
  canvas.style.height = '360px';
  canvas.style.background = '#111';
  canvas.style.border = '1px solid #333';
  containerElement.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  // Draw lanes
  ctx.font = '14px Arial';
  ctx.fillStyle = '#ddd';
  ctx.strokeStyle = '#444';
  laneNames.forEach(name => {
    const y = laneY[name];
    ctx.beginPath();
    ctx.moveTo(80, y);
    ctx.lineTo(width - 20, y);
    ctx.stroke();
    ctx.fillText(name, 20, y + 5);
  });

  // Normalize time to X range
  const events = opsByProgram
    .filter(op => op.time_idx !== undefined && op.time_idx >= 0)
    .map(op => ({
      t: op.time_idx,
      src: (op.mem_src || '').toUpperCase(),
      dst: (op.mem_dst || '').toUpperCase(),
      bytes: Number(op.bytes || 0),
      type: op.type,
      uuid: op.uuid
    }));
  if (events.length === 0) {
    ctx.fillStyle = '#aaa';
    ctx.fillText('No NKI metadata found. Run NKI example or add mem_src/mem_dst/bytes/time_idx to records.', 80, height/2);
    return () => { containerElement.innerHTML = ''; };
  }
  events.sort((a,b)=>a.t-b.t);
  const tMin = events[0].t;
  const tMax = events[events.length-1].t || (tMin+1);
  const toX = t => 80 + (width-120) * (t - tMin) / Math.max(1, (tMax - tMin));

  // Color mapping per type
  const colorFor = (type) => {
    switch ((type||'').toLowerCase()){
      case 'load': return '#00bcd4';
      case 'store': return '#ff9800';
      case 'dot': return '#8bc34a';
      default: return '#9e9e9e';
    }
  };

  // Draw arrows for each event
  events.forEach(e => {
    const y1 = laneY[e.src] ?? laneY.SBUF;
    const y2 = laneY[e.dst] ?? laneY.SBUF;
    const x = toX(e.t);
    const col = colorFor(e.type);
    // Arrow thickness by bytes (log scaled)
    const w = Math.max(1, Math.log10((e.bytes||1)));
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();
    // Arrow head
    ctx.beginPath();
    const dir = Math.sign(y2 - y1) || 1;
    const hx = x; const hy = y2;
    ctx.moveTo(hx, hy);
    ctx.lineTo(hx - 6, hy - 6*dir);
    ctx.lineTo(hx + 6, hy - 6*dir);
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();
    // Label
    ctx.fillStyle = '#ccc';
    ctx.font = '11px Arial';
    const label = `${e.type}  ${e.bytes||0}B`;
    ctx.fillText(label, x + 6, (y1+y2)/2 - 6);
  });

  // Legend
  const legend = document.createElement('div');
  legend.style.position = 'relative';
  legend.style.marginTop = '6px';
  legend.style.color = '#ddd';
  legend.style.font = '12px Arial';
  legend.innerHTML = `<b>Flow Diagram</b> &nbsp; <span style="color:#00bcd4">■ Load</span>&nbsp;&nbsp;<span style="color:#8bc34a">■ Dot(PSUM)</span>&nbsp;&nbsp;<span style="color:#ff9800">■ Store</span>`;
  containerElement.appendChild(legend);

  return () => { containerElement.innerHTML = ''; };
}

// Backward compatibility alias
export const createNKIFlow = createFlowDiagram;
