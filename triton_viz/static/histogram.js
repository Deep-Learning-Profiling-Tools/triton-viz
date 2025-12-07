export function createHistogramOverlay(containerElement, options) {
    const {
        title = "Value Distribution",
        sources = [],
        apiBase = "",
        buildRequestBody,
        defaultBins = 64,
    } = options;

    if (!buildRequestBody) {
        throw new Error("buildRequestBody is required for histogram overlay");
    }

    const button = document.createElement("button");
    button.textContent = "Value Histogram";

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed",
        top: "50px",
        right: "50px",
        width: "480px",
        padding: "12px",
        background: "rgba(0, 0, 0, 0.85)",
        color: "#fff",
        borderRadius: "8px",
        border: "1px solid #555",
        zIndex: 3000,
        display: "none",
    });

    const header = document.createElement("div");
    header.textContent = title;
    header.style.fontSize = "16px";
    header.style.marginBottom = "8px";
    header.style.fontWeight = "bold";
    overlay.appendChild(header);

    const controls = document.createElement("div");
    controls.style.display = "flex";
    controls.style.gap = "8px";
    controls.style.flexWrap = "wrap";

    const select = document.createElement("select");
    sources.forEach((src) => {
        const opt = document.createElement("option");
        opt.value = src.value;
        opt.textContent = src.label;
        select.appendChild(opt);
    });
    controls.appendChild(select);

    const binInput = document.createElement("input");
    binInput.type = "number";
    binInput.value = defaultBins;
    binInput.min = 4;
    binInput.max = 512;
    binInput.step = 2;
    binInput.style.width = "80px";
    binInput.title = "Number of bins";
    controls.appendChild(binInput);

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "Refresh";
    controls.appendChild(refreshBtn);

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.addEventListener("click", () => {
        overlay.style.display = "none";
    });
    controls.appendChild(closeBtn);

    overlay.appendChild(controls);

    const info = document.createElement("div");
    info.style.margin = "6px 0";
    info.style.fontSize = "12px";
    overlay.appendChild(info);

    const canvas = document.createElement("canvas");
    canvas.width = 440;
    canvas.height = 240;
    canvas.style.background = "#111";
    canvas.style.border = "1px solid #444";
    overlay.appendChild(canvas);

    const status = document.createElement("div");
    status.style.fontSize = "12px";
    status.style.marginTop = "4px";
    overlay.appendChild(status);

    button.addEventListener("click", () => {
        overlay.style.display = "block";
        updateHistogram();
    });

    refreshBtn.addEventListener("click", () => {
        updateHistogram();
    });

    select.addEventListener("change", () => {
        updateHistogram();
    });

    async function updateHistogram() {
        status.textContent = "Loading histogram...";
        info.textContent = "";
        const bins = parseInt(binInput.value, 10) || defaultBins;

        try {
            const body = buildRequestBody(select.value, bins);
            body.bins = bins;
            body.max_samples = body.max_samples || 200000;
            const res = await fetch(`${apiBase}/api/histogram`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                throw new Error(data.error || "Failed to fetch histogram");
            }
            drawHistogram(canvas, data.counts, data.edges);
            info.textContent = `Min: ${data.min.toFixed(6)} | Max: ${data.max.toFixed(6)} | Total values: ${data.n} | Sampled: ${data.sampled}`;
            status.textContent = "";
        } catch (err) {
            status.textContent = `Histogram error: ${err.message}`;
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    function drawHistogram(canvasEl, counts, edges) {
        const ctx = canvasEl.getContext("2d");
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        if (!counts || !counts.length) {
            ctx.fillStyle = "#888";
            ctx.fillText("No data", 20, 30);
            return;
        }

        const width = canvasEl.width - 40;
        const height = canvasEl.height - 40;
        const originX = 30;
        const originY = canvasEl.height - 20;

        const maxCount = Math.max(...counts);
        const barWidth = width / counts.length;

        ctx.strokeStyle = "#555";
        ctx.beginPath();
        ctx.moveTo(originX, originY);
        ctx.lineTo(originX + width, originY);
        ctx.stroke();

        counts.forEach((count, idx) => {
            const barHeight = maxCount ? (count / maxCount) * height : 0;
            const x = originX + idx * barWidth;
            const y = originY - barHeight;
            ctx.fillStyle = "#ffa500";
            ctx.fillRect(x, y, Math.max(1, barWidth - 2), barHeight);
        });

        ctx.fillStyle = "#ccc";
        ctx.font = "10px monospace";
        ctx.fillText(`${edges[0].toFixed(4)}`, originX, originY + 12);
        const lastEdge = edges[edges.length - 1];
        ctx.fillText(`${lastEdge.toFixed(4)}`, originX + width - 40, originY + 12);
    }

    (containerElement || document.body).appendChild(overlay);

    return {
        button,
        overlay,
        destroy() {
            overlay.remove();
        },
    };
}
