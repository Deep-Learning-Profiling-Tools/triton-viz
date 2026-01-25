import * as THREE from 'https://esm.sh/three@0.155.0';
import { Text } from 'https://esm.sh/troika-three-text@0.52.4?deps=three@0.155.0';

export function createVectorText(text, color, options = {}) {
    const {
        fontSize = 0.15,
        billboard = true,
        depthTest = false,
        strokeWidth = 0,
        strokeColor = 0x000000
    } = options;

    const textMesh = new Text();
    textMesh.text = text;
    textMesh.fontSize = fontSize;
    textMesh.color = color;
    textMesh.anchorX = 'center';
    textMesh.anchorY = 'middle';
    textMesh.strokeWidth = strokeWidth;
    textMesh.strokeColor = strokeColor;

    // troika-three-text creates its material internally; we can override properties after first sync or via material property
    if (depthTest === false) {
        textMesh.material.depthTest = false;
        textMesh.material.depthWrite = false;
    }
    textMesh.material.transparent = true;

    if (billboard) {
        textMesh.onBeforeRender = function(renderer, scene, camera) {
            this.quaternion.copy(camera.quaternion);
        };
    }

    textMesh.sync();
    return textMesh;
}

export function createCadDimension(scene, start, end, label, axis, color, options = {}) {
    const {
        offset = 0.5,
        extensionLength = 0.8,
        extensionOffset = 0.1,
        arrowSize = 0.15,
        arrowWidth = 0.08,
        textOffset = 0.25,
        flipThreshold = 1.0,
        lineWidth = 1,
        opacity = 0.8
    } = options;

    const group = new THREE.Group();
    const material = new THREE.LineBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        linewidth: lineWidth
    });

    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    const normalizedDir = dir.clone().normalize();

    // Arrow flipping logic: if the distance is too short, flip arrows to outside
    const isFlipped = length < flipThreshold;

    console.log(`[Dimension] Creating ${axis}-axis dimension for "${label}". distance=${length.toFixed(3)}, flipped=${isFlipped}`);

    // Determine extension direction (perpendicular to normalizedDir)
    let extDir;
    if (axis === 'x') {
        extDir = new THREE.Vector3(0, 1, 0); // Extension along Y
    } else if (axis === 'y') {
        extDir = new THREE.Vector3(-1, 0, 0); // Extension along X
    } else {
        extDir = new THREE.Vector3(-1, 0, 0); // Default to X for Z axis
    }

    // Extension points
    const p1_start = start.clone().add(extDir.clone().multiplyScalar(extensionOffset));
    const p1_end = start.clone().add(extDir.clone().multiplyScalar(extensionLength + extensionOffset));
    const p2_start = end.clone().add(extDir.clone().multiplyScalar(extensionOffset));
    const p2_end = end.clone().add(extDir.clone().multiplyScalar(extensionLength + extensionOffset));

    // Extension lines
    group.add(createLine([p1_start, p1_end], material));
    group.add(createLine([p2_start, p2_end], material));

    // Dimension line position
    const dimLinePos = offset + extensionOffset;
    const d1 = start.clone().add(extDir.clone().multiplyScalar(dimLinePos));
    const d2 = end.clone().add(extDir.clone().multiplyScalar(dimLinePos));

    // Arrowheads
    if (isFlipped) {
        // Flipped case: arrowheads at outer points, pointing inward towards extension lines.
        const arrow1_outer = d1.clone().sub(normalizedDir.clone().multiplyScalar(arrowSize));
        const arrow2_outer = d2.clone().add(normalizedDir.clone().multiplyScalar(arrowSize));

        // Use the outer points as tips, pointing towards the extension lines (normalizedDir for d1, -normalizedDir for d2)
        group.add(createArrowhead(arrow1_outer, normalizedDir, arrowSize, arrowWidth, material));
        group.add(createArrowhead(arrow2_outer, normalizedDir.clone().negate(), arrowSize, arrowWidth, material));

        // Main dimension line stays between the extension lines per spec
        group.add(createLine([d1, d2], material));
    } else {
        // Points inside, pointing out (towards the extension lines)
        group.add(createArrowhead(d1, normalizedDir.clone().negate(), arrowSize, arrowWidth, material));
        group.add(createArrowhead(d2, normalizedDir, arrowSize, arrowWidth, material));
        group.add(createLine([d1, d2], material));
    }

    // Label
    const midPoint = new THREE.Vector3().addVectors(d1, d2).multiplyScalar(0.5);
    // Position text very close to the line (minimal offset)
    const labelPos = midPoint.clone().add(extDir.clone().multiplyScalar(textOffset * 0.5));
    const vectorText = createVectorText(label, color, {
        fontSize: 0.35,
        depthTest: false,
        strokeWidth: 0.04,
        strokeColor: '#000'
    });
    vectorText.position.copy(labelPos);
    vectorText.renderOrder = 100; // Ensure it's rendered on top
    group.add(vectorText);

    scene.add(group);
    return group;
}

function createLine(points, material) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
}

function createArrowhead(tip, direction, size, width, material) {
    const group = new THREE.Group();
    const side = new THREE.Vector3();
    if (Math.abs(direction.x) > 0.5) {
        side.set(0, 1, 0);
    } else {
        side.set(1, 0, 0);
    }
    const perp = new THREE.Vector3().crossVectors(direction, side).normalize().multiplyScalar(width);

    const p1 = tip.clone().sub(direction.clone().multiplyScalar(size)).add(perp);
    const p2 = tip.clone().sub(direction.clone().multiplyScalar(size)).sub(perp);

    group.add(createLine([tip, p1], material));
    group.add(createLine([tip, p2], material));
    group.add(createLine([p1, p2], material));

    return group;
}

export function createShapeLegend(container, tensors) {
    let legend = container.querySelector('.viz-shape-legend');
    if (!legend) {
        legend = document.createElement('div');
        legend.className = 'viz-shape-legend viz-floating-badge';
        Object.assign(legend.style, {
            top: 'auto',
            bottom: '80px',
            left: '24px',
            right: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            padding: '12px',
            background: 'rgba(0,0,0,0.7)',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '13px',
            fontFamily: 'monospace',
            zIndex: '2000',
            border: '1px solid rgba(255,255,255,0.1)'
        });
        container.appendChild(legend);
    }

    const AXIS_COLORS = {
        x: '#f87171',
        y: '#4ade80',
        z: '#60a5fa'
    };

    legend.innerHTML = '<div style="font-weight:bold; margin-bottom:4px; font-family:sans-serif;">Tensor Shapes</div>';
    tensors.forEach(t => {
        const item = document.createElement('div');
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '8px';

        const swatch = document.createElement('div');
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.backgroundColor = t.color;
        swatch.style.border = '1px solid rgba(255,255,255,0.5)';

        const label = document.createElement('span');
        label.style.color = '#fff';

        // Render shape with colored dimension numbers
        let shapeHtml = `${t.name}: [`;
        const shape = t.shape || [];
        const dimColors = t.dimColors || []; // Optional custom colors per dimension

        if (shape.length === 1) {
            const color = dimColors[0] || AXIS_COLORS.x;
            shapeHtml += `<span style="color:${color}">${shape[0]}</span>`;
        } else if (shape.length === 2) {
            const color0 = dimColors[0] || AXIS_COLORS.y;
            const color1 = dimColors[1] || AXIS_COLORS.x;
            shapeHtml += `<span style="color:${color0}">${shape[0]}</span>, `;
            shapeHtml += `<span style="color:${color1}">${shape[1]}</span>`;
        } else if (shape.length >= 3) {
            const color0 = dimColors[0] || AXIS_COLORS.z;
            const color1 = dimColors[1] || AXIS_COLORS.y;
            const color2 = dimColors[2] || AXIS_COLORS.x;
            shapeHtml += `<span style="color:${color0}">${shape[0]}</span>, `;
            shapeHtml += `<span style="color:${color1}">${shape[1]}</span>, `;
            shapeHtml += `<span style="color:${color2}">${shape[2]}</span>`;
        }
        shapeHtml += ']';

        label.innerHTML = shapeHtml;

        item.appendChild(swatch);
        item.appendChild(label);
        legend.appendChild(item);
    });

    return legend;
}
