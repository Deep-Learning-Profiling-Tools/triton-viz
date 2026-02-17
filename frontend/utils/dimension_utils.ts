import * as THREE from 'https://esm.sh/three@0.155.0';
import { Text } from 'https://esm.sh/troika-three-text@0.52.4?deps=three@0.155.0';

type ThreeScene = any;
type ThreeVector3 = any;
type ThreeGroup = any;
type ThreeLineBasicMaterial = any;
type ThreeLine = any;
type ColorInput = any;
type TroikaText = any;

type VectorTextOptions = {
    fontSize?: number;
    billboard?: boolean;
    depthTest?: boolean;
    strokeWidth?: number;
    strokeColor?: number | string;
};

type CadDimensionOptions = {
    offset?: number;
    extensionLength?: number;
    extensionOffset?: number;
    extensionDirection?: ThreeVector3;
    arrowSize?: number;
    arrowWidth?: number;
    textOffset?: number;
    flipThreshold?: number;
    lineWidth?: number;
    opacity?: number;
};

export function createVectorText(
    text: string,
    color: ColorInput,
    options: VectorTextOptions = {},
): TroikaText {
    const {
        fontSize = 0.15,
        billboard = true,
        depthTest = false,
        strokeWidth = 0,
        strokeColor = 0x000000
    } = options;

    const textMesh = new (Text as any)();
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
        textMesh.onBeforeRender = function (_renderer: any, _scene: any, camera: any) {
            this.quaternion.copy(camera.quaternion);
        };
    }

    textMesh.sync();
    return textMesh;
}

export function createCadDimension(
    scene: ThreeScene,
    start: ThreeVector3,
    end: ThreeVector3,
    label: string,
    axis: 'x' | 'y' | 'z',
    color: ColorInput,
    options: CadDimensionOptions = {},
): ThreeGroup {
    const {
        offset = 0.5,
        extensionLength = 0.08,
        extensionOffset = 0.1,
        extensionDirection,
        arrowSize = 0.15,
        arrowWidth = 0.08,
        textOffset = 0.25,
        flipThreshold = 1.0,
        lineWidth = 3,
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

    // determine extension direction (perpendicular to normalizedDir)
    let extDir = extensionDirection
        ? extensionDirection.clone().normalize()
        : null;
    if (!extDir || extDir.lengthSq() < 1e-9) {
        if (axis === 'x') extDir = new THREE.Vector3(0, 1, 0);
        else extDir = new THREE.Vector3(-1, 0, 0);
    }

    // dimension line points
    const dimLinePos = offset + extensionOffset;
    const d1 = start.clone().add(extDir.clone().multiplyScalar(dimLinePos));
    const d2 = end.clone().add(extDir.clone().multiplyScalar(dimLinePos));

    // extension points: always connect object->dimension line
    const extEndPos = dimLinePos + Math.max(0, extensionLength);
    const p1_start = start.clone().add(extDir.clone().multiplyScalar(extensionOffset));
    const p1_end = start.clone().add(extDir.clone().multiplyScalar(extEndPos));
    const p2_start = end.clone().add(extDir.clone().multiplyScalar(extensionOffset));
    const p2_end = end.clone().add(extDir.clone().multiplyScalar(extEndPos));

    // Extension lines
    group.add(createLine([p1_start, p1_end], material));
    group.add(createLine([p2_start, p2_end], material));

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

export function defaultAxisColor(axis: number): string {
    const base = ['#60a5fa', '#4ade80', '#f87171'];
    if (axis < base.length) return base[axis] ?? base[0] ?? '#60a5fa';
    const color = new THREE.Color();
    color.setHSL((axis * 0.1618) % 1, 0.65, 0.58);
    return `#${color.getHexString()}`;
}

function createLine(points: ThreeVector3[], material: ThreeLineBasicMaterial): ThreeLine {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
}

function createArrowhead(
    tip: ThreeVector3,
    direction: ThreeVector3,
    size: number,
    width: number,
    material: ThreeLineBasicMaterial,
): ThreeGroup {
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

type ShapeLegendEntry = {
    name: string;
    shape?: number[];
    color: string;
    dimColors?: string[];
    descriptor?: {
        shape: number[];
        stride?: number[];
        color: string;
        dimColors?: string[];
    };
};

export function createShapeLegend(container: HTMLElement, tensors: ShapeLegendEntry[]): HTMLElement {
    let legend = container.querySelector('.viz-shape-legend') as HTMLElement | null;
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

    let title = legend.querySelector('.viz-shape-legend-title') as HTMLElement | null;
    if (!title) {
        title = document.createElement('div');
        title.className = 'viz-shape-legend-title';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '4px';
        title.style.fontFamily = 'sans-serif';
        title.textContent = 'Tensor Shapes';
        legend.prepend(title);
    }
    let body = legend.querySelector('.viz-shape-legend-body') as HTMLElement | null;
    if (!body) {
        body = document.createElement('div');
        body.className = 'viz-shape-legend-body';
        body.style.display = 'grid';
        body.style.gap = '8px';
        if (title.nextSibling) legend.insertBefore(body, title.nextSibling);
        else legend.appendChild(body);
    }
    body.innerHTML = '';
    tensors.forEach(t => {
        const item = document.createElement('div');
        item.style.display = 'grid';
        item.style.gap = '2px';

        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';

        const swatch = document.createElement('div');
        swatch.style.width = '12px';
        swatch.style.height = '12px';
        swatch.style.backgroundColor = t.color;
        swatch.style.border = '1px solid rgba(255,255,255,0.5)';

        const label = document.createElement('span');
        label.style.color = '#fff';

        // render all shape dims
        let shapeHtml = `${t.name}: [`;
        const shape = t.shape || [];
        const dimColors = t.dimColors || [];
        shapeHtml += shape.map((dim, axis) => {
            const color = dimColors[axis] || defaultAxisColor(axis);
            return `<span style="color:${color}">${dim}</span>`;
        }).join(', ');
        shapeHtml += ']';

        label.innerHTML = shapeHtml;

        row.appendChild(swatch);
        row.appendChild(label);
        item.appendChild(row);

        if (t.descriptor && t.descriptor.shape.length > 0) {
            const descriptorShape = t.descriptor.shape;
            const descriptorColor = t.descriptor.color;
            const descriptorDimColors = t.descriptor.dimColors || [];
            const renderArray = (arr: number[]): string => arr.map((v, i) => {
                const color = descriptorDimColors[i] || descriptorColor;
                return `<span style="color:${color}">${v}</span>`;
            }).join(', ');
            const selectionRow = document.createElement('div');
            selectionRow.style.display = 'flex';
            selectionRow.style.alignItems = 'center';
            selectionRow.style.gap = '8px';

            const selectionSwatch = document.createElement('div');
            selectionSwatch.style.width = '12px';
            selectionSwatch.style.height = '12px';
            selectionSwatch.style.backgroundColor = descriptorColor;
            selectionSwatch.style.border = '1px solid rgba(255,255,255,0.5)';

            const selectionLabel = document.createElement('span');
            selectionLabel.style.color = '#fff';
            selectionLabel.innerHTML = `selection: [${renderArray(descriptorShape)}]`;

            selectionRow.appendChild(selectionSwatch);
            selectionRow.appendChild(selectionLabel);
            item.appendChild(selectionRow);
        }

        body.appendChild(item);
    });

    return legend;
}
