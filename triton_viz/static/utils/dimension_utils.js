import * as THREE from 'https://esm.sh/three@0.155.0';
import { Text } from 'https://esm.sh/troika-three-text@0.52.4?deps=three@0.155.0';
// @ts-ignore: remote esm loader module has no local type declarations
import { FontLoader } from 'https://esm.sh/three@0.155.0/examples/jsm/loaders/FontLoader.js';
const DIMENSION_FONT_URL = 'https://unpkg.com/three@0.155.0/examples/fonts/helvetiker_bold.typeface.json';
let cachedDimensionFont = null;
let dimensionFontPromise = null;
function ensureDimensionFont() {
    if (cachedDimensionFont)
        return Promise.resolve(cachedDimensionFont);
    if (dimensionFontPromise)
        return dimensionFontPromise;
    const loader = new FontLoader();
    dimensionFontPromise = fetch(DIMENSION_FONT_URL)
        .then((resp) => resp.json())
        .then((json) => {
        cachedDimensionFont = loader.parse(json);
        return cachedDimensionFont;
    })
        .catch(() => null);
    return dimensionFontPromise;
}
function createDimensionLabelSprite(text, color, worldHeight = 0.19333333333333333) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return null;
    const fontPx = 220;
    const padX = 42;
    const padY = 30;
    ctx.font = `700 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
    const textWidth = Math.ceil(ctx.measureText(text).width);
    canvas.width = Math.max(256, textWidth + padX * 2);
    canvas.height = Math.max(192, fontPx + padY * 2);
    ctx.font = `700 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 26;
    ctx.strokeStyle = 'rgba(0,0,0,0.92)';
    ctx.fillStyle = '#ffffff';
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeText(text, cx, cy);
    ctx.fillText(text, cx, cy);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    const aspect = canvas.width / canvas.height;
    const height = Math.max(0.4, worldHeight);
    sprite.scale.set(height * aspect, height, 1);
    sprite.frustumCulled = false;
    sprite.renderOrder = 9999;
    return sprite;
}
function createDimensionLabelShapeMesh(text, color, worldHeight = 0.19333333333333333) {
    if (!cachedDimensionFont)
        return null;
    const shapes = cachedDimensionFont.generateShapes(text, worldHeight);
    if (!shapes || shapes.length === 0)
        return null;
    const geometry = new THREE.ShapeGeometry(shapes);
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    if (bbox) {
        const cx = (bbox.min.x + bbox.max.x) / 2;
        const cy = (bbox.min.y + bbox.max.y) / 2;
        geometry.translate(-cx, -cy, 0);
    }
    const material = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false,
        transparent: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 10000;
    mesh.onBeforeRender = function (_renderer, _scene, camera) {
        this.quaternion.copy(camera.quaternion);
    };
    return mesh;
}
export function createVectorText(text, color, options = {}) {
    const { fontSize = 0.15, billboard = true, depthTest = false, strokeWidth = 0, strokeColor = 0x000000, onSync, } = options;
    const textMesh = new Text();
    textMesh.text = text;
    textMesh.fontSize = fontSize;
    textMesh.color = color;
    textMesh.anchorX = 'center';
    textMesh.anchorY = 'middle';
    textMesh.strokeWidth = strokeWidth;
    textMesh.strokeColor = strokeColor;
    const applyMaterialSettings = () => {
        const material = textMesh.material;
        if (!material)
            return;
        if (depthTest === false) {
            material.depthTest = false;
            material.depthWrite = false;
        }
        material.transparent = true;
    };
    applyMaterialSettings();
    if (billboard) {
        textMesh.onBeforeRender = function (_renderer, _scene, camera) {
            this.quaternion.copy(camera.quaternion);
        };
    }
    textMesh.sync(() => {
        applyMaterialSettings();
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('triton-viz-text-sync'));
        }
        onSync?.();
    });
    return textMesh;
}
export function createCadDimension(scene, start, end, label, axis, color, options = {}) {
    const { offset = 0.5, extensionLength = 0.08, extensionOffset = 0.1, extensionDirection, textOffset = 0.25, lineWidth = 3, opacity = 0.8, } = options;
    const group = new THREE.Group();
    const material = new THREE.LineBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        linewidth: lineWidth
    });
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    console.log(`[Dimension] Creating ${axis}-axis dimension for "${label}". distance=${length.toFixed(3)}`);
    // determine extension direction (perpendicular to normalizedDir)
    let extDir = extensionDirection
        ? extensionDirection.clone().normalize()
        : null;
    if (!extDir || extDir.lengthSq() < 1e-9) {
        if (axis === 'x')
            extDir = new THREE.Vector3(0, 1, 0);
        else
            extDir = new THREE.Vector3(-1, 0, 0);
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
    group.add(createLine([d1, d2], material));
    // Label
    const midPoint = new THREE.Vector3().addVectors(d1, d2).multiplyScalar(0.5);
    // Position text very close to the line (minimal offset)
    const labelPos = midPoint.clone().add(extDir.clone().multiplyScalar(textOffset * 0.5));
    const fallbackSprite = createDimensionLabelSprite(label, color, 0.17333333333333334);
    if (fallbackSprite) {
        fallbackSprite.position.copy(labelPos);
        group.add(fallbackSprite);
    }
    ensureDimensionFont().then((font) => {
        if (!font)
            return;
        const shapeMesh = createDimensionLabelShapeMesh(label, color, 0.17333333333333334);
        if (!shapeMesh)
            return;
        shapeMesh.position.copy(labelPos);
        group.add(shapeMesh);
        if (fallbackSprite)
            fallbackSprite.visible = false;
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('triton-viz-text-sync'));
        }
    });
    scene.add(group);
    return group;
}
export function defaultAxisColor(axis) {
    const base = ['#60a5fa', '#4ade80', '#f87171'];
    if (axis < base.length)
        return base[axis] ?? base[0] ?? '#60a5fa';
    const color = new THREE.Color();
    color.setHSL((axis * 0.1618) % 1, 0.65, 0.58);
    return `#${color.getHexString()}`;
}
function createLine(points, material) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
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
            background: '#efe6d5',
            borderRadius: '8px',
            color: '#1f2937',
            fontSize: '13px',
            fontFamily: 'monospace',
            zIndex: '2000',
            border: '1px solid rgba(31, 41, 55, 0.25)'
        });
        container.appendChild(legend);
    }
    let title = legend.querySelector('.viz-shape-legend-title');
    if (!title) {
        title = document.createElement('div');
        title.className = 'viz-shape-legend-title';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '4px';
        title.style.fontFamily = 'sans-serif';
        title.textContent = 'Tensor Shapes';
        legend.prepend(title);
    }
    let body = legend.querySelector('.viz-shape-legend-body');
    if (!body) {
        body = document.createElement('div');
        body.className = 'viz-shape-legend-body';
        body.style.display = 'grid';
        body.style.gap = '8px';
        if (title.nextSibling)
            legend.insertBefore(body, title.nextSibling);
        else
            legend.appendChild(body);
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
        swatch.style.border = '1px solid rgba(31, 41, 55, 0.35)';
        const label = document.createElement('span');
        label.style.color = '#1f2937';
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
            const renderArray = (arr) => arr.map((v, i) => {
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
            selectionSwatch.style.border = '1px solid rgba(31, 41, 55, 0.35)';
            const selectionLabel = document.createElement('span');
            selectionLabel.style.color = '#1f2937';
            selectionLabel.innerHTML = `selection: [${renderArray(descriptorShape)}]`;
            selectionRow.appendChild(selectionSwatch);
            selectionRow.appendChild(selectionLabel);
            item.appendChild(selectionRow);
        }
        body.appendChild(item);
    });
    return legend;
}
