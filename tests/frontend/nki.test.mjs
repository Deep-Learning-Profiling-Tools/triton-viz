import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createFlowDiagram } from '../../triton_viz/static/ops/nki.js';
import { getHue } from '../../triton_viz/static/utils/colormap.js';

function createCanvasContext(logs) {
    return {
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 1,
        font: '',
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
        fill() {},
        closePath() {},
        clearRect() {},
        strokeRect() {},
        fillText(text) {
            logs.push(String(text));
        },
    };
}

function createElement(tag, ctx) {
    return {
        tagName: tag.toUpperCase(),
        style: {},
        children: [],
        innerHTML: '',
        textContent: '',
        clientWidth: 0,
        width: 0,
        height: 0,
        appendChild(child) {
            this.children.push(child);
            return child;
        },
        addEventListener() {},
        removeEventListener() {},
        getContext(kind) {
            return tag === 'canvas' && kind === '2d' ? ctx : null;
        },
    };
}

function installDom() {
    const labels = [];
    const ctx = createCanvasContext(labels);
    const body = createElement('body', ctx);
    const document = {
        body,
        createElement(tag) {
            return createElement(tag, ctx);
        },
    };
    const window = {
        location: { href: 'http://localhost/visualizer/' },
        addEventListener() {},
        removeEventListener() {},
    };
    const previous = { document: globalThis.document, window: globalThis.window };
    globalThis.document = document;
    globalThis.window = window;
    return {
        labels,
        restore() {
            globalThis.document = previous.document;
            globalThis.window = previous.window;
        },
    };
}

test('transfer hue matches load hue', () => {
    assert.equal(getHue('Transfer'), 200);
});
