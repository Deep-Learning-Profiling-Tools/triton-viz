import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
    getVisualizer,
    hasVisualizer,
    listVisualizers,
    registerVisualizer,
} from '../../triton_viz/static/ops/registry.js';

test('registry stores and returns visualizers', () => {
    const type = 'TestRegistry';
    const create = () => () => {};
    registerVisualizer(type, create);
    assert.equal(getVisualizer(type), create);
    assert.equal(hasVisualizer(type), true);
    assert.ok(listVisualizers().includes(type));
});

test('registry ignores invalid entries', () => {
    const type = 'BadRegistry';
    registerVisualizer(type, null);
    assert.equal(hasVisualizer(type), false);
});
