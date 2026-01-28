import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
    getState,
    resetToggles,
    setActiveOp,
    setActiveProgram,
    setToggles,
    subscribe,
} from '../../triton_viz/static/core/state.js';

function resetState() {
    setActiveProgram({ x: 0, y: 0, z: 0 });
    setActiveOp(null);
    resetToggles();
}

test('state initializes with defaults', () => {
    resetState();
    const state = getState();
    assert.deepEqual(state.activeProgram, { x: 0, y: 0, z: 0 });
    assert.equal(state.activeOp, null);
    assert.equal(state.toggles.colorize, false);
    assert.equal(state.toggles.histogram, false);
    assert.equal(state.toggles.allPrograms, false);
});

test('setActiveProgram updates state', () => {
    resetState();
    setActiveProgram({ x: 2, y: 3, z: 4 });
    assert.deepEqual(getState().activeProgram, { x: 2, y: 3, z: 4 });
});

test('setToggles merges partial updates', () => {
    resetState();
    setToggles({ colorize: true });
    const { toggles } = getState();
    assert.equal(toggles.colorize, true);
    assert.equal(toggles.histogram, false);
    assert.equal(toggles.allPrograms, false);
});

test('subscribe receives updates', () => {
    resetState();
    let latest = null;
    const unsubscribe = subscribe((state) => {
        latest = state;
    });
    assert.ok(latest);
    setActiveOp({ type: 'Dot', uuid: 'abc' });
    assert.equal(latest.activeOp.type, 'Dot');
    unsubscribe();
});
