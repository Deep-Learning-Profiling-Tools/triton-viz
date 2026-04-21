import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

function read(path) {
    return readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');
}

test('built frontend includes transfer support hooks', () => {
    const defaults = read('triton_viz/static/ops/defaults.js');
    const transfer = read('triton_viz/static/ops/transfer.js');
    const workspace = read('triton_viz/static/components/op_workspace.js');
    const tensorView = read('triton_viz/static/components/tensor_view.js');

    assert.match(defaults, /registerVisualizer\('Transfer', createTransferVisualization\)/);
    assert.match(transfer, /type:\s*'Transfer'/);
    assert.match(transfer, /isStoreLike/);
    assert.match(workspace, /op\.type === 'Transfer'/);
    assert.match(tensorView, /type === 'Transfer'/);
});
