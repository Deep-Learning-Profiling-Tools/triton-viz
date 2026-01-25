import assert from 'node:assert/strict';
import { test } from 'node:test';
import { getJson, postJson, requestJson } from '../../triton_viz/static/api.js';

function mockFetch(handler) {
    const original = globalThis.fetch;
    globalThis.fetch = handler;
    return () => {
        globalThis.fetch = original;
    };
}

test('postJson sends json and parses response', async (t) => {
    const restore = mockFetch(async (url, options) => {
        assert.equal(url, 'http://localhost/api/test');
        assert.equal(options.method, 'POST');
        assert.equal(options.headers['Content-Type'], 'application/json');
        assert.equal(options.body, JSON.stringify({ ok: true }));
        return {
            ok: true,
            status: 200,
            statusText: 'OK',
            text: async () => JSON.stringify({ ok: true }),
        };
    });
    t.after(restore);
    const data = await postJson('/api/test', { ok: true }, { base: 'http://localhost' });
    assert.deepEqual(data, { ok: true });
});

test('requestJson throws on http error', async (t) => {
    const restore = mockFetch(async () => ({
        ok: false,
        status: 500,
        statusText: 'Server Error',
        text: async () => JSON.stringify({ error: 'boom' }),
    }));
    t.after(restore);
    await assert.rejects(() => requestJson('/api/fail'), /boom/);
});

test('getJson throws on error payload', async (t) => {
    const restore = mockFetch(async () => ({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => JSON.stringify({ error: 'bad' }),
    }));
    t.after(restore);
    await assert.rejects(() => getJson('/api/bad'), /bad/);
});
