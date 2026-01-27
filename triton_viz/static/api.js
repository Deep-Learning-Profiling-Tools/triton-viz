const getDefaultBase = () => {
    if (typeof globalThis === 'undefined')
        return '';
    const globalBase = globalThis.__TRITON_VIZ_API__;
    return globalBase || '';
};
const buildUrl = (path, baseOverride) => {
    if (!path)
        return baseOverride || getDefaultBase();
    if (/^https?:\/\//i.test(path))
        return path;
    const base = baseOverride !== undefined ? baseOverride : getDefaultBase();
    if (!base)
        return path.startsWith('/') ? path : `/${path}`;
    return `${base}${path.startsWith('/') ? '' : '/'}${path}`;
};
const parseJson = (text) => {
    if (!text)
        return null;
    try {
        return JSON.parse(text);
    }
    catch (err) {
        return null;
    }
};
const isApiError = (value) => {
    return !!value && typeof value === 'object' && 'error' in value;
};
export async function requestJson(path, options = {}) {
    const { method = 'GET', body = null, headers = {}, base, signal, } = options;
    const url = buildUrl(path, base);
    const nextHeaders = { ...headers };
    let nextBody = body;
    if (body !== null && body !== undefined && typeof body !== 'string' && !(body instanceof FormData)) {
        nextBody = JSON.stringify(body);
        if (!nextHeaders['Content-Type']) {
            nextHeaders['Content-Type'] = 'application/json';
        }
    }
    const requestInit = { method, headers: nextHeaders, body: nextBody };
    if (signal !== undefined)
        requestInit.signal = signal;
    const response = await fetch(url, requestInit);
    const text = await response.text();
    const data = parseJson(text);
    if (!response.ok) {
        const message = isApiError(data) && data.error ? data.error : `${response.status} ${response.statusText}`.trim();
        throw new Error(message || 'request failed');
    }
    if (isApiError(data) && data.error) {
        throw new Error(data.error);
    }
    return data;
}
export function getApiBase() {
    return getDefaultBase();
}
export function getJson(path, options = {}) {
    return requestJson(path, { ...options, method: 'GET' });
}
export function postJson(path, body, options = {}) {
    return requestJson(path, { ...options, method: 'POST', body: body });
}
