export type RequestOptions = {
    method?: string;
    body?: BodyInit | null;
    headers?: Record<string, string>;
    base?: string;
    signal?: AbortSignal | null;
};

const getDefaultBase = () => {
    if (typeof globalThis === 'undefined') return '';
    return globalThis.__TRITON_VIZ_API__ || '';
};

const buildUrl = (path, baseOverride) => {
    if (!path) return baseOverride || getDefaultBase();
    if (/^https?:\/\//i.test(path)) return path;
    const base = baseOverride !== undefined ? baseOverride : getDefaultBase();
    if (!base) return path.startsWith('/') ? path : `/${path}`;
    return `${base}${path.startsWith('/') ? '' : '/'}${path}`;
};

const parseJson = (text) => {
    if (!text) return null;
    try {
        return JSON.parse(text);
    } catch (err) {
        return null;
    }
};

export async function requestJson<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
    const {
        method = 'GET',
        body = null,
        headers = {},
        base,
        signal,
    } = options;
    const url = buildUrl(path, base);
    const nextHeaders = { ...headers };
    let nextBody: BodyInit | null = body as BodyInit | null;
    if (body !== null && body !== undefined && typeof body !== 'string' && !(body instanceof FormData)) {
        nextBody = JSON.stringify(body);
        if (!nextHeaders['Content-Type']) {
            nextHeaders['Content-Type'] = 'application/json';
        }
    }
    const response = await fetch(url, {
        method,
        headers: nextHeaders,
        body: nextBody,
        signal,
    });
    const text = await response.text();
    const data = parseJson(text);
    if (!response.ok) {
        const message = (data && data.error) ? data.error : `${response.status} ${response.statusText}`.trim();
        throw new Error(message || 'request failed');
    }
    if (data && data.error) {
        throw new Error(data.error);
    }
    return data as T;
}

export function getApiBase() {
    return getDefaultBase();
}

export function getJson<T = unknown>(path: string, options: RequestOptions = {}) {
    return requestJson<T>(path, { ...options, method: 'GET' });
}

export function postJson<T = unknown>(
    path: string,
    body: BodyInit | Record<string, unknown> | null,
    options: RequestOptions = {},
) {
    return requestJson<T>(path, { ...options, method: 'POST', body: body as BodyInit | null });
}
