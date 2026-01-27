export type RequestOptions = {
    method?: string;
    body?: BodyInit | null;
    headers?: Record<string, string>;
    base?: string;
    signal?: AbortSignal | null;
};

const getDefaultBase = (): string => {
    if (typeof globalThis === 'undefined') return '';
    const globalBase = (globalThis as typeof globalThis & { __TRITON_VIZ_API__?: string }).__TRITON_VIZ_API__;
    return globalBase || '';
};

const buildUrl = (path?: string | null, baseOverride?: string | null): string => {
    if (!path) return baseOverride || getDefaultBase();
    if (/^https?:\/\//i.test(path)) return path;
    const base = baseOverride !== undefined ? baseOverride : getDefaultBase();
    if (!base) return path.startsWith('/') ? path : `/${path}`;
    return `${base}${path.startsWith('/') ? '' : '/'}${path}`;
};

type ApiError = { error?: string };

const parseJson = (text: string): unknown | null => {
    if (!text) return null;
    try {
        return JSON.parse(text);
    } catch (err) {
        return null;
    }
};

const isApiError = (value: unknown): value is ApiError => {
    return !!value && typeof value === 'object' && 'error' in value;
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
    const requestInit: RequestInit = { method, headers: nextHeaders, body: nextBody };
    if (signal !== undefined) requestInit.signal = signal;
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
    return data as T;
}

export function getApiBase(): string {
    return getDefaultBase();
}

export function getJson<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
    return requestJson<T>(path, { ...options, method: 'GET' });
}

export function postJson<T = unknown>(
    path: string,
    body: BodyInit | Record<string, unknown> | null,
    options: RequestOptions = {},
): Promise<T> {
    return requestJson<T>(path, { ...options, method: 'POST', body: body as BodyInit | null });
}
