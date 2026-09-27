const LOG_PREFIX = '[tilelens]';

export function logAction(action: string, details: Record<string, unknown> = {}): void {
    console.info(LOG_PREFIX, action, details);
}

export function logInfo(message: string, details: Record<string, unknown> = {}): void {
    console.info(LOG_PREFIX, message, details);
}
