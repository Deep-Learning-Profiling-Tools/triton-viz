const LOG_PREFIX = '[triton-viz]';

export function logAction(action, details = {}) {
    console.info(LOG_PREFIX, action, details);
}

export function logInfo(message, details = {}) {
    console.info(LOG_PREFIX, message, details);
}
