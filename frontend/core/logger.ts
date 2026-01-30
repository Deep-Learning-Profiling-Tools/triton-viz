const LOG_PREFIX = '[triton-viz]';

/**
 * Log a structured UI action for debugging and analytics.
 * @param action - Action identifier.
 * @param details - Arbitrary payload metadata.
 */
export function logAction(action: string, details: Record<string, unknown> = {}): void {
    console.info(LOG_PREFIX, action, details);
}

/**
 * Log a general information message.
 * @param message - Message string.
 * @param details - Arbitrary payload metadata.
 */
export function logInfo(message: string, details: Record<string, unknown> = {}): void {
    console.info(LOG_PREFIX, message, details);
}
