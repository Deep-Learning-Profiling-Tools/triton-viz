export type ProgramAxes = {
    x: number;
    y: number;
    z: number;
};

export interface OpRecord {
    type: string;
    uuid?: string | null;
    global_shape?: number[];
    global_coords?: number[];
    slice_shape?: number[];
    slice_coords?: number[];
    input_shape?: number[];
    other_shape?: number[];
    output_shape?: number[];
    overall_key?: string;
    time_idx?: number;
    op_index?: number;
    mem_src?: string;
    mem_dst?: string;
    bytes?: number;
    [key: string]: unknown;
}

export interface TensorHighlights {
    type?: string;
    start?: number[];
    shape?: number[];
    stride?: number[];
    data?: number[][];
}

export interface TensorPayload {
    min?: number;
    max?: number;
    values?: unknown[];
    shape?: number[];
    dims?: number;
    highlights?: TensorHighlights | null;
}

export interface ProgramCountsPayload {
    counts?: number[][];
    max_count?: number;
}

export interface ProgramSubsetsPayload {
    coords?: number[][];
    counts?: number[][];
    subsets?: Record<string, number[][]>;
    subset_count?: number;
    max_count?: number;
}

export interface OpCodeLine {
    no: number;
    text: string;
}

export interface OpCodePayload {
    filename?: string;
    lineno?: number;
    highlight?: number;
    lines?: OpCodeLine[];
    error?: string;
}

export interface SbufTimelinePoint {
    time_idx: number;
    usage: number;
}

export interface SbufTimelinePayload {
    timeline?: SbufTimelinePoint[];
    limit_bytes?: number;
    max_usage?: number;
    overflow_points?: number[];
}

export interface ApiErrorPayload {
    error?: string;
}
