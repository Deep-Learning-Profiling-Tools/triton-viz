interface Window {
    __TRITON_VIZ_API__?: string;
    __tritonVizOpState?: {
        colorize?: boolean;
        histogram?: boolean;
        allPrograms?: boolean;
        showCode?: boolean;
    };
    __tritonVizCodeToggle?: (force?: boolean) => boolean | Promise<boolean>;
    __tritonVizCodeHide?: () => boolean;
    __tritonVizCodeVisible?: () => boolean;
    __tritonVizActiveBlock?: { blockData?: Array<{ uuid?: string | null }> };
    __tritonVizPreserveCodePanel?: boolean;
    setOpControlHandlers?: (handlers: {
        toggleColorize?: () => boolean | Promise<boolean>;
        toggleShowCode?: () => boolean | Promise<boolean> | void;
        toggleHistogram?: () => boolean | Promise<boolean>;
        toggleAllPrograms?: (() => boolean | Promise<boolean>) | null;
    } | null) => void;
    setOpControlState?: (state: {
        colorize?: boolean;
        histogram?: boolean;
        allPrograms?: boolean;
        showCode?: boolean;
    }) => void;
    resetOpControls?: () => void;
    current_op_uuid?: string | null;
    last_op?: unknown;
    last_op_global_shape?: number[];
    last_global_coords?: number[];
    last_slice_shape?: number[];
    last_slice_coords?: number[];
}

interface HTMLElement {
    __vizGetState?: () => unknown;
}
