interface Window {
    __TILELENS_API__?: string;
    __tileLensOpState?: {
        colorize?: boolean;
        histogram?: boolean;
        allPrograms?: boolean;
        showCode?: boolean;
        editTensorView?: boolean;
    };
    __tileLensCodeToggle?: (force?: boolean) => boolean | Promise<boolean>;
    __tileLensCodeHide?: () => boolean;
    __tileLensCodeVisible?: () => boolean;
    __tileLensActiveBlock?: { blockData?: Array<{ uuid?: string | null }> };
    __tileLensPreserveCodePanel?: boolean;
    setOpControlHandlers?: (handlers: {
        toggleColorize?: () => boolean | Promise<boolean>;
        toggleShowCode?: () => boolean | Promise<boolean> | void;
        toggleHistogram?: () => boolean | Promise<boolean>;
        toggleAllPrograms?: (() => boolean | Promise<boolean>) | null;
        toggleEditTensorView?: () => boolean | Promise<boolean>;
        download?: {
            trigger: (sources: string[]) => void | Promise<void>;
            options: Array<{ value: string; label: string; color?: string }>;
            buttonLabel?: string;
        } | null;
    } | null) => void;
    setOpControlState?: (state: {
        colorize?: boolean;
        histogram?: boolean;
        allPrograms?: boolean;
        showCode?: boolean;
        editTensorView?: boolean;
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
