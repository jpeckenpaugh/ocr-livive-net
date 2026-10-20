/* tslint:disable */
/* eslint-disable */

export class ModelHandle {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    method(): string;
    /**
     * Static model data.
     */
    modelData(): any;
    predict(image: Uint8Array): any;
    /**
     * Export model to binary ONNX format.
     */
    toOnnx(): Uint8Array;
}

export function preprocess(rgba: Uint8Array, width: number, height: number): Uint8Array;

export function trainCentroid(images: Uint8Array, labels: Uint8Array, _seed: number): ModelHandle;

export function trainKnn(images: Uint8Array, labels: Uint8Array, _seed: number): ModelHandle;

export function trainMlp(images: Uint8Array, labels: Uint8Array, seed: number): ModelHandle;

export function trainPerceptron(images: Uint8Array, labels: Uint8Array, seed: number): ModelHandle;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_modelhandle_free: (a: number, b: number) => void;
    readonly modelhandle_method: (a: number) => [number, number];
    readonly modelhandle_modelData: (a: number) => any;
    readonly modelhandle_predict: (a: number, b: number, c: number) => any;
    readonly modelhandle_toOnnx: (a: number) => [number, number, number, number];
    readonly preprocess: (a: number, b: number, c: number, d: number) => [number, number];
    readonly trainCentroid: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly trainKnn: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly trainMlp: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly trainPerceptron: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
