/**
 * ScenePlyLoader splits a single combined GES scene PLY (scene.ply) — which tags
 * every vertex with a `prim_type` field (0 = gaussian, 1 = surfel) — back into two
 * standard 3DGS PLY blobs. The `prim_type` property is stripped from the output so
 * the existing SurfelLoader / GaussianLoader (and the @mkkellogg PLY parser) consume
 * them unchanged.
 *
 * Combined PLY layout (binary_little_endian):
 *   x,y,z, nx,ny,nz, f_dc_0..2, [f_rest_0..N], opacity, scale_0..2, rot_0..3, prim_type(uchar)
 */

const PLY_TYPE_SIZES: Record<string, number> = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4,
    float: 4, float32: 4,
    double: 8, float64: 8,
};

const PRIM_TYPE_GAUSSIAN = 0;
const PRIM_TYPE_SURFEL = 1;
const PRIM_TYPE_FIELD = "prim_type";

interface PlyProperty {
    name: string;
    type: string;
    size: number;
    offset: number; // byte offset within a vertex record
}

export interface SplitScene {
    /** Object URL for the surfel-only standard 3DGS PLY (empty string if none). */
    surfelUrl: string;
    /** Object URL for the gaussian-only standard 3DGS PLY (empty string if none). */
    gaussianUrl: string;
    surfelCount: number;
    gaussianCount: number;
}

export class ScenePlyLoader {
    /**
     * Fetch a combined scene.ply and split it into surfel + gaussian PLY object URLs.
     */
    public static async splitSceneFromUrl(url: string): Promise<SplitScene> {
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`Failed to fetch scene PLY (${res.status}) from ${url}`);
        }
        const buffer = await res.arrayBuffer();
        return this.splitSceneBuffer(buffer);
    }

    /**
     * Split an in-memory combined scene.ply ArrayBuffer.
     */
    public static splitSceneBuffer(buffer: ArrayBuffer): SplitScene {
        const bytes = new Uint8Array(buffer);

        // --- Locate and parse the ASCII header ---
        const marker = "end_header\n";
        const probeLen = Math.min(bytes.length, 1 << 16);
        const headerText = new TextDecoder("ascii").decode(bytes.subarray(0, probeLen));
        const markerIdx = headerText.indexOf(marker);
        if (markerIdx < 0) {
            throw new Error("Invalid PLY: 'end_header' not found.");
        }
        const dataOffset = markerIdx + marker.length;
        const headerLines = headerText.substring(0, markerIdx).split("\n");

        let format = "";
        let vertexCount = 0;
        const properties: PlyProperty[] = [];
        let inVertexElement = false;
        let offset = 0;

        for (const rawLine of headerLines) {
            const line = rawLine.trim();
            if (line.startsWith("format ")) {
                format = line.split(/\s+/)[1];
            } else if (line.startsWith("element ")) {
                const parts = line.split(/\s+/);
                inVertexElement = parts[1] === "vertex";
                if (inVertexElement) {
                    vertexCount = parseInt(parts[2], 10);
                }
            } else if (line.startsWith("property ") && inVertexElement) {
                const parts = line.split(/\s+/);
                // Only scalar properties expected (no lists) in a 3DGS PLY.
                const type = parts[1];
                const name = parts[2];
                const size = PLY_TYPE_SIZES[type];
                if (size === undefined) {
                    throw new Error(`Unsupported PLY property type: ${type}`);
                }
                properties.push({ name, type, size, offset });
                offset += size;
            }
        }

        if (format !== "binary_little_endian") {
            throw new Error(`Unsupported PLY format '${format}'. Expected binary_little_endian.`);
        }

        const stride = offset;
        const primProp = properties.find((p) => p.name === PRIM_TYPE_FIELD);
        if (!primProp) {
            throw new Error(
                `PLY has no '${PRIM_TYPE_FIELD}' property — this is not a combined GES scene.ply.`
            );
        }

        // Output properties exclude prim_type; output stride drops its bytes.
        const outProps = properties.filter((p) => p.name !== PRIM_TYPE_FIELD);
        const outStride = stride - primProp.size;

        // --- First pass: count each primitive type ---
        let surfelCount = 0;
        let gaussianCount = 0;
        for (let i = 0; i < vertexCount; i++) {
            const t = bytes[dataOffset + i * stride + primProp.offset];
            if (t === PRIM_TYPE_SURFEL) surfelCount++;
            else gaussianCount++;
        }

        // --- Second pass: copy vertex bytes (minus prim_type) into per-type buffers ---
        const surfelBody = new Uint8Array(surfelCount * outStride);
        const gaussianBody = new Uint8Array(gaussianCount * outStride);
        let sCursor = 0;
        let gCursor = 0;

        for (let i = 0; i < vertexCount; i++) {
            const base = dataOffset + i * stride;
            const t = bytes[base + primProp.offset];
            // Copy the bytes before prim_type and the bytes after it (prim_type is last,
            // so the trailing slice is typically empty — but handle either position).
            const head = bytes.subarray(base, base + primProp.offset);
            const tail = bytes.subarray(base + primProp.offset + primProp.size, base + stride);
            if (t === PRIM_TYPE_SURFEL) {
                surfelBody.set(head, sCursor);
                surfelBody.set(tail, sCursor + head.length);
                sCursor += outStride;
            } else {
                gaussianBody.set(head, gCursor);
                gaussianBody.set(tail, gCursor + head.length);
                gCursor += outStride;
            }
        }

        return {
            surfelUrl: surfelCount > 0 ? this.makePlyUrl(outProps, surfelCount, surfelBody) : "",
            gaussianUrl: gaussianCount > 0 ? this.makePlyUrl(outProps, gaussianCount, gaussianBody) : "",
            surfelCount,
            gaussianCount,
        };
    }

    /**
     * Build a standard 3DGS binary_little_endian PLY blob and return an object URL.
     */
    private static makePlyUrl(props: PlyProperty[], count: number, body: Uint8Array): string {
        let header = "ply\n";
        header += "format binary_little_endian 1.0\n";
        header += `element vertex ${count}\n`;
        for (const p of props) {
            header += `property ${p.type} ${p.name}\n`;
        }
        header += "end_header\n";

        const headerBytes = new TextEncoder().encode(header);
        const out = new Uint8Array(headerBytes.length + body.length);
        out.set(headerBytes, 0);
        out.set(body, headerBytes.length);

        const blob = new Blob([out], { type: "application/octet-stream" });
        return URL.createObjectURL(blob);
    }
}
