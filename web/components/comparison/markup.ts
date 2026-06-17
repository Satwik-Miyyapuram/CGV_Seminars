/**
 * Pure HTML builders for the Comparison tab.
 *
 * These functions only assemble strings — no DOM, no state. Every card on the tab
 * shares the same skeleton (number badge, title, plain-English takeaway, an optional
 * "what's input/intermediate/output" flow, a body, a caption, and collapsible maths),
 * so that skeleton lives here once and the actual content is declared as data in
 * ComparisonTab.ts.
 *
 * Whitespace in the output is irrelevant to the DOM; the only thing that matters is that
 * the emitted ids / classes match what the diagram + viewer classes look up.
 */

/** A single chip in a "viz-flow" strip. `kind` picks the colour, `label` is the small
 *  uppercase tag (input / intermediate / output…), `value` is the human text. */
export interface FlowTag {
    kind: "in" | "mid" | "out";
    label: string;
    value: string;
}

/** Declarative description of one card on the tab. */
export interface CardSpec {
    num: string;          // badge contents, e.g. "0".."4" or "▶"
    title: string;
    take: string;         // plain-English takeaway (HTML allowed)
    numClass?: string;    // extra class on the number badge, e.g. "live"
    cardClass?: string;   // extra class on the card, e.g. "intro"
    /** Stages shown as chips. Each inner array is one "column" of parallel chips;
     *  an arrow is drawn between columns but not within a column. */
    flow?: FlowTag[][];
    body?: string;        // the visual itself (canvas / sliders / viewport HTML)
    caption?: string;     // explanatory caption under the visual
    math?: string;        // inner HTML of the collapsible "show the maths" block
}

const KIND_CLASS: Record<FlowTag["kind"], string> = {
    in: "t-in",
    mid: "t-mid",
    out: "t-out",
};

function flowTag(t: FlowTag): string {
    return `<span class="viz-tag ${KIND_CLASS[t.kind]}"><span class="k">${t.label}</span> ${t.value}</span>`;
}

/** Build a "what flows through this visual" strip: columns separated by → arrows. */
export function vizFlow(columns: FlowTag[][]): string {
    const arrow = `<span class="viz-arrow">→</span>`;
    const columnsHtml = columns.map((col) => col.map(flowTag).join("\n"));
    return `<div class="viz-flow">\n${columnsHtml.join(`\n${arrow}\n`)}\n</div>`;
}

/** Collapsible "show the maths" block. `body` is the inner HTML of `.math-body`. */
export function mathDetails(body: string): string {
    return `<details class="math">
            <summary>show the maths</summary>
            <div class="math-body">${body}</div>
        </details>`;
}

/** A labelled range slider inside a `.control-group`. The id / valueId are what the
 *  diagram classes read and write. */
export function slider(o: {
    id: string;
    label: string;
    valueId: string;
    valueText: string;
    min: string;
    max: string;
    step: string;
    value: string;
}): string {
    return `<div class="control-group">
            <label for="${o.id}">${o.label} <span class="slider-label-val" id="${o.valueId}">${o.valueText}</span></label>
            <input type="range" id="${o.id}" min="${o.min}" max="${o.max}" step="${o.step}" value="${o.value}">
        </div>`;
}

/** An empty status box a diagram fills with live text. */
export function statusBox(id: string): string {
    return `<div id="${id}" class="status-box"></div>`;
}

/** A canvas a 2D diagram draws into. */
export function diagramCanvas(id: string): string {
    return `<canvas class="diagram-canvas" id="${id}"></canvas>`;
}

/** Assemble one card from its spec. The element order (head → take → flow → body →
 *  caption → maths) is fixed; any piece left out of the spec is simply skipped. */
export function card(spec: CardSpec): string {
    const cardClass = spec.cardClass ? ` ${spec.cardClass}` : "";
    const numClass = spec.numClass ? ` ${spec.numClass}` : "";
    return `<div class="ccard${cardClass}">
            <div class="ccard-head">
                <div class="ccard-num${numClass}">${spec.num}</div>
                <h3 class="ccard-title">${spec.title}</h3>
            </div>
            <p class="ccard-take">${spec.take}</p>
            ${spec.flow ? vizFlow(spec.flow) : ""}
            ${spec.body ?? ""}
            ${spec.caption ? `<div class="viz-caption">${spec.caption}</div>` : ""}
            ${spec.math ? mathDetails(spec.math) : ""}
        </div>`;
}
