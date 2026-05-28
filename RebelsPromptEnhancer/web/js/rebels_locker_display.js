import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "RebelAI.PromptLocker.Display",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RebelsPromptLocker") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            // Remove any existing display widget so we don't stack them on re-runs.
            if (this.widgets) {
                const idx = this.widgets.findIndex(w => w.name === "locked_text_display");
                if (idx !== -1) {
                    this.widgets[idx].onRemove?.();
                    this.widgets.splice(idx, 1);
                }
            }

            const text = (message?.text || []).join("");

            const widget = ComfyWidgets["STRING"](
                this,
                "locked_text_display",
                ["STRING", { multiline: true }],
                app
            ).widget;

            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.85;
            widget.inputEl.style.fontFamily = "monospace";
            widget.inputEl.style.fontSize = "11px";
            widget.value = text;

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        };
    },
});