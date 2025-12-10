// jsnodes.js — robust pair reconciliation for MultiMaskCouple
import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "conditioning.multimaskcouple.perregion.posneg",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "MultiMaskCouple") return;

    const origCreated = nodeType.prototype.onNodeCreated;
    const origConfigure = nodeType.prototype.onConfigure;

    // helpers
    function getWidget(node, name) {
      return node.widgets?.find(w => w.name === name);
    }
    function hasInputNamed(node, name) {
      return !!(node.inputs || []).some(inp => inp?.name === name);
    }
    function addPair(node, i) {
      if (!hasInputNamed(node, `region_${i}_positive`)) node.addInput(`region_${i}_positive`, "CONDITIONING");
      if (!hasInputNamed(node, `region_${i}_negative`)) node.addInput(`region_${i}_negative`, "CONDITIONING");
    }
    function removePair(node, i) {
      // remove *all* inputs that match either name (in case of dupes)
      for (let pass = 0; pass < 2; pass++) {
        const names = (node.inputs || []).map(x => x?.name);
        for (let j = names.length - 1; j >= 0; j--) {
          const n = names[j];
          if (n === `region_${i}_positive` || n === `region_${i}_negative`) {
            node.removeInput(j);
          }
        }
      }
    }
    function listExistingIndices(node) {
      const names = new Set((node.inputs || []).map(x => x?.name).filter(Boolean));
      const indices = [];
      // scan a reasonable upper bound
      for (let i = 1; i <= 32; i++) {
        if (names.has(`region_${i}_positive`) || names.has(`region_${i}_negative`)) {
          indices.push(i);
        }
      }
      return indices;
    }
    function countCurrentPairs(node) {
      const names = new Set((node.inputs || []).map(x => x?.name).filter(Boolean));
      let count = 0;
      for (let i = 1; i <= 32; i++) {
        if (names.has(`region_${i}_positive`) && names.has(`region_${i}_negative`)) count++;
        else if (names.has(`region_${i}_positive`) || names.has(`region_${i}_negative`)) count++; // partial counts as 1 to force repair
        else if (i > 2) break; // beyond predeclared, stop early
      }
      return count;
    }

    function reconcilePairs(node, desiredRaw) {
      // Determine desired count: prefer widget value; else keep current; min 1
      let desired = Math.max(1, Number.isFinite(+desiredRaw) ? +desiredRaw : countCurrentPairs(node));
      desired = Math.min(desired, 32);

      // Ensure 1..desired exist
      for (let i = 1; i <= desired; i++) addPair(node, i);

      // Remove any pairs beyond desired
      const existing = listExistingIndices(node);
      for (const i of existing) {
        if (i > desired) removePair(node, i);
      }

      node.setDirtyCanvas(true, true);
    }

    nodeType.prototype.onNodeCreated = function () {
      origCreated?.apply(this, arguments);

      // Hook widget changes
      const w = getWidget(this, "inputcount");
      if (w) {
        const origCb = w.callback;
        w.callback = (...args) => {
          origCb?.apply(w, args);
          reconcilePairs(this, w.value);
        };
      }

      // Initial pass after widgets exist (creation-time). Don’t assume default=3.
      // Use a microtask to allow Comfy to finish attaching widgets first.
      queueMicrotask(() => {
        const ww = getWidget(this, "inputcount");
        reconcilePairs(this, ww?.value);
      });
    };

    // Also reconcile after the node is deserialized from a saved workflow.
    nodeType.prototype.onConfigure = function () {
      const ret = origConfigure?.apply(this, arguments);
      const w = getWidget(this, "inputcount");
      // Schedule after configure so restored widget values are present.
      setTimeout(() => reconcilePairs(this, w?.value), 0);
      return ret;
    };
  },
});
