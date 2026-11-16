import {
  ExecutableNodeDTO,
  LGraph,
  LGraphCanvas,
  LGraphEventMode,
  LGraphNode,
  LiteGraph,
} from "@comfyorg/litegraph";
import "@comfyorg/litegraph/style.css";

const clone = (value) => structuredClone(value);
const isLink = (value, nodes) => Array.isArray(value) && value.length === 2
  && /^\d+(?::\d+)*$/.test(String(value[0])) && Number.isInteger(value[1])
  && nodes.has(String(value[0]));

function fieldsFor(definition) {
  const input = definition?.input || {};
  return { ...(input.required || {}), ...(input.optional || {}) };
}

function widgetKind(spec) {
  const type = spec?.[0];
  if (Array.isArray(type) || type === "COMBO") return "combo";
  if (type === "INT" || type === "FLOAT") return "number";
  if (type === "BOOLEAN") return "toggle";
  if (type === "STRING") return "text";
  return null;
}

function widgetDefault(spec) {
  const type = spec?.[0];
  const options = spec?.[1] || {};
  if (options.default !== undefined) return clone(options.default);
  if (Array.isArray(type)) return type[0] ?? "";
  if (type === "COMBO") return (options.options || options.values || [])[0] ?? "";
  if (type === "BOOLEAN") return false;
  if (type === "INT" || type === "FLOAT") return 0;
  return "";
}

function inputType(spec) {
  const type = spec?.[0];
  return Array.isArray(type) ? "COMBO" : typeof type === "string" ? type : "*";
}

export class WorkflowGraphEditor {
  constructor(canvas, { onChange = () => {}, onSelect = () => {} } = {}) {
    this.element = canvas;
    canvas.tabIndex = 0;
    Object.defineProperty(canvas, "workflowGraphEditor", { value: this });
    this.graph = new LGraph();
    this.canvas = new LGraphCanvas(canvas, this.graph, { autoresize: false });
    this.canvas.background_image = "";
    this.canvas.render_canvas_border = false;
    this.canvas.show_info = false;
    this.canvas.low_quality_zoom_threshold = 0.3;
    this.canvas.node_title_color = "#ddd";
    this.canvas.default_link_color = "#9baec1";
    this.canvas.default_connection_color_byType = {
      MODEL: "#b39ddb", CLIP: "#ffd54f", VAE: "#ff8a65", CONDITIONING: "#ffa726",
      LATENT: "#ff80ab", IMAGE: "#90caf9", MASK: "#81c784", INT: "#64b5f6", FLOAT: "#64b5f6",
    };
    LiteGraph.alt_drag_do_clone_nodes = true;
    this.canvas.onRenderBackground = (buffer, ctx) => this.drawBackground(buffer, ctx);
    this.domWidgets = new Map();
    this.domLayer = document.createElement("div");
    this.domLayer.className = "graph-dom-widgets";
    canvas.parentElement.append(this.domLayer);
    this.canvas.onDrawOverlay = () => this.syncDomWidgets();
    this.definitions = {};
    this.registered = new Set();
    this.loading = false;
    this.dirty = false;
    this.activeName = "";
    this.originalUi = null;
    this._compiledApi = null;
    this._compiledApiPromise = null;
    this._serializedUi = null;
    this.apiCompileCount = 0;
    this.revision = 0;
    this.history = [];
    this.historyIndex = -1;
    this.historyTimer = null;
    this.savedSnapshot = "";
    this.onChange = onChange;
    this.onSelect = onSelect;
    this.graph.onAfterChange = () => this.changed();
    this.graph.onConnectionChange = () => this.changed();
    this.graph.onNodeAdded = () => this.changed();
    this.graph.onNodeRemoved = () => this.changed();
    this.graph.onChange = () => this.changed();
    this.canvas.onNodeSelected = (node) => this.onSelect(node);
    this.canvas.onNodeDeselected = () => this.onSelect(null);
    this.canvas.getExtraMenuOptions = () => [
      null,
      { content: "撤销 (Ctrl+Z)", disabled: !this.canUndo, callback: () => this.undo() },
      { content: "重做 (Ctrl+Y)", disabled: !this.canRedo, callback: () => this.redo() },
      { content: "适配画布", callback: () => this.fit() },
    ];
    this.keyHandler = (event) => {
      if (this.element.offsetParent === null
          || event.target.closest?.("input, textarea, select, [contenteditable=true]")) return;
      if ((event.ctrlKey || event.metaKey) && ["z", "y"].includes(event.key.toLowerCase())) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (event.key.toLowerCase() === "y" || event.shiftKey) this.redo();
        else this.undo();
      }
    };
    document.addEventListener("keydown", this.keyHandler, true);
    canvas.addEventListener("pointerup", () => setTimeout(() => this.recordHistory(), 0));
    this.resizeHandler = () => this.resize();
    window.addEventListener("resize", this.resizeHandler);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();
  }

  resize() {
    const bounds = this.element.parentElement.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.resize(Math.round(bounds.width * dpr), Math.round(bounds.height * dpr));
    // LiteGraph uses CSS coordinates for nodes/input and physical pixels for its buffers.
    this.canvas.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.canvas.setDirty(true, true);
    if (this.pixelRatio !== dpr) {
      this.dprQuery?.removeEventListener("change", this.resizeHandler);
      this.pixelRatio = dpr;
      this.dprQuery = matchMedia(`(resolution: ${dpr}dppx)`);
      this.dprQuery.addEventListener("change", this.resizeHandler, { once: true });
    }
  }

  drawBackground(buffer, ctx) {
    const dpr = window.devicePixelRatio || 1;
    const width = buffer.width / dpr;
    const height = buffer.height / dpr;
    const { scale, offset } = this.canvas.ds;
    let step = 24 * scale;
    while (step < 14) step *= 2;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#1a1a1c";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#2d2d33";
    const startX = ((offset[0] * scale) % step + step) % step;
    const startY = ((offset[1] * scale) % step + step) % step;
    for (let x = startX; x < width; x += step) {
      for (let y = startY; y < height; y += step) ctx.fillRect(x, y, 1.2, 1.2);
    }
    ctx.restore();
    return true;
  }

  eventToGraph(event) {
    return this.canvas.convertEventToCanvasOffset(event);
  }

  changed() {
    if (this.loading) return;
    this.revision++;
    this._compiledApi = null;
    this._serializedUi = null;
    this.dirty = true;
    this.onChange();
    this.canvas.setDirty(true, true);
    clearTimeout(this.historyTimer);
    this.historyTimer = setTimeout(() => this.recordHistory(), 180);
  }

  setDefinitions(definitions) {
    this.definitions = definitions || {};
    for (const [type, definition] of Object.entries(this.definitions)) this.register(type, definition);
  }

  register(type, definition = {}, rawNode = null) {
    if (this.registered.has(type)) return;
    const owner = this;
    class ComfyNode extends LGraphNode {
      constructor() {
        super(definition.display_name || type);
        this.comfyClass = type;
        this.type = type;
        this.serialize_widgets = true;
        this._apiInputs = {};
        const fields = fieldsFor(definition);
        for (const [name, spec] of Object.entries(fields)) {
          const kind = widgetKind(spec);
          if (kind && !spec?.[1]?.forceInput) {
            const options = { ...(spec?.[1] || {}) };
            if (Array.isArray(spec?.[0])) options.values = spec[0];
            if (kind === "combo" && !Array.isArray(options.values)) options.values = options.options || [];
            if (kind === "number") {
              options.step2 = options.step || (spec[0] === "INT" ? 1 : 0.01);
              options.precision = spec[0] === "INT" ? 0 : options.precision ?? 3;
            }
            const callback = (value) => {
              this._apiInputs[name] = value;
              owner.changed();
              owner.onSelect(this);
            };
            const widget = kind === "text" && options.multiline
              ? owner.addMultilineWidget(this, name, widgetDefault(spec), options)
              : this.addWidget(kind, name, widgetDefault(spec), callback, options);
            this._apiInputs[name] = widgetDefault(spec);
            if (!options.socketless) this.addInput(name, inputType(spec), { widget: { name } });
            if (options.defaultInput) {
              (this.properties["comfyui_direct.converted"] ||= []).push(name);
            }
            if (options.control_after_generate || name === "seed" || name === "noise_seed") {
              const controlName = typeof options.control_after_generate === "string"
                ? options.control_after_generate : "control_after_generate";
              const control = this.addWidget("combo", controlName, "fixed", () => owner.changed(), {
                values: ["fixed", "increment", "decrement", "randomize"], serialize: false,
              });
              control._seedWidget = widget;
            }
          } else {
            this.addInput(name, inputType(spec));
          }
        }
        if (!Object.keys(fields).length && rawNode) {
          for (const input of rawNode.inputs || []) this.addInput(input.name, input.type || "*");
        }
        const outputs = definition.output || [];
        const names = definition.output_name || [];
        if (outputs.length) outputs.forEach((output, index) => this.addOutput(names[index] || String(output), String(output)));
        else if (rawNode) for (const output of rawNode.outputs || []) this.addOutput(output.name, output.type || "*");
        this.size = [Math.max(250, this.computeSize()[0]), Math.max(90, this.computeSize()[1])];
        owner.bindWidgetInputs(this);
      }

      onSerialize(data) {
        data.properties ||= {};
        data.properties["comfyui_direct.inputs"] = clone(this._apiInputs || {});
        if (this._importedWidgets) data.properties["comfyui_direct.widgets"] = clone(this._importedWidgets);
      }

      onConfigure(data) {
        this._apiInputs = clone(data.properties?.["comfyui_direct.inputs"] || {});
        this._importedWidgets = data.properties?.["comfyui_direct.widgets"] === undefined
          ? undefined : clone(data.properties["comfyui_direct.widgets"]);
        owner.bindWidgetInputs(this);
      }

      getExtraMenuOptions() {
        const entries = (this.widgets || []).filter((widget) => {
          return Object.hasOwn(fieldsFor(definition), widget.name) && !widget.options?.socketless;
        }).map((widget) => ({
          content: `${widget.hidden ? "还原参数" : "转为输入"} · ${widget.name}`,
          callback: () => owner.convertWidgetInput(this, widget),
        }));
        return entries.length ? [{ content: "参数输入", submenu: { options: entries } }] : [];
      }
    }
    ComfyNode.title = definition.display_name || type;
    ComfyNode.desc = definition.description || "";
    LiteGraph.registerNodeType(type, ComfyNode);
    // registerNodeType derives a category from the type name; ComfyUI supplies its own.
    ComfyNode.category = definition.category || "其他";
    this.registered.add(type);
  }

  bindWidgetInputs(node) {
    const converted = node.properties["comfyui_direct.converted"] || [];
    for (const widget of node.widgets || []) {
      if (!Object.hasOwn(fieldsFor(this.definitions[node.type]), widget.name)) continue;
      const input = node.inputs.find((item) => item.name === widget.name);
      widget.hidden = converted.includes(widget.name);
      if (input) {
        input.widget = widget.hidden ? undefined : { name: widget.name };
        input._widget = widget;
      }
    }
  }

  convertWidgetInput(node, widget) {
    const converted = new Set(node.properties["comfyui_direct.converted"] || []);
    if (converted.has(widget.name)) converted.delete(widget.name);
    else converted.add(widget.name);
    node.properties["comfyui_direct.converted"] = [...converted];
    this.bindWidgetInputs(node);
    node.setSize(node.computeSize());
    this.changed();
    this.onSelect(node);
  }

  addMultilineWidget(node, name, value, options) {
    const widget = {
      type: "customtext", name, value, options, y: 0,
      computeLayoutSize: () => ({ minWidth: 260, minHeight: 100, maxHeight: Infinity }),
      draw(ctx, _node, width, y, _height, lowQuality) {
        ctx.fillStyle = "#181818";
        ctx.fillRect(10, y + 4, width - 20, (this.computedHeight || 100) - 8);
        if (lowQuality) return;
        ctx.fillStyle = "#b8b8b8";
        ctx.font = "12px sans-serif";
        ctx.fillText(name, 14, y + 19);
      },
    };
    return node.addCustomWidget(widget);
  }

  syncDomWidgets() {
    const active = new Set();
    const { scale } = this.canvas.ds;
    for (const node of this.canvas.graph?._nodes || []) {
      for (const widget of node.widgets || []) {
        if (widget.type !== "customtext") continue;
        active.add(widget);
        let entry = this.domWidgets.get(widget);
        if (!entry) {
          const element = document.createElement("textarea");
          element.className = "graph-node-textarea";
          element.dataset.nodeId = String(node.id);
          element.dataset.field = widget.name;
          element.setAttribute("aria-label", `#${node.id} ${widget.name}`);
          element.spellcheck = false;
          element.addEventListener("pointerdown", () => this.canvas.selectNode(node));
          element.addEventListener("keydown", (event) => event.stopPropagation());
          element.addEventListener("wheel", (event) => event.stopPropagation());
          element.addEventListener("input", () => {
            widget.value = element.value;
            node._apiInputs[widget.name] = element.value;
            this.changed();
            this.onSelect(node);
          });
          element.addEventListener("change", () => this.recordHistory());
          this.domLayer.append(element);
          entry = { element };
          this.domWidgets.set(widget, entry);
        }
        const { element } = entry;
        const visible = !node.flags?.collapsed && !widget.hidden && scale >= 0.2;
        element.hidden = !visible;
        if (!visible) continue;
        const [x, y] = this.canvas.ds.convertOffsetToCanvas([node.pos[0] + 10, node.pos[1] + widget.y + 4]);
        element.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        element.style.width = `${node.size[0] - 20}px`;
        element.style.height = `${(widget.computedHeight || 100) - 8}px`;
        element.disabled = node.inputs.some((input) => input.name === widget.name && input.link != null);
        if (element.value !== String(widget.value ?? "") && document.activeElement !== element) {
          element.value = String(widget.value ?? "");
        }
      }
    }
    for (const [widget, { element }] of this.domWidgets) {
      if (active.has(widget)) continue;
      element.remove();
      this.domWidgets.delete(widget);
    }
  }

  load(name, api, ui = null) {
    this.loading = true;
    try {
      this.activeName = name || "";
      this.originalUi = ui ? clone(ui) : null;
      this.revision++;
      this._compiledApi = null;
      this._compiledApiPromise = null;
      this._serializedUi = null;
      this.graph.clear();
      const entries = Object.entries(api || {});
      for (const [, node] of entries) this.register(node.class_type, this.definitions[node.class_type]);
      if (ui?.nodes) for (const node of ui.nodes) this.register(node.type, this.definitions[node.type], node);
      if (ui?.nodes?.length) {
        try {
          this.graph.configure(clone(ui));
        } catch (error) {
          this.graph.clear();
          this.buildFromApi(api);
          console.warn("ComfyUI 画布恢复失败，使用 API 图重建", error);
        }
      } else this.buildFromApi(api);
      for (const node of this.graph._nodes) {
        const entry = api?.[String(node.id)];
        node._apiInputs = entry ? clone(entry.inputs || {}) : node._apiInputs || {};
        node._importedWidgets = ui?.nodes?.find((item) => String(item.id) === String(node.id))?.widgets_values;
        if (!entry && node.type.includes("Power Lora Loader") && Array.isArray(node._importedWidgets)) {
          let index = 1;
          for (const value of node._importedWidgets) {
            if (value && typeof value === "object" && !Array.isArray(value) && "lora" in value) {
              node._apiInputs[`lora_${index++}`] = clone(value);
            }
          }
        }
        this.syncWidgets(node);
        this.bindWidgetInputs(node);
      }
      if (!ui && entries.length && !entries.some(([, node]) => node._meta?.pos)) this.graph.arrange(50);
      this.dirty = false;
      this.canvas.setDirty(true, true);
      const view = ui?.extra?.ds;
      if (view && Number.isFinite(view.scale) && view.scale > 0 && Array.isArray(view.offset)
          && view.offset.length === 2 && view.offset.every(Number.isFinite)) {
        this.canvas.ds.scale = Math.min(this.canvas.ds.max_scale, Math.max(this.canvas.ds.min_scale, view.scale));
        this.canvas.ds.offset = view.offset;
      } else this.fit();
      this.canvas.draw(true, true);
      this.resetHistory();
      this.onSelect(null);
      this.onChange();
    } finally {
      this.loading = false;
    }
  }

  buildFromApi(api) {
    const entries = Object.entries(api || {});
    const nodes = new Map();
    for (const [id, data] of entries) {
      const node = LiteGraph.createNode(data.class_type);
      if (!node) throw new Error(`无法创建节点 ${data.class_type}`);
      node.id = Number(id);
      if (data._meta?.title) node.title = data._meta.title;
      if (data._meta?.pos) node.pos = [Number(data._meta.pos.x) || 0, Number(data._meta.pos.y) || 0];
      this.graph.add(node);
      nodes.set(String(id), node);
    }
    for (const [id, data] of entries) {
      const target = nodes.get(String(id));
      for (const [field, value] of Object.entries(data.inputs || {})) {
        if (!isLink(value, nodes)) continue;
        const source = nodes.get(String(value[0]));
        let slot = target.findInputSlot(field);
        if (slot < 0) {
          target.addInput(field, source.outputs?.[value[1]]?.type || "*");
          slot = target.findInputSlot(field);
        }
        source.connect(value[1], target, slot);
      }
    }
  }

  syncWidgets(node) {
    const inputs = node._apiInputs || {};
    for (const widget of node.widgets || []) {
      if (Object.hasOwn(inputs, widget.name) && !Array.isArray(inputs[widget.name])) widget.value = clone(inputs[widget.name]);
    }
  }

  addNode(type, pos = null) {
    this.register(type, this.definitions[type]);
    const node = LiteGraph.createNode(type);
    if (!node) throw new Error(`远端没有节点 ${type}`);
    const bounds = this.element.getBoundingClientRect();
    node.pos = pos || this.canvas.ds.convertCanvasToOffset([bounds.width / 2, bounds.height / 2]);
    this.graph.add(node);
    this.canvas.selectNode(node);
    this.changed();
    return node;
  }

  deleteSelected() {
    this.canvas.deleteSelected();
    this.onSelect(null);
    this.changed();
  }

  historySnapshot() {
    const snapshot = clone(this.graph.serialize());
    // Execution order is derived during prompt compilation, not a user edit.
    for (const node of snapshot.nodes) delete node.order;
    // LiteGraph updates extra.ds on serialize; moving the camera must preserve redo.
    if (snapshot.extra) delete snapshot.extra.ds;
    return JSON.stringify(snapshot);
  }

  resetHistory() {
    clearTimeout(this.historyTimer);
    const snapshot = this.historySnapshot();
    this.history = [snapshot];
    this.historyIndex = 0;
    this.savedSnapshot = snapshot;
    this.dirty = false;
  }

  recordHistory() {
    clearTimeout(this.historyTimer);
    if (this.loading || this.canvas.pointer.isDown) return;
    const snapshot = this.historySnapshot();
    if (snapshot !== this.history[this.historyIndex]) {
      this.history.splice(this.historyIndex + 1);
      this.history.push(snapshot);
      if (this.history.length > 100) this.history.shift();
      this.historyIndex = this.history.length - 1;
    }
    this.dirty = snapshot !== this.savedSnapshot;
    this.onChange();
  }

  get canUndo() { return this.historyIndex > 0; }
  get canRedo() { return this.historyIndex < this.history.length - 1; }

  undo() {
    this.recordHistory();
    if (this.canUndo) this.restoreHistory(this.historyIndex - 1);
  }

  redo() {
    this.recordHistory();
    if (this.canRedo) this.restoreHistory(this.historyIndex + 1);
  }

  restoreHistory(index) {
    const view = { scale: this.canvas.ds.scale, offset: [...this.canvas.ds.offset] };
    this.loading = true;
    try {
      this.canvas.deselectAllNodes();
      this.graph.configure(JSON.parse(this.history[index]));
      this.canvas.ds.scale = view.scale;
      this.canvas.ds.offset = view.offset;
      this.historyIndex = index;
      this.revision++;
      this._compiledApi = this._serializedUi = null;
      this.dirty = this.history[index] !== this.savedSnapshot;
      this.canvas.setDirty(true, true);
      this.onSelect(null);
      this.onChange();
    } finally { this.loading = false; }
  }

  afterQueued() {
    for (const node of this.graph._nodes) {
      for (const control of node.widgets || []) {
        const target = control._seedWidget;
        if (!target || control.value === "fixed"
            || node.inputs.some((input) => input.name === target.name && input.link != null)) continue;
        const min = Math.max(0, target.options?.min ?? 0);
        const max = Math.min(Number.MAX_SAFE_INTEGER, target.options?.max ?? Number.MAX_SAFE_INTEGER);
        let value = Number(target.value);
        if (control.value === "increment") value = value >= max ? min : value + 1;
        else if (control.value === "decrement") value = value <= min ? max : value - 1;
        else if (control.value === "randomize") value = min + Math.floor(Math.random() * (max - min + 1));
        target.value = Math.max(min, Math.min(max, value));
        node._apiInputs[target.name] = target.value;
        this.changed();
        if (this.canvas.selected_nodes[node.id]) this.onSelect(node);
      }
    }
  }

  async exportWorkflow() {
    return this.exportWorkflowSnapshot();
  }

  async exportWorkflowSnapshot() {
    while (true) {
      const revision = this.revision;
      const api = await this.getCompiledApi(revision);
      if (revision !== this.revision) continue;
      if (!this._serializedUi) this._serializedUi = this.serializeUiWorkflow(api);
      if (revision !== this.revision) {
        this._serializedUi = null;
        continue;
      }
      const ui = clone(this._serializedUi);
      ui.extra.ds = { scale: this.canvas.ds.scale, offset: [...this.canvas.ds.offset] };
      return { workflow: clone(api), ui_workflow: ui };
    }
  }

  async getCompiledApi(revision) {
    if (this._compiledApi) return this._compiledApi;
    if (!this._compiledApiPromise) {
      this._compiledApiPromise = this.compileApiWorkflow().then((api) => {
        if (revision === this.revision) this._compiledApi = api;
        return api;
      }).finally(() => {
        this._compiledApiPromise = null;
      });
    }
    const api = await this._compiledApiPromise;
    if (revision !== this.revision) return this.getCompiledApi(this.revision);
    return api;
  }

  async compileApiWorkflow() {
    this.apiCompileCount++;
    const nodeDtoMap = new Map();
    for (const node of this.graph.computeExecutionOrder(false)) {
      const dto = new ExecutableNodeDTO(node, [], nodeDtoMap);
      nodeDtoMap.set(dto.id, dto);
      if (node.mode === LGraphEventMode.NEVER || node.mode === LGraphEventMode.BYPASS) continue;
      for (const innerNode of dto.getInnerNodes()) nodeDtoMap.set(innerNode.id, innerNode);
    }
    const api = {};
    const executionIds = new Set(nodeDtoMap.keys());
    for (const dto of nodeDtoMap.values()) {
      if (dto.isVirtualNode || dto.mode === LGraphEventMode.NEVER || dto.mode === LGraphEventMode.BYPASS) continue;
      const node = dto.node;
      const inputs = clone(node._apiInputs || {});
      for (const [widgetIndex, widget] of (node.widgets || []).entries()) {
        if (!widget.name || widget.options?.serialize === false) continue;
        const value = widget.serializeValue ? await widget.serializeValue(node, widgetIndex) : widget.value;
        inputs[widget.name] = widget.type === "curve" && value != null
          ? { __type__: "CURVE", __value__: clone(value) }
          : Array.isArray(value) ? { __value__: clone(value) } : clone(value);
      }
      for (const [slot, input] of dto.inputs.entries()) {
        const resolved = dto.resolveInput(slot);
        if (!resolved) {
          if (isLink(inputs[input.name], executionIds)) delete inputs[input.name];
          continue;
        }
        if (resolved.widgetInfo) {
          const value = resolved.widgetInfo.value;
          inputs[input.name] = Array.isArray(value) ? { __value__: clone(value) } : clone(value);
        } else {
          inputs[input.name] = [resolved.origin_id, Number.parseInt(resolved.origin_slot, 10)];
        }
      }
      for (const [field, value] of Object.entries(inputs)) {
        if (isLink(value, executionIds) && !dto.inputs.some((input) => input.name === field && input.linkId != null)) {
          delete inputs[field];
        }
      }
      const title = dto.title || node.title || dto.type;
      api[String(dto.id)] = {
        class_type: dto.comfyClass || node.comfyClass || dto.type,
        inputs,
        _meta: {
          title,
          ...(node.pos ? { pos: { x: node.pos[0], y: node.pos[1] } } : {}),
        },
      };
    }
    const outputIds = new Set(Object.keys(api));
    for (const node of Object.values(api)) {
      for (const [name, value] of Object.entries(node.inputs)) {
        if (Array.isArray(value) && value.length === 2 && !outputIds.has(String(value[0]))) delete node.inputs[name];
      }
    }
    return api;
  }

  serializeUiWorkflow(api) {
    const ui = clone(this.graph.serialize());
    ui.extra ||= {};
    ui.extra.ds = { scale: this.canvas.ds.scale, offset: [...this.canvas.ds.offset] };
    const uiNodes = new Map(ui.nodes.map((node) => [String(node.id), node]));
    for (const [id, node] of uiNodes) {
      const live = this.graph.getNodeById(Number(id));
      if (!live) continue;
      if (Array.isArray(live._importedWidgets) && live._importedWidgets.length > (live.widgets || []).length) {
        node.widgets_values = clone(live._importedWidgets);
      }
      if (node.type?.includes("Power Lora Loader") && api[id]) {
        const loras = Object.entries(api[id].inputs)
          .filter(([key, value]) => /^lora_\d+$/.test(key) && value && typeof value === "object" && !Array.isArray(value))
          .sort(([a], [b]) => Number(a.slice(5)) - Number(b.slice(5)))
          .map(([, value]) => value);
        if (loras.length) node.widgets_values = clone(loras);
      }
    }
    return ui;
  }

  markSaved() {
    this.recordHistory();
    this.savedSnapshot = this.historySnapshot();
    this.dirty = false;
    this.onChange();
  }

  fit() {
    const nodes = this.graph._nodes;
    if (!nodes.length) return;
    const left = Math.min(...nodes.map((node) => node.pos[0]));
    const top = Math.min(...nodes.map((node) => node.pos[1]));
    const right = Math.max(...nodes.map((node) => node.pos[0] + node.size[0]));
    const bottom = Math.max(...nodes.map((node) => node.pos[1] + node.size[1]));
    this.canvas.ds.fitToBounds([left - 60, top - 60, right - left + 120, bottom - top + 120]);
    this.canvas.setDirty(true, true);
  }
}
