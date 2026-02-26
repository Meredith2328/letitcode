(function () {
  const STORAGE_PREFIX = "pytorch_practice_code_";
  const STORAGE_FAVS = "pytorch_practice_favorites";
  const STORAGE_CUSTOM_PREFIX = "pytorch_practice_custom_";
  const STORAGE_LAYOUT = "pytorch_practice_layout_v1";

  const SPLITTER_SIZE = 8;
  const LAYOUT_LIMITS = {
    leftMin: 320,
    rightMin: 420,
    topMin: 190,
    bottomMin: 190,
  };

  const el = {
    workspace: document.getElementById("workspace"),
    rightPanel: document.getElementById("rightPanel"),
    splitterVertical: document.getElementById("splitterVertical"),
    splitterHorizontal: document.getElementById("splitterHorizontal"),
    problemSelect: document.getElementById("problemSelect"),
    prevProblemBtn: document.getElementById("prevProblemBtn"),
    nextProblemBtn: document.getElementById("nextProblemBtn"),
    problemTitle: document.getElementById("problemTitle"),
    problemMeta: document.getElementById("problemMeta"),
    problemDescription: document.getElementById("problemDescription"),
    solutionDescription: document.getElementById("solutionDescription"),
    tabProblemBtn: document.getElementById("tabProblemBtn"),
    tabSolutionBtn: document.getElementById("tabSolutionBtn"),
    favoriteBtn: document.getElementById("favoriteBtn"),
    favoritesOnlyChk: document.getElementById("favoritesOnlyChk"),
    feedbackBtn: document.getElementById("feedbackBtn"),
    feedbackModal: document.getElementById("feedbackModal"),
    feedbackCloseBtn: document.getElementById("feedbackCloseBtn"),
    feedbackCancelBtn: document.getElementById("feedbackCancelBtn"),
    feedbackSaveBtn: document.getElementById("feedbackSaveBtn"),
    feedbackStatus: document.getElementById("feedbackStatus"),
    feedbackTabProblemBtn: document.getElementById("feedbackTabProblemBtn"),
    feedbackTabSolutionBtn: document.getElementById("feedbackTabSolutionBtn"),
    feedbackProblemPanel: document.getElementById("feedbackProblemPanel"),
    feedbackSolutionPanel: document.getElementById("feedbackSolutionPanel"),
    feedbackProblemDescriptionInput: document.getElementById("feedbackProblemDescriptionInput"),
    feedbackStarterCodeInput: document.getElementById("feedbackStarterCodeInput"),
    feedbackSolutionExplanationInput: document.getElementById("feedbackSolutionExplanationInput"),
    feedbackSolutionCodeInput: document.getElementById("feedbackSolutionCodeInput"),
    testBtn: document.getElementById("testBtn"),
    submitBtn: document.getElementById("submitBtn"),
    customCasesInput: document.getElementById("customCasesInput"),
    resultOutput: document.getElementById("resultOutput"),
    editorHost: document.getElementById("editor"),
    plainEditor: document.getElementById("plainEditor"),
  };

  const state = {
    problems: [],
    details: new Map(),
    solutions: new Map(),
    favorites: new Set(),
    favoritesOnly: false,
    filteredIds: [],
    currentProblemId: null,
    activeLeftTab: "problem",
    activeFeedbackTab: "problem",
    monacoEditor: null,
    monacoReady: false,
    usePlainEditor: false,
    isRunning: false,
    leftRatio: 0.46,
    topRatio: 0.44,
    resizeMode: null,
    feedbackLoading: false,
  };

  function init() {
    state.favorites = loadFavorites();
    loadLayoutState();
    state.favoritesOnly = false;
    if (el.favoritesOnlyChk) {
      el.favoritesOnlyChk.checked = false;
    }
    bindEvents();
    initResizableLayout();
    initEditor();
    bootstrap();
  }

  async function bootstrap() {
    try {
      const resp = await fetch("/api/problems");
      const data = await safeParseJson(resp);
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      state.problems = data.problems || [];
      if (!state.problems.length) {
        writeOutput("未找到题目。");
        return;
      }
      rebuildProblemOptions();
      if (state.filteredIds.length > 0) {
        await selectProblem(state.filteredIds[0]);
      }
    } catch (err) {
      writeOutput("加载题目列表失败: " + String(err));
    }
  }

  function bindEvents() {
    el.problemSelect.addEventListener("change", async (e) => {
      await selectProblem(e.target.value);
    });

    el.prevProblemBtn.addEventListener("click", async () => {
      const idx = state.filteredIds.indexOf(state.currentProblemId);
      if (idx > 0) {
        await selectProblem(state.filteredIds[idx - 1]);
      }
    });

    el.nextProblemBtn.addEventListener("click", async () => {
      const idx = state.filteredIds.indexOf(state.currentProblemId);
      if (idx >= 0 && idx < state.filteredIds.length - 1) {
        await selectProblem(state.filteredIds[idx + 1]);
      }
    });

    el.tabProblemBtn.addEventListener("click", () => {
      setActiveTab("problem");
    });

    el.tabSolutionBtn.addEventListener("click", async () => {
      setActiveTab("solution");
      await ensureSolutionLoaded(state.currentProblemId);
    });

    el.favoriteBtn.addEventListener("click", () => {
      const pid = state.currentProblemId;
      if (!pid) return;
      if (state.favorites.has(pid)) {
        state.favorites.delete(pid);
      } else {
        state.favorites.add(pid);
      }
      saveFavorites(state.favorites);
      updateFavoriteButton();
      rebuildProblemOptions();
      if (!state.filteredIds.includes(pid) && state.filteredIds.length > 0) {
        selectProblem(state.filteredIds[0]);
      }
      if (state.filteredIds.length === 0) {
        clearProblemView();
      }
    });

    el.favoritesOnlyChk.addEventListener("change", async (e) => {
      state.favoritesOnly = e.target.checked;
      const previous = state.currentProblemId;
      rebuildProblemOptions();
      if (state.filteredIds.length === 0) {
        clearProblemView();
        writeOutput("收藏为空。");
        return;
      }
      if (previous && state.filteredIds.includes(previous)) {
        await selectProblem(previous);
      } else {
        await selectProblem(state.filteredIds[0]);
      }
    });

    el.feedbackBtn.addEventListener("click", () => {
      openFeedbackModal();
    });

    el.feedbackCloseBtn.addEventListener("click", closeFeedbackModal);
    el.feedbackCancelBtn.addEventListener("click", closeFeedbackModal);
    el.feedbackSaveBtn.addEventListener("click", saveFeedbackEdits);

    el.feedbackModal.addEventListener("click", (event) => {
      const target = event.target;
      if (target instanceof HTMLElement && target.dataset.closeFeedback === "1") {
        closeFeedbackModal();
      }
    });

    el.feedbackTabProblemBtn.addEventListener("click", () => {
      setFeedbackTab("problem");
    });
    el.feedbackTabSolutionBtn.addEventListener("click", () => {
      setFeedbackTab("solution");
    });

    el.testBtn.addEventListener("click", () => runJudge("test"));
    el.submitBtn.addEventListener("click", () => runJudge("submit"));

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !el.feedbackModal.classList.contains("hidden")) {
        closeFeedbackModal();
      }
    });

    window.addEventListener("beforeunload", () => {
      persistCurrentCode();
      persistCurrentCustomCases();
      saveLayoutState();
    });
  }

  function setActiveTab(tab) {
    state.activeLeftTab = tab;
    const showProblem = tab === "problem";
    el.tabProblemBtn.classList.toggle("active", showProblem);
    el.tabSolutionBtn.classList.toggle("active", !showProblem);
    el.problemDescription.classList.toggle("hidden", !showProblem);
    el.solutionDescription.classList.toggle("hidden", showProblem);
  }

  function setFeedbackTab(tab) {
    state.activeFeedbackTab = tab;
    const showProblem = tab === "problem";
    el.feedbackTabProblemBtn.classList.toggle("active", showProblem);
    el.feedbackTabSolutionBtn.classList.toggle("active", !showProblem);
    el.feedbackProblemPanel.classList.toggle("hidden", !showProblem);
    el.feedbackSolutionPanel.classList.toggle("hidden", showProblem);
  }

  async function openFeedbackModal() {
    const pid = state.currentProblemId;
    if (!pid) {
      writeOutput("请先选择题目。");
      return;
    }

    persistCurrentCode();
    persistCurrentCustomCases();
    setFeedbackTab("problem");
    el.feedbackModal.classList.remove("hidden");
    setFeedbackStatus("加载中...");
    setFeedbackLoading(true);

    try {
      const resp = await fetch(`/api/problems/${pid}/feedback`);
      const data = await safeParseJson(resp);
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      if (!data.ok) {
        throw new Error(data.error || "加载失败");
      }
      const fb = data.feedback || {};
      const problem = fb.problem || {};
      const solution = fb.solution || {};
      el.feedbackProblemDescriptionInput.value = problem.description || "";
      el.feedbackStarterCodeInput.value = problem.starter_code || "";
      el.feedbackSolutionExplanationInput.value = solution.explanation || "";
      el.feedbackSolutionCodeInput.value = solution.code || "";
      setFeedbackStatus("已加载，可编辑后保存。");
    } catch (err) {
      setFeedbackStatus("加载反馈编辑失败: " + String(err));
    } finally {
      setFeedbackLoading(false);
    }
  }

  function closeFeedbackModal() {
    if (state.feedbackLoading) return;
    el.feedbackModal.classList.add("hidden");
    setFeedbackStatus("");
  }

  function setFeedbackLoading(isLoading) {
    state.feedbackLoading = isLoading;
    el.feedbackSaveBtn.disabled = isLoading;
    el.feedbackCloseBtn.disabled = isLoading;
    el.feedbackCancelBtn.disabled = isLoading;
  }

  function setFeedbackStatus(text) {
    el.feedbackStatus.textContent = text || "";
  }

  async function saveFeedbackEdits() {
    const pid = state.currentProblemId;
    if (!pid) return;

    const payload = {
      problem: {
        description: el.feedbackProblemDescriptionInput.value,
        starter_code: el.feedbackStarterCodeInput.value,
      },
      solution: {
        explanation: el.feedbackSolutionExplanationInput.value,
        code: el.feedbackSolutionCodeInput.value,
      },
    };

    setFeedbackStatus("保存中...");
    setFeedbackLoading(true);

    try {
      const resp = await fetch(`/api/problems/${pid}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await safeParseJson(resp);
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      if (!data.ok) {
        throw new Error(data.error || "保存失败");
      }

      if (data.problem) {
        state.details.set(pid, data.problem);
      }
      if (data.solution) {
        state.solutions.set(pid, data.solution);
      }

      persistCurrentCode();
      persistCurrentCustomCases();
      if (data.problem) {
        renderProblem(data.problem);
      }
      if (state.activeLeftTab === "solution" && data.solution) {
        renderSolution(data.solution);
      }

      setFeedbackStatus("已保存。");
    } catch (err) {
      setFeedbackStatus("保存失败: " + String(err));
    } finally {
      setFeedbackLoading(false);
    }
  }

  async function ensureSolutionLoaded(problemId) {
    if (!problemId) return;
    const cached = state.solutions.get(problemId);
    if (cached) {
      renderSolution(cached);
      return;
    }

    renderMarkdownTo(el.solutionDescription, "加载题解中...");
    try {
      const resp = await fetch(`/api/problems/${problemId}/solution`);
      const data = await safeParseJson(resp);
      if (!resp.ok) {
        const hint = resp.status === 404 ? "（可能后端未重启到最新版本）" : "";
        throw new Error(`HTTP ${resp.status} ${hint}`);
      }
      if (!data.ok) {
        throw new Error(data.error || "加载失败");
      }
      state.solutions.set(problemId, data.solution);
      if (state.currentProblemId === problemId && state.activeLeftTab === "solution") {
        renderSolution(data.solution);
      }
    } catch (err) {
      renderMarkdownTo(el.solutionDescription, `题解加载失败:\n\n${String(err)}`);
    }
  }

  function renderSolution(solution) {
    const explanation = solution?.explanation || "暂无讲解。";
    const code = solution?.code || "";
    const composed = `${explanation}\n\n## 参考代码\n\n\`\`\`python\n${code}\n\`\`\``;
    renderMarkdownTo(el.solutionDescription, composed);
  }

  function initResizableLayout() {
    applyLayout();
    window.addEventListener("resize", applyLayout);

    el.splitterVertical?.addEventListener("pointerdown", (event) => {
      if (isNarrowMode()) return;
      startResize("vertical", event);
    });

    el.splitterHorizontal?.addEventListener("pointerdown", (event) => {
      if (isNarrowMode()) return;
      startResize("horizontal", event);
    });

    window.addEventListener("pointermove", onResizeMove);
    window.addEventListener("pointerup", endResize);
    window.addEventListener("pointercancel", endResize);
  }

  function startResize(mode, event) {
    event.preventDefault();
    state.resizeMode = mode;
    document.body.style.userSelect = "none";
    document.body.style.cursor = mode === "vertical" ? "col-resize" : "row-resize";
  }

  function onResizeMove(event) {
    if (!state.resizeMode || isNarrowMode()) return;

    if (state.resizeMode === "vertical") {
      const rect = el.workspace.getBoundingClientRect();
      const available = rect.width - SPLITTER_SIZE;
      if (available <= 0) return;
      const rawLeft = event.clientX - rect.left;
      const leftPx = clampByMinBounds(rawLeft, available, LAYOUT_LIMITS.leftMin, LAYOUT_LIMITS.rightMin);
      state.leftRatio = leftPx / available;
      applyLayout();
      return;
    }

    const rect = el.rightPanel.getBoundingClientRect();
    const available = rect.height - SPLITTER_SIZE;
    if (available <= 0) return;
    const rawTop = event.clientY - rect.top;
    const topPx = clampByMinBounds(rawTop, available, LAYOUT_LIMITS.topMin, LAYOUT_LIMITS.bottomMin);
    state.topRatio = topPx / available;
    applyLayout();
  }

  function endResize() {
    if (!state.resizeMode) return;
    state.resizeMode = null;
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    saveLayoutState();
  }

  function applyLayout() {
    if (isNarrowMode()) {
      el.workspace.style.gridTemplateColumns = "1fr";
      el.workspace.style.gridTemplateRows = "minmax(0, 1fr) minmax(0, 1fr)";
      el.rightPanel.style.gridTemplateRows = "minmax(180px, 1fr) minmax(180px, 1fr)";
      return;
    }

    el.workspace.style.gridTemplateRows = "";
    const workspaceWidth = el.workspace.clientWidth;
    if (workspaceWidth > 0) {
      const available = workspaceWidth - SPLITTER_SIZE;
      const fallbackLeft = available * 0.46;
      const leftPx = clampByMinBounds(
        available * state.leftRatio,
        available,
        LAYOUT_LIMITS.leftMin,
        LAYOUT_LIMITS.rightMin,
        fallbackLeft
      );
      state.leftRatio = available > 0 ? leftPx / available : state.leftRatio;
      el.workspace.style.gridTemplateColumns = `${Math.round(leftPx)}px ${SPLITTER_SIZE}px minmax(0, 1fr)`;
    }

    const rightHeight = el.rightPanel.clientHeight;
    if (rightHeight > 0) {
      const available = rightHeight - SPLITTER_SIZE;
      const fallbackTop = available * 0.44;
      const topPx = clampByMinBounds(
        available * state.topRatio,
        available,
        LAYOUT_LIMITS.topMin,
        LAYOUT_LIMITS.bottomMin,
        fallbackTop
      );
      state.topRatio = available > 0 ? topPx / available : state.topRatio;
      el.rightPanel.style.gridTemplateRows = `${Math.round(topPx)}px ${SPLITTER_SIZE}px minmax(0, 1fr)`;
    }
  }

  function clampByMinBounds(value, total, minFirst, minSecond, fallback) {
    const min = minFirst;
    const max = total - minSecond;
    if (max < min) {
      const fallbackValue = typeof fallback === "number" ? fallback : total / 2;
      return clamp(fallbackValue, 120, Math.max(120, total - 120));
    }
    return clamp(value, min, max);
  }

  async function selectProblem(problemId) {
    if (!problemId) return;
    if (state.currentProblemId === problemId && state.details.has(problemId)) {
      renderProblem(state.details.get(problemId));
      if (state.activeLeftTab === "solution") {
        await ensureSolutionLoaded(problemId);
      }
      return;
    }

    persistCurrentCode();
    persistCurrentCustomCases();
    state.currentProblemId = problemId;
    updateNavButtons();

    if (state.details.has(problemId)) {
      renderProblem(state.details.get(problemId));
      if (state.activeLeftTab === "solution") {
        await ensureSolutionLoaded(problemId);
      }
      return;
    }

    try {
      const resp = await fetch(`/api/problems/${problemId}`);
      const data = await safeParseJson(resp);
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      if (!data.ok) {
        throw new Error(data.error || "加载失败");
      }
      state.details.set(problemId, data.problem);
      renderProblem(data.problem);
      if (state.activeLeftTab === "solution") {
        await ensureSolutionLoaded(problemId);
      }
    } catch (err) {
      writeOutput("加载题目失败: " + String(err));
    }
  }

  function renderProblem(problem) {
    el.problemSelect.value = problem.id;
    el.problemTitle.textContent = problem.title;
    renderMarkdownTo(el.problemDescription, problem.description || "");

    const diffClass = difficultyClass(problem.difficulty);
    const tags = problem.tags || [];
    const tagHtml = tags.map((t) => `<span class="meta-tag">${escapeHtml(String(t))}</span>`).join("");
    el.problemMeta.innerHTML = `<span class="meta-diff ${diffClass}">${escapeHtml(
      problem.difficulty
    )}</span>${tagHtml}`;

    updateFavoriteButton();
    updateNavButtons();

    const savedCode = loadProblemCode(problem.id);
    setEditorValue(savedCode || problem.starter_code || "");

    const savedCustom = loadCustomCases(problem.id);
    el.customCasesInput.value =
      savedCustom || JSON.stringify(problem.custom_case_example || [], null, 2);
  }

  function rebuildProblemOptions() {
    const problems = state.problems.filter((p) => !state.favoritesOnly || state.favorites.has(p.id));
    state.filteredIds = problems.map((p) => p.id);

    const current = state.currentProblemId;
    el.problemSelect.innerHTML = "";
    for (const p of problems) {
      const option = document.createElement("option");
      option.value = p.id;
      option.textContent = p.title;
      el.problemSelect.appendChild(option);
    }

    el.problemSelect.disabled = problems.length === 0;
    el.prevProblemBtn.disabled = problems.length <= 1;
    el.nextProblemBtn.disabled = problems.length <= 1;

    if (current && state.filteredIds.includes(current)) {
      el.problemSelect.value = current;
    } else if (state.filteredIds.length > 0) {
      el.problemSelect.value = state.filteredIds[0];
    }
  }

  function clearProblemView() {
    state.currentProblemId = null;
    el.problemTitle.textContent = "无可显示题目";
    el.problemMeta.innerHTML = "";
    el.problemDescription.textContent = "";
    el.solutionDescription.textContent = "";
    setEditorValue("");
  }

  function updateFavoriteButton() {
    const pid = state.currentProblemId;
    const fav = pid && state.favorites.has(pid);
    el.favoriteBtn.classList.toggle("active", Boolean(fav));
    el.favoriteBtn.textContent = fav ? "★ 已收藏" : "☆ 收藏";
  }

  function updateNavButtons() {
    const idx = state.filteredIds.indexOf(state.currentProblemId);
    el.prevProblemBtn.disabled = idx <= 0;
    el.nextProblemBtn.disabled = idx < 0 || idx >= state.filteredIds.length - 1;
  }

  function renderMarkdownTo(target, markdownText) {
    const markedApi = window.marked;
    const parse =
      markedApi && typeof markedApi.parse === "function"
        ? markedApi.parse.bind(markedApi)
        : typeof markedApi === "function"
          ? markedApi
          : null;

    if (!parse) {
      target.textContent = markdownText;
      return;
    }

    const rawHtml = parse(markdownText, { breaks: true, gfm: true });
    if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
      target.innerHTML = window.DOMPurify.sanitize(rawHtml);
    } else {
      target.innerHTML = rawHtml;
    }
  }

  async function runJudge(mode) {
    const pid = state.currentProblemId;
    if (!pid) {
      writeOutput("请先选择题目。");
      return;
    }

    const code = getEditorValue();
    if (!code.trim()) {
      writeOutput("代码为空。");
      return;
    }

    persistCurrentCode();
    persistCurrentCustomCases();

    let customCases = null;
    if (mode === "test") {
      const raw = el.customCasesInput.value.trim();
      if (raw) {
        try {
          customCases = JSON.parse(raw);
        } catch (err) {
          writeOutput("自定义用例 JSON 解析失败: " + String(err));
          return;
        }
      }
    }

    setRunning(true);
    writeOutput(mode === "test" ? "运行可见测试中..." : "运行提交（隐藏用例）中...");

    try {
      const url = mode === "test" ? "/api/test" : "/api/submit";
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_id: pid,
          code,
          custom_cases: customCases,
        }),
      });

      const data = await safeParseJson(resp);
      if (!data.ok) {
        writeOutput("请求失败: " + (data.error || "unknown error"));
        return;
      }
      writeOutput(formatJudgeResult(data.result, mode));
    } catch (err) {
      writeOutput("请求异常: " + String(err));
    } finally {
      setRunning(false);
    }
  }

  function formatJudgeResult(result, mode) {
    if (!result) return "空结果。";
    const lines = [];

    if (result.compile_error) {
      lines.push("编译错误:");
      lines.push(result.compile_error);
      return lines.join("\n");
    }
    if (result.runtime_error) {
      lines.push("运行错误:");
      lines.push(result.runtime_error);
      return lines.join("\n");
    }

    lines.push(result.passed ? "通过" : "未通过");
    lines.push(
      `通过用例: ${result.summary?.passed ?? 0}/${result.summary?.total ?? 0} | 耗时: ${
        result.elapsed_ms ?? 0
      }ms`
    );
    lines.push("");

    const cases = result.cases || [];
    for (const c of cases) {
      const mark = c.passed ? "[PASS]" : "[FAIL]";
      const diffText =
        typeof c.max_abs_diff === "number" ? ` | max_abs_diff=${c.max_abs_diff.toExponential(3)}` : "";
      lines.push(`${mark} ${c.name} (${c.type}) - ${c.message}${diffText}`);
      if (!c.passed && c.traceback) {
        lines.push(c.traceback);
      }
      if (mode === "test" && !c.passed && c.expected && c.actual) {
        lines.push("  expected: " + JSON.stringify(c.expected));
        lines.push("  actual:   " + JSON.stringify(c.actual));
      }
    }
    return lines.join("\n");
  }

  function setRunning(isRunning) {
    state.isRunning = isRunning;
    el.testBtn.disabled = isRunning;
    el.submitBtn.disabled = isRunning;
    el.feedbackBtn.disabled = isRunning;
    if (isRunning) {
      el.problemSelect.disabled = true;
      el.prevProblemBtn.disabled = true;
      el.nextProblemBtn.disabled = true;
      return;
    }
    el.problemSelect.disabled = state.filteredIds.length === 0;
    updateNavButtons();
  }

  function writeOutput(text) {
    el.resultOutput.textContent = text || "";
  }

  async function safeParseJson(resp) {
    try {
      return await resp.json();
    } catch {
      return { ok: false, error: `HTTP ${resp.status}: non-JSON response` };
    }
  }

  function initEditor() {
    el.plainEditor.style.display = "none";
    el.editorHost.style.display = "block";

    const plainFallback = () => {
      state.usePlainEditor = true;
      state.monacoReady = false;
      el.editorHost.style.display = "none";
      el.plainEditor.style.display = "block";
    };

    if (!(window.require && window.require.config)) {
      plainFallback();
      return;
    }

    try {
      window.require.config({
        paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs" },
      });

      window.require(["vs/editor/editor.main"], function () {
        if (state.usePlainEditor) return;
        state.monacoEditor = window.monaco.editor.create(el.editorHost, {
          value: el.plainEditor.value || "",
          language: "python",
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: "on",
          automaticLayout: true,
          tabSize: 4,
          theme: "vs",
        });
        state.monacoReady = true;
        state.usePlainEditor = false;
        el.plainEditor.style.display = "none";
        el.editorHost.style.display = "block";
      });

      setTimeout(() => {
        if (!state.monacoReady) {
          plainFallback();
        }
      }, 2500);
    } catch (err) {
      plainFallback();
      writeOutput("Monaco 初始化失败，已降级为普通编辑框: " + String(err));
    }
  }

  function getEditorValue() {
    if (!state.usePlainEditor && state.monacoReady && state.monacoEditor) {
      return state.monacoEditor.getValue();
    }
    return el.plainEditor.value;
  }

  function setEditorValue(value) {
    if (!state.usePlainEditor && state.monacoReady && state.monacoEditor) {
      state.monacoEditor.setValue(value || "");
      return;
    }
    el.plainEditor.value = value || "";
  }

  function persistCurrentCode() {
    const pid = state.currentProblemId;
    if (!pid) return;
    localStorage.setItem(STORAGE_PREFIX + pid, getEditorValue());
  }

  function persistCurrentCustomCases() {
    const pid = state.currentProblemId;
    if (!pid) return;
    localStorage.setItem(STORAGE_CUSTOM_PREFIX + pid, el.customCasesInput.value || "");
  }

  function loadProblemCode(problemId) {
    return localStorage.getItem(STORAGE_PREFIX + problemId);
  }

  function loadCustomCases(problemId) {
    return localStorage.getItem(STORAGE_CUSTOM_PREFIX + problemId);
  }

  function loadFavorites() {
    try {
      const raw = localStorage.getItem(STORAGE_FAVS);
      if (!raw) return new Set();
      const arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return new Set();
      return new Set(arr.map(String));
    } catch {
      return new Set();
    }
  }

  function saveFavorites(setObj) {
    localStorage.setItem(STORAGE_FAVS, JSON.stringify(Array.from(setObj)));
  }

  function loadLayoutState() {
    try {
      const raw = localStorage.getItem(STORAGE_LAYOUT);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (typeof parsed?.leftRatio === "number") state.leftRatio = clamp(parsed.leftRatio, 0.2, 0.8);
      if (typeof parsed?.topRatio === "number") state.topRatio = clamp(parsed.topRatio, 0.2, 0.8);
    } catch {
      // ignore invalid cache
    }
  }

  function saveLayoutState() {
    localStorage.setItem(
      STORAGE_LAYOUT,
      JSON.stringify({
        leftRatio: state.leftRatio,
        topRatio: state.topRatio,
      })
    );
  }

  function difficultyClass(diff) {
    if (diff === "简单") return "diff-simple";
    if (diff === "困难") return "diff-hard";
    return "diff-medium";
  }

  function escapeHtml(str) {
    return str
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isNarrowMode() {
    return window.matchMedia("(max-width: 1024px)").matches;
  }

  setActiveTab("problem");
  setFeedbackTab("problem");
  init();
})();
