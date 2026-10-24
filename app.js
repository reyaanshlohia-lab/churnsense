'use strict';
// ChurnSense: the page. The Python engine runs in worker.js (Pyodide); this file handles the page.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const S = {
  defaults: null, files: [], opts: null, setup: null, classes: [], trained: null, version: 0, views: {},
  busy: false, rankCsv: null, sampleNames: new Set(),
};
const KEEP = ['preset', 'test_size', 'seed', 'scoring', 'models', 'balanced', 'logistic_max_iter', 'mlp_max_iter',
              'search', 'threshold', 'metric', 'cutoffs', 'k', 'kmax', 'missing', 'exclude'];
const SAMPLE_WORDS = [
  ['employees.csv', /\b(employees?|staff|attrition)\b/i], ['customers.csv', /\b(customers?|clients?|churn)/i],
  ['students_SYNTHETIC.csv', /\b(students?|pupils?|drop ?outs?)\b/i], ['teachers_SYNTHETIC.csv', /\bteachers?\b/i],
  ['custom_join_SYNTHETIC.csv', /\b(applicants?|enrol)/i],
];
const AI_SYSTEM = 'You are the assistant inside ChurnSense, a tool that predicts which people are likely to ' +
  'leave or join an organisation and explains why. Answer the user\'s question using only the results summary you ' +
  'are given. If the summary does not contain the answer, say so and suggest what to ask the tool instead (for ' +
  'example "explain employee 1043", "show the segments", "compare the models"). Be concise and concrete; use short ' +
  'bullet points where helpful. Never present predictions as certainties, and treat suggested actions as things to ' +
  'discuss with people, not automatic decisions. Do not invent numbers.';

// ---------------- the Python engine (worker) ----------------
let worker;
let nextId = 1;
const pending = new Map();

function startEngine() {
  if (location.protocol === 'file:') {
    engineFailed('This page must be opened from a web address, not as a file. Put the folder on a free host ' +
      '(see README) or run "python -m http.server" in the folder and open http://localhost:8000.');
    return;
  }
  worker = new Worker('worker.js', { type: 'module' });
  worker.onerror = (e) => engineFailed(e.message || 'The analysis engine could not start.');
  worker.onmessage = (event) => {
    const m = event.data;
    if (m.type === 'status') return engineStatus(m.text, m.fraction);
    if (m.type === 'ready') return engineDone();
    if (m.type === 'failed') return engineFailed(m.error);
    if (m.type === 'progress') return busyProgress(m.fraction, m.text);
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.ok) p.resolve(m.result); else p.reject(new Error(m.error));
  };
}

function py(fn, args = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, fn, args });
  });
}

function pyFile(name, buffer) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, file: name, buffer }, [buffer]);
  });
}

function engineStatus(text, fraction) {
  $('#engine-text').textContent = text + '… (the first visit downloads about 55 MB; later visits are faster)';
  $('#engine-bar').style.width = Math.round(100 * fraction) + '%';
}

async function engineDone() {
  $('#engine-bar').style.width = '100%';
  S.defaults = await py('defaults');
  $('#engine').classList.add('done');
  fillStaticOptions();
}

function engineFailed(message) {
  const box = $('#engine');
  box.classList.remove('done');
  box.classList.add('failed');
  $('#engine-text').textContent = 'The analysis engine could not start: ' + message;
}

// ---------------- small helpers ----------------
function toast(message, kind = 'error') {
  const t = $('#toast');
  t.textContent = message;
  t.className = 'toast' + (kind === 'info' ? ' info' : '');
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, kind === 'info' ? 3500 : 8000);
}

async function busy(title, fn) {
  if (S.busy) { toast('Please wait: the current task is still running.', 'info'); return undefined; }
  S.busy = true;
  $('#busy-title').textContent = title;
  $('#busy-text').textContent = S.defaults ? '' : 'Waiting for the analysis engine to start…';
  $('#busy-bar').style.width = '0%';
  $('#busy').hidden = false;
  try {
    return await fn();
  } catch (error) {
    toast(error.message || String(error));
    return undefined;
  } finally {
    S.busy = false;
    $('#busy').hidden = true;
  }
}

function busyProgress(fraction, text) {
  $('#busy-text').textContent = text;
  $('#busy-bar').style.width = Math.round(100 * fraction) + '%';
}

function download(name, content, type = 'text/plain') {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function fmt(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : (+v.toFixed(4)).toString();
  return String(v);
}

function sizeText(bytes) {
  return bytes > 1e6 ? (bytes / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(bytes / 1e3)) + ' KB';
}

// ---------------- rendering: charts, tables, blocks ----------------
function plot(div, figure) {
  if (!window.Plotly || !figure) return;
  const layout = Object.assign({}, figure.layout, { autosize: true });
  delete layout.width;
  window.Plotly.newPlot(div, figure.data || [], layout,
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
}

function renderTable(container, table, opts = {}) {
  container.replaceChildren();
  if (!table || !table.columns) return;
  const rows = table.rows.slice(0, opts.maxRows || 1000);
  const tbl = el('table');
  const head = el('tr');
  let sortCol = -1;
  let asc = true;
  const body = el('tbody');
  const draw = () => {
    body.replaceChildren(...rows.map((r) => el('tr', {}, ...r.map((v) => el('td', { class: typeof v === 'number' ? 'num' : '', text: fmt(v) })))));
  };
  table.columns.forEach((c, i) => {
    head.append(el('th', {
      text: c, scope: 'col', title: 'Click to sort',
      onclick: () => {
        asc = sortCol === i ? !asc : true;
        sortCol = i;
        rows.sort((a, b) => {
          const x = a[i]; const y = b[i];
          if (x === y) return 0;
          if (x === null) return 1;
          if (y === null) return -1;
          return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * (asc ? 1 : -1);
        });
        draw();
      },
    }));
  });
  tbl.append(el('thead', {}, head), body);
  draw();
  container.append(tbl);
  if (table.rows.length > rows.length) container.append(el('p', { class: 'hint', text: `Showing ${rows.length} of ${table.rows.length} rows.` }));
}

function renderBlocks(container, blocks, append = false) {
  if (!append) container.replaceChildren();
  for (const b of blocks || []) {
    if (!b) continue;
    const box = el('div', { class: 'block' });
    if (b.title) box.append(el('h4', { text: b.title }));
    if (b.figure) {
      const chart = el('div', { class: 'chart' });
      box.append(chart);
      container.append(box);
      plot(chart, b.figure);
    } else {
      container.append(box);
    }
    if (b.html) {
      const text = el('div', { class: 'analysis' });
      text.innerHTML = b.html;    // produced by the engine's Markdown converter, which escapes everything
      box.append(text);
    }
    if (b.table) {
      const wrap = el('div', { class: 'table-wrap' });
      box.append(wrap);
      renderTable(wrap, b.table);
    }
  }
}

function setHTML(sel, html) { $(sel).innerHTML = html || ''; }

// ---------------- files ----------------
function renderFileList() {
  const list = $('#file-list');
  list.replaceChildren();
  if (!S.files.length) {
    list.append(el('li', { class: 'empty', text: 'No files yet. Add a spreadsheet below, or try a sample.' }));
    return;
  }
  for (const f of S.files) {
    const selected = S.opts && S.opts.file === f.name;
    const detail = f.error ? f.error
      : f.info ? `${f.info.rows.toLocaleString()} rows × ${f.info.columns} columns · ${sizeText(f.size)}` +
                 (f.info.synthetic ? ' · synthetic demo data' : '')
        : `${sizeText(f.size)} · reading…`;
    const radio = el('input', { type: 'radio', name: 'datafile', value: f.name, checked: selected, disabled: !f.info,
                                'aria-label': `Analyse ${f.name}`, onchange: () => selectFile(f.name) });
    list.append(el('li', { class: 'file' + (selected ? ' selected' : '') },
      el('span', { class: 'icon', 'aria-hidden': 'true', text: f.name.toLowerCase().endsWith('.xlsx') ? '📗' : '📄' }),
      el('div', { class: 'meta' }, el('div', { class: 'name', text: f.name }),
        el('div', { class: 'detail' + (f.error ? ' error' : ''), text: detail })),
      el('label', { class: 'use' }, radio, document.createTextNode('Analyse')),
      el('button', { type: 'button', class: 'icon-btn', title: `Remove ${f.name}`, 'aria-label': `Remove ${f.name}`,
                     text: '✕', onclick: () => removeFile(f.name) })));
  }
}

async function addFiles(list) {
  const files = [...list].filter((f) => {
    const ok = /\.(csv|xlsx)$/i.test(f.name);
    if (!ok) toast(`${f.name}: only .csv and .xlsx files can be used.`);
    return ok;
  });
  for (const file of files) {
    S.files = S.files.filter((f) => f.name !== file.name);
    const entry = { name: file.name, size: file.size, info: null, error: null };
    S.files.push(entry);
    renderFileList();
    try {
      const buffer = await file.arrayBuffer();
      entry.info = await pyFile(file.name, buffer);
    } catch (error) {
      entry.error = error.message;
    }
    renderFileList();
    refreshFileChoices();
    if (entry.info && (!S.opts || !S.files.some((f) => f.name === S.opts.file))) await selectFile(entry.name);
  }
}

async function addSample(name) {
  const res = await fetch(name);
  if (!res.ok) { toast(`Could not load the sample ${name}.`); return; }
  const blob = await res.blob();
  S.sampleNames.add(name);
  await addFiles([new File([blob], name, { type: 'text/csv' })]);
}

async function removeFile(name) {
  await py('remove_file', { name });
  S.files = S.files.filter((f) => f.name !== name);
  if (S.opts && S.opts.file === name) {
    S.opts = null;
    S.setup = null;
    clearTrained();
    renderSetup();
  }
  renderFileList();
  refreshFileChoices();
}

async function selectFile(name, scenario, mode) {
  await busy(`Reading ${name}`, async () => {
    const setup = await py('select_data', { name, scenario, mode });
    const keep = S.opts ? Object.fromEntries(KEEP.filter((k) => k in S.opts).map((k) => [k, S.opts[k]])) : {};
    S.setup = setup;
    S.classes = setup.classes;
    S.opts = { ...optionsFrom(setup), ...keep };
    clearTrained();
    renderSetup();
    renderTrainSettings();
    renderFileList();
  });
}

function optionsFrom(setup) {
  const d = S.defaults.defaults;
  return {
    file: setup.file, scenario: setup.scenario, mode: setup.mode, wording: { ...setup.wording }, target: setup.target,
    positive: setup.positive, drop: [...setup.drop], features: [...setup.features], sensitive: [...setup.sensitive],
    exclude: d.exclude_sensitive, missing: d.missing_strategy, rules: setup.rules, preset: S.defaults.preset,
    test_size: d.test_size, seed: d.random_state, scoring: d.scoring, models: [...S.defaults.models],
    balanced: d.balanced, logistic_max_iter: d.logistic_max_iter, mlp_max_iter: d.mlp_max_iter,
    search: JSON.parse(JSON.stringify(S.defaults.search)), threshold: d.threshold, metric: 'ROC-AUC',
    cutoffs: [...d.level_cutoffs], k: S.defaults.clustering.n_clusters, kmax: S.defaults.clustering.max_k,
  };
}

function refreshFileChoices() {
  const select = $('#rank-source');
  const current = select.value;
  select.replaceChildren(el('option', { value: '', text: 'The test set' }),
    ...S.files.filter((f) => f.info).map((f) => el('option', { value: f.name, text: f.name })));
  select.value = [...select.options].some((o) => o.value === current) ? current : '';
}

// ---------------- setup tab ----------------
function fillSelect(select, values, value, labels) {
  select.replaceChildren(...values.map((v, i) => el('option', { value: v, text: labels ? labels[i] : v })));
  if (value !== undefined && value !== null) select.value = value;
}

function checklist(container, values, checked, onchange) {
  container.replaceChildren(...values.map((v) => el('label', {},
    el('input', { type: 'checkbox', value: v, checked: checked.includes(v), onchange }), document.createTextNode(v))));
}

const checkedValues = (container) => $$('input:checked', container).map((i) => i.value);

function fillStaticOptions() {
  const d = S.defaults;
  fillSelect($('#o-scenario'), Object.keys(d.scenarios), undefined, Object.values(d.scenarios));
  fillSelect($('#o-missing'), d.missing, d.defaults.missing_strategy);
  fillSelect($('#o-preset'), d.presets, d.preset);
  fillSelect($('#o-scoring'), d.scoring, d.defaults.scoring);
  checklist($('#o-models'), d.models, d.models, () => { if (S.opts) S.opts.models = checkedValues($('#o-models')); });
}

function renderSetup() {
  const has = !!(S.opts && S.setup);
  $('#setup-body').hidden = !has;
  $('#setup-empty').hidden = has;
  if (!has) return;
  const o = S.opts;
  const s = S.setup;
  setHTML('#setup-status', s.status_html);
  setHTML('#setup-info', s.info_html);
  $('#o-scenario').value = o.scenario;
  $('#o-mode').value = o.mode;
  $('#mode-label').hidden = o.scenario !== 'Custom';
  $('#wording').hidden = o.scenario !== 'Custom';
  for (const k of ['person', 'people', 'org', 'outcome_past', 'outcome_ing', 'stay_past']) $('#w-' + k).value = (o.wording || {})[k] || '';
  $('#o-missing').value = o.missing;
  fillSelect($('#o-target'), s.columns, o.target);
  fillSelect($('#o-positive'), S.classes, o.positive);
  checklist($('#o-drop'), s.columns.filter((c) => c !== o.target), o.drop, () => {
    o.drop = checkedValues($('#o-drop'));
    o.features = s.columns.filter((c) => c !== o.target && !o.drop.includes(c));
    renderSetup();
  });
  checklist($('#o-features'), s.columns.filter((c) => c !== o.target && !o.drop.includes(c)), o.features,
    () => { o.features = checkedValues($('#o-features')); });
  checklist($('#o-sensitive'), s.columns.filter((c) => c !== o.target), o.sensitive,
    () => { o.sensitive = checkedValues($('#o-sensitive')); });
  $('#o-exclude').checked = !!o.exclude;
  renderTable($('#preview'), s.preview);
  renderTable($('#column-info'), s.column_info);
  renderRules();
  fillSelect($('#x-x'), ['(none)', ...s.columns], s.numeric[0] || '(none)');
  fillSelect($('#x-y'), ['(none)', ...s.columns], '(none)');
  fillSelect($('#x-color'), ['(none)', ...s.columns], o.target);
  fillSelect($('#rate-col'), s.columns.filter((c) => c !== o.target && !o.drop.includes(c)), o.features[0]);
  fillSelect($('#reg-target'), s.numeric, s.regression.target);
  checklist($('#reg-features'), s.numeric, s.regression.features || [], () => {});
}

function renderRules() {
  const o = S.opts;
  const rules = o.rules || { columns: ['Feature', 'Condition', 'Value', 'Suggestion'], rows: [] };
  o.rules = rules;
  const list = 'rule-features';
  const tbl = el('table', { class: 'rules' });
  tbl.append(el('thead', {}, el('tr', {}, ...rules.columns.map((c) => el('th', { text: c })), el('th', { text: '' }))));
  const body = el('tbody');
  rules.rows.forEach((row, r) => {
    const cells = rules.columns.map((c, i) => {
      if (c === 'Condition') {
        const select = el('select', { 'aria-label': `Rule ${r + 1} condition`, onchange: (e) => { row[i] = e.target.value; } });
        fillSelect(select, S.defaults.conditions, row[i] || '==');
        return el('td', {}, select);
      }
      return el('td', {}, el('input', { value: row[i] ?? '', list: c === 'Feature' ? list : null,
                                        'aria-label': `Rule ${r + 1} ${c}`, oninput: (e) => { row[i] = e.target.value; } }));
    });
    body.append(el('tr', {}, ...cells, el('td', {}, el('button', {
      type: 'button', class: 'icon-btn', text: '✕', title: 'Delete this rule', 'aria-label': `Delete rule ${r + 1}`,
      onclick: () => { rules.rows.splice(r, 1); renderRules(); },
    }))));
  });
  tbl.append(body);
  const datalist = el('datalist', { id: list }, ...['*', ...S.setup.columns].map((c) => el('option', { value: c })));
  $('#rules').replaceChildren(tbl, datalist);
}

// ---------------- train settings ----------------
function renderTrainSettings() {
  const o = S.opts;
  if (!o) return;
  $('#o-preset').value = o.preset;
  $('#o-test_size').value = o.test_size;
  $('#o-seed').value = o.seed;
  $('#o-scoring').value = o.scoring;
  $$('input', $('#o-models')).forEach((i) => { i.checked = o.models.includes(i.value); });
  $('#o-balanced').checked = !!o.balanced;
  $('#o-logistic_max_iter').value = o.logistic_max_iter;
  $('#o-mlp_max_iter').value = o.mlp_max_iter;
  $('#o-threshold').value = o.threshold;
  $('#thr-value').textContent = (+o.threshold).toFixed(2);
  $('#o-metric').value = o.metric;
  $('#o-k').value = o.k;
  $('#o-kmax').value = o.kmax;
  $('#cut-low').value = o.cutoffs[0];
  $('#cut-high').value = o.cutoffs[1];
  const boxes = $('#search-boxes');
  boxes.replaceChildren();
  for (const [model, spec] of Object.entries(o.search)) {
    const fields = [
      el('label', {}, 'CV folds', el('input', { type: 'number', min: 2, step: 1, value: spec.cv, oninput: (e) => { spec.cv = +e.target.value; } })),
      el('label', {}, 'Search iterations (n_iter)', el('input', { type: 'number', min: 1, step: 1, value: spec.n_iter, oninput: (e) => { spec.n_iter = +e.target.value; } })),
      ...Object.entries(spec.grid).map(([param, text]) => el('label', {}, `${param} (comma-separated)`,
        el('input', { value: text, oninput: (e) => { spec.grid[param] = e.target.value; } }))),
    ];
    boxes.append(el('fieldset', {}, el('legend', { text: model }), el('div', { class: 'grid4' }, ...fields)));
  }
}

function bindOptionInputs() {
  const num = (id, key) => $(id).addEventListener('change', (e) => { if (S.opts) S.opts[key] = +e.target.value; });
  num('#o-test_size', 'test_size');
  num('#o-seed', 'seed');
  num('#o-logistic_max_iter', 'logistic_max_iter');
  num('#o-mlp_max_iter', 'mlp_max_iter');
  num('#o-k', 'k');
  num('#o-kmax', 'kmax');
  $('#o-scoring').addEventListener('change', (e) => { if (S.opts) S.opts.scoring = e.target.value; });
  $('#o-balanced').addEventListener('change', (e) => { if (S.opts) S.opts.balanced = e.target.checked; });
  $('#o-exclude').addEventListener('change', (e) => { if (S.opts) S.opts.exclude = e.target.checked; });
  $('#o-missing').addEventListener('change', (e) => { if (S.opts) S.opts.missing = e.target.value; });
  $('#o-positive').addEventListener('change', (e) => { if (S.opts) S.opts.positive = e.target.value; });
  for (const k of ['person', 'people', 'org', 'outcome_past', 'outcome_ing', 'stay_past']) {
    $('#w-' + k).addEventListener('change', (e) => { if (S.opts) S.opts.wording[k] = e.target.value; });
  }
  $('#o-preset').addEventListener('change', async (e) => {
    if (!S.opts) return;
    S.opts.preset = e.target.value;
    S.opts.search = await py('preset_values', { preset: e.target.value });
    renderTrainSettings();
  });
  $('#o-target').addEventListener('change', async (e) => {
    const o = S.opts;
    o.target = e.target.value;
    o.drop = o.drop.filter((c) => c !== o.target);
    o.sensitive = o.sensitive.filter((c) => c !== o.target);
    const info = await py('target_changed', { target: o.target, drop: o.drop });
    S.classes = info.classes;
    o.positive = info.positive;
    o.features = info.features;
    renderSetup();
  });
  $('#o-scenario').addEventListener('change', (e) => selectFile(S.opts.file, e.target.value));
  $('#o-mode').addEventListener('change', async (e) => {
    const info = await py('custom_mode', { name: S.opts.file, mode: e.target.value });
    S.opts.mode = e.target.value;
    S.opts.wording = info.wording;
    S.opts.rules = info.rules;
    renderSetup();
  });
  $('#o-threshold').addEventListener('input', (e) => { $('#thr-value').textContent = (+e.target.value).toFixed(2); });
  $('#o-threshold').addEventListener('change', (e) => {
    S.opts.threshold = +e.target.value;
    S.views = {};
    showView(currentTab());
  });
  $('#o-metric').addEventListener('change', (e) => { S.opts.metric = e.target.value; S.views = {}; showView('compare'); });
  for (const id of ['#cut-low', '#cut-high']) {
    $(id).addEventListener('change', () => { S.opts.cutoffs = [+$('#cut-low').value, +$('#cut-high').value]; });
  }
}

// ---------------- training and the tabs ----------------
function clearTrained() {
  S.trained = null;
  S.views = {};
  S.version++;
  $('#train-body').hidden = true;
  $('#train-status').replaceChildren();
  $$('.model-only').forEach((x) => { x.hidden = true; });
  $$('.needs-model').forEach((x) => { x.hidden = false; });
  $('#why-tab').textContent = '5. Why They Leave';
}

async function trainNow() {
  if (!S.opts) { toast('Add a spreadsheet and select it first.'); return; }
  const info = await busy('Training the models', () => py('train', { options: S.opts }));
  if (info) afterTrained(info);
}

function afterTrained(info) {
  S.trained = info;
  S.views = {};
  S.version++;
  setHTML('#train-status', info.status_html);
  $('#train-body').hidden = false;
  fillSelect($('#model-choice'), info.models, info.best);
  $('#why-tab').textContent = '5. ' + info.why_label;
  $$('.model-only').forEach((x) => { x.hidden = false; });
  $$('.needs-model').forEach((x) => { x.hidden = true; });
  setHTML('#predict-header', info.predict_header_html);
  personForm($('#person-form'), info.person);
  $('#person-result').hidden = true;
  $('#rank-table').replaceChildren();
  $('#rank-notes').replaceChildren();
  $('#rank-csv').hidden = true;
  checklist($('#seg-columns'), info.numeric, info.cluster_columns, () => {});
  fillSelect($('#seg-x'), info.cluster_columns, info.cluster_columns[0]);
  fillSelect($('#seg-y'), info.cluster_columns, info.cluster_columns[Math.min(1, info.cluster_columns.length - 1)]);
  refreshFileChoices();
  showView(currentTab());
}

function currentTab() {
  return ($('.tabs [aria-selected=true]') || {}).dataset?.tab || 'setup';
}

function selectTab(name) {
  $$('.tabs [role=tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  $$('.panel').forEach((p) => { p.hidden = p.dataset.panel !== name; });
  showView(name);
  $$('.chart', $(`.panel[data-panel="${name}"]`)).forEach((c) => { if (c.data && window.Plotly) window.Plotly.Plots.resize(c); });
}

async function showView(name) {
  if (!S.trained || !S.opts) return;
  const o = S.opts;
  const key = `${name}|${S.version}|${o.threshold}|${o.metric}|${$('#model-choice').value}`;
  if (S.views[name] === key) return;
  if (name === 'train') {
    const v = await busy('Drawing the model results', () => py('model_view', { name: $('#model-choice').value, threshold: o.threshold }));
    if (!v) return;
    setHTML('#model-params', v.params_html);
    renderTable($('#model-metrics'), v.metrics);
    renderBlocks($('#model-blocks'), v.blocks);
  } else if (name === 'compare') {
    const v = await busy('Comparing the models', () => py('compare_view', { metric: o.metric, threshold: o.threshold }));
    if (!v) return;
    renderTable($('#compare-table'), v.table);
    renderBlocks($('#compare-blocks'), v.blocks);
  } else if (name === 'why') {
    const v = await busy('Explaining the best model', async () => {
      const why = await py('why_view');
      const seg = await py('segments_view', segmentArgs());
      return { why, seg };
    });
    if (!v) return;
    setHTML('#why-title', v.why.title_html);
    setHTML('#why-note', v.why.note_html);
    renderBlocks($('#why-blocks'), v.why.blocks);
    showSegments(v.seg);
  } else if (name === 'fairness') {
    const v = await busy('Checking fairness', () => py('fairness_view', { threshold: o.threshold }));
    if (!v) return;
    renderBlocks($('#fair-blocks'), [v.block]);
    renderTable($('#fair-table'), v.table);
  } else {
    return;
  }
  S.views[name] = key;
}

function segmentArgs() {
  const o = S.opts;
  return { columns: checkedValues($('#seg-columns')), k: o.k, kmax: o.kmax, x: $('#seg-x').value, y: $('#seg-y').value,
           threshold: o.threshold, rules: o.rules };
}

function showSegments(seg) {
  S.segBlocks = seg.blocks;
  renderBlocks($('#seg-blocks'), seg.blocks);
  renderTable($('#seg-table'), seg.table);
  setHTML('#summary', seg.summary_html);
  fillSelect($('#seg-x'), seg.columns, seg.x);
  fillSelect($('#seg-y'), seg.columns, seg.y);
}

// ---------------- predict & explain ----------------
function personForm(container, table) {
  container.replaceChildren();
  if (!table) return;
  table.columns.forEach((c, i) => {
    container.append(el('label', {}, c, el('input', { 'data-col': c, value: fmt(table.rows[0][i]).replace(/,/g, '') })));
  });
}

function readForm(container) {
  const numeric = new Set((S.trained && S.trained.numeric) || []);
  const columns = [];
  const values = [];
  $$('input', container).forEach((i) => {
    columns.push(i.dataset.col);
    const v = i.value.trim();
    values.push(numeric.has(i.dataset.col) && v !== '' && !Number.isNaN(+v) ? +v : v);
  });
  return { columns, rows: [values] };
}

async function explainNow() {
  const row = readForm($('#person-form'));
  const r = await busy('Explaining this person', () => py('explain', { row, cutoffs: S.opts.cutoffs, rules: S.opts.rules }));
  if (!r) return;
  $('#person-result').hidden = false;
  const header = $('#person-header');
  header.innerHTML = r.header_html;
  header.style.background = r.colour;
  setHTML('#person-text', r.html);
  renderBlocks($('#person-blocks'), [{ title: 'Top reasons', figure: r.figure, html: '', table: null },
                                      { title: 'Reasons table', figure: null, html: '', table: r.reasons }]);
  personForm($('#whatif-form'), row);
  $('#whatif-result').replaceChildren();
}

// ---------------- prompt ----------------
async function runPrompt() {
  const text = $('#prompt').value.trim();
  if (!text) { toast('Type a request first, or click one of the examples.', 'info'); return; }
  if (!S.defaults) { toast('The analysis engine is still starting; try again in a moment.', 'info'); return; }
  if (!S.files.some((f) => f.info) && !/^\s*(help|\?)/i.test(text)) {
    const match = SAMPLE_WORDS.find(([, re]) => re.test(text));
    if (match) {
      toast(`No file yet, so I loaded the ${match[0]} sample.`, 'info');
      await addSample(match[0]);
    }
  }
  const result = await busy('Working on your request', () => py('run_prompt', { prompt: text, options: S.opts, ai: !!$('#ai-key').value.trim() }));
  if (!result) return;
  showAnswer(result);
  if (result.options) {
    const fileChanged = !S.opts || result.options.file !== S.opts.file;
    S.opts = result.options;
    if (fileChanged || !S.setup) S.setup = await py('setup_info');
    S.classes = (await py('target_changed', { target: S.opts.target, drop: S.opts.drop })).classes;
    renderSetup();
    renderTrainSettings();
    renderFileList();
  }
  if (result.retrained) afterTrained(await py('trained_info'));
  else { S.views = {}; showView(currentTab()); }
  const key = $('#ai-key').value.trim();
  if (key && !/^\s*(help|\?)\s*$/i.test(text)) askAI(text, key);
}

function showAnswer(r) {
  $('#answer').hidden = false;
  setHTML('#answer-body', r.html);
  $('#ai-answer').hidden = true;
  renderBlocks($('#answer-blocks'), r.blocks);
  const dl = $('#answer-downloads');
  dl.replaceChildren(...(r.downloads || []).map((d) => el('button', {
    type: 'button', class: 'secondary', text: d.label || `Download ${d.name}`,
    onclick: () => download(d.name, d.text, d.type || 'text/csv'),
  })));
  $('#answer').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function askAI(question, key) {
  const box = $('#ai-answer');
  const label = '<div class="label">AI answer (Gemini)</div>';
  box.hidden = false;
  box.innerHTML = label + '<p>Thinking…</p>';
  try {
    const context = await py('ai_context');
    const model = $('#ai-model').value.trim() || 'gemini-flash-latest';
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: AI_SYSTEM }] },
        contents: [{ role: 'user', parts: [{ text: `Results summary from ChurnSense:\n${context}\n\nQuestion: ${question}` }] }],
        generationConfig: { temperature: 0.3 },
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data.error && data.error.message) || `HTTP ${res.status}`);
    const text = ((data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [])
      .map((p) => p.text || '').join('').trim();
    if (!text) throw new Error('Gemini returned no answer (it may have been blocked or the quota used up).');
    const html = (await py('md_html', { text })).html;
    box.innerHTML = label + html + '<p class="hint">AI answers can be wrong: check them against the charts.</p>';
  } catch (error) {
    box.replaceChildren();
    box.innerHTML = label;
    box.append(el('p', { text: 'Could not get an AI answer: ' + error.message }));
  }
}

function loadAIKey() {
  try {
    const key = localStorage.getItem('churnsense-gemini-key');
    if (key) { $('#ai-key').value = key; $('#ai-remember').checked = true; }
    const model = localStorage.getItem('churnsense-gemini-model');
    if (model) $('#ai-model').value = model;
  } catch (e) { /* storage unavailable: the key is only kept for this visit */ }
}

function saveAIKey() {
  try {
    if ($('#ai-remember').checked) {
      localStorage.setItem('churnsense-gemini-key', $('#ai-key').value.trim());
      localStorage.setItem('churnsense-gemini-model', $('#ai-model').value.trim());
    } else {
      localStorage.removeItem('churnsense-gemini-key');
      localStorage.removeItem('churnsense-gemini-model');
    }
  } catch (e) { /* storage unavailable */ }
}

// ---------------- downloads ----------------
function reportArgs() {
  const o = S.opts;
  return { threshold: o.threshold, metric: o.metric, rules: o.rules, columns: checkedValues($('#seg-columns')), k: o.k,
           reg_target: $('#reg-target').value || null, reg_features: checkedValues($('#reg-features')) };
}

async function downloadCharts() {
  await busy('Making the chart images', async () => {
    const list = await py('chart_list', reportArgs());
    const holder = el('div', { style: 'position:fixed;left:-10000px;top:0;width:1100px' });
    document.body.append(holder);
    const names = [];
    const data = [];
    try {
      for (const [i, c] of list.charts.entries()) {
        busyProgress(i / list.charts.length, `Chart ${i + 1} of ${list.charts.length}`);
        const div = el('div');
        holder.append(div);
        const height = (c.figure.layout && c.figure.layout.height) || 500;
        await window.Plotly.newPlot(div, c.figure.data, Object.assign({}, c.figure.layout, { width: 1100, height }), { staticPlot: true });
        const url = await window.Plotly.toImage(div, { format: 'png', width: 1100, height });
        names.push(c.name + '.png');
        data.push(url.split(',')[1]);
        window.Plotly.purge(div);
        div.remove();
      }
    } finally {
      holder.remove();
    }
    const zip = await py('make_zip', { names, data });
    const bytes = Uint8Array.from(atob(zip.b64), (ch) => ch.charCodeAt(0));
    download(list.zip_name, new Blob([bytes], { type: 'application/zip' }));
  });
}

// ---------------- wiring ----------------
function wire() {
  const zone = $('#dropzone');
  const input = $('#file-input');
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (e) => { if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
  // a file dropped elsewhere on the page should not open it in the browser tab
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => { e.preventDefault(); if (e.target.closest && !e.target.closest('#dropzone') && e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
  $$('[data-sample]').forEach((b) => b.addEventListener('click', () => addSample(b.dataset.sample)));
  $$('[data-example]').forEach((b) => b.addEventListener('click', () => { $('#prompt').value = b.dataset.example; $('#prompt').focus(); }));
  $('#run').addEventListener('click', runPrompt);
  $('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runPrompt(); } });
  $('#ai-key').addEventListener('change', saveAIKey);
  $('#ai-model').addEventListener('change', saveAIKey);
  $('#ai-remember').addEventListener('change', saveAIKey);
  $$('.tabs [role=tab]').forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));
  $('.tabs').addEventListener('keydown', (e) => {
    const tabs = $$('.tabs [role=tab]');
    const i = tabs.indexOf(document.activeElement);
    if (i < 0 || !['ArrowRight', 'ArrowLeft'].includes(e.key)) return;
    const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    selectTab(next.dataset.tab);
  });
  $('#apply').addEventListener('click', trainNow);
  $('#train-run').addEventListener('click', trainNow);
  $('#rule-add').addEventListener('click', () => { S.opts.rules.rows.push(['', '==', '', '']); renderRules(); });
  $('#model-choice').addEventListener('change', () => showView('train'));
  $('#explore-run').addEventListener('click', async () => {
    const v = await busy('Drawing the chart', () => py('explore', { kind: $('#x-kind').value, x: $('#x-x').value, y: $('#x-y').value, color: $('#x-color').value }));
    if (v) renderBlocks($('#explore-out'), [v]);
  });
  $('#rate-run').addEventListener('click', async () => {
    const v = await busy('Drawing the chart', () => py('outcome_rate', { column: $('#rate-col').value }));
    if (v) renderBlocks($('#rate-out'), [v]);
  });
  $('#results-csv').addEventListener('click', async () => {
    const v = await busy('Preparing the table', () => py('results_csv', { threshold: S.opts.threshold }));
    if (v) download(v.name, v.text, 'text/csv');
  });
  $('#seg-run').addEventListener('click', async () => {
    const v = await busy('Finding the segments', () => py('segments_view', segmentArgs()));
    if (v) showSegments(v);
  });
  for (const id of ['#seg-x', '#seg-y']) {
    $(id).addEventListener('change', async () => {
      const v = await busy('Drawing the segments', () => py('cluster_axes', { x: $('#seg-x').value, y: $('#seg-y').value }));
      if (v && S.segBlocks) { S.segBlocks[1] = v; renderBlocks($('#seg-blocks'), S.segBlocks); }
    });
  }
  $('#report-btn').addEventListener('click', async () => {
    const v = await busy('Writing the report', () => py('report', reportArgs()));
    if (v) download(v.name, v.html, 'text/html');
  });
  $('#charts-btn').addEventListener('click', downloadCharts);
  $('#explain-run').addEventListener('click', explainNow);
  $('#whatif-run').addEventListener('click', async () => {
    const v = await busy('Running the what-if', () => py('whatif', { row: readForm($('#whatif-form')), cutoffs: S.opts.cutoffs }));
    if (v) setHTML('#whatif-result', v.html);
  });
  $('#person-load').addEventListener('click', async () => {
    const v = await busy('Loading the person', () => py('load_person', { key: $('#person-id').value }));
    if (v) { personForm($('#person-form'), v.row); $('#person-result').hidden = true; }
  });
  $('#rank-run').addEventListener('click', async () => {
    const v = await busy('Ranking', () => py('rank', { cutoffs: S.opts.cutoffs, file: $('#rank-source').value || null }));
    if (!v) return;
    setHTML('#rank-notes', `<p class="hint">${v.total.toLocaleString()} people from ${v.source === 'the test set' ? 'the test set' : 'the uploaded file'}.</p>` + v.notes_html);
    renderTable($('#rank-table'), v.table);
    S.rankCsv = v;
    $('#rank-csv').hidden = false;
  });
  $('#rank-csv').addEventListener('click', () => { if (S.rankCsv) download(S.rankCsv.name, S.rankCsv.csv, 'text/csv'); });
  $('#reg-run').addEventListener('click', async () => {
    const v = await busy('Running the regression', () => py('regression', {
      target: $('#reg-target').value, features: checkedValues($('#reg-features')),
      test_size: S.opts ? S.opts.test_size : 0.2, seed: S.opts ? S.opts.seed : 42,
    }));
    if (!v) return;
    renderBlocks($('#reg-blocks'), [v.block]);
    renderTable($('#reg-table'), v.coefficients);
  });
  bindOptionInputs();
}

wire();
loadAIKey();
startEngine();
