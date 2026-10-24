// ChurnSense: runs the Python analysis engine (engine.py) in Pyodide, off the page's main thread.
// A module worker (Pyodide needs one).
// Messages in:  {id, fn, args}            -> calls api_call(fn, args) in Python
//               {id, file: name, buffer}  -> writes an uploaded file and reads it (add_file)
// Messages out: {type: 'status', text, fraction}   while starting
//               {type: 'ready'} / {type: 'failed', error}
//               {type: 'progress', fraction, text} while training
//               {id, ok, result | error}

import { loadPyodide } from './pyodide.mjs';

const BASE = new URL('./', import.meta.url).href;
const PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'micropip', 'narwhals', 'packaging'];
const WHEELS = ['plotly-7.1.0-py3-none-any.whl', 'et_xmlfile-2.0.0-py3-none-any.whl',
                'openpyxl-3.1.5-py2.py3-none-any.whl'];

const status = (text, fraction) => self.postMessage({ type: 'status', text, fraction });

async function start() {
  status('Starting Python', 0.05);
  const pyodide = await loadPyodide({ indexURL: BASE });
  status('Loading the data science packages', 0.25);
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });
  status('Loading the chart and Excel packages', 0.6);
  const micropip = pyodide.pyimport('micropip');
  await micropip.install(WHEELS.map((w) => BASE + w));
  status('Starting the analysis engine', 0.75);
  const engine = await (await fetch(BASE + 'engine.py')).text();
  pyodide.runPython(engine);
  pyodide.globals.get('PROGRESS').set('fn', (fraction, text) => self.postMessage({ type: 'progress', fraction, text }));
  pyodide.FS.mkdirTree('/uploads');
  return pyodide;
}

const ready = start().then(
  (pyodide) => { self.postMessage({ type: 'ready' }); return pyodide; },
  (error) => { self.postMessage({ type: 'failed', error: String(error && error.message || error) }); throw error; },
);

self.onmessage = async (event) => {
  const msg = event.data;
  let pyodide;
  try {
    pyodide = await ready;
  } catch (error) {
    self.postMessage({ id: msg.id, ok: false, error: 'The analysis engine could not start.' });
    return;
  }
  const apiCall = pyodide.globals.get('api_call');
  let reply;
  try {
    if (msg.file) {
      const safe = msg.file.replace(/[\\/]/g, '_');
      const path = '/uploads/' + safe;
      pyodide.FS.writeFile(path, new Uint8Array(msg.buffer));
      reply = apiCall('add_file', JSON.stringify({ name: msg.file, path }));
    } else {
      reply = apiCall(msg.fn, JSON.stringify(msg.args || {}));
    }
    self.postMessage({ id: msg.id, ...JSON.parse(reply) });
  } catch (error) {
    self.postMessage({ id: msg.id, ok: false, error: 'Something went wrong: ' + String(error && error.message || error) });
  } finally {
    apiCall.destroy();
  }
};
