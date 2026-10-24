// ChurnSense Help: a chat that explains the charts on the page (every line, colour and star), the words used,
// and how to use the app. It runs in the browser with no AI service: it reads the charts that are on the page
// and answers from the knowledge base below. With a Gemini key (AI answers), questions it cannot match go to Gemini.
(() => {
  const H = { chart: null, open: false };
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  // escaped text with **bold** and line breaks
  const rich = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>');

  // ---------------- wording that fits the current analysis ----------------
  function words() {
    const state = typeof S !== 'undefined' ? S : null;   // app.js's state (a top-level const, not a window property)
    const k = (state && state.trained && state.trained.kpis) || {};
    const leave = !(state && state.trained && state.trained.leave === false);
    return { P: k.people || 'people', O: k.outcome || (leave ? 'left' : 'joined'), leave };
  }

  // ---------------- the charts on the page ----------------
  function pageCharts() {
    return [...document.querySelectorAll('#app .block')]
      .filter((b) => b.querySelector('.chart') && b.offsetParent !== null)
      .map((b, i) => {
        const gd = b.querySelector('.chart');
        const h4 = b.querySelector('h4');
        const plotTitle = ((gd.layout && gd.layout.title && (gd.layout.title.text || gd.layout.title)) || '').toString();
        return { n: i + 1, block: b, gd, title: (h4 && h4.childNodes[0] ? h4.childNodes[0].textContent : plotTitle).trim(),
                 plotTitle: plotTitle.replace(/<[^>]+>/g, ' ') };
      });
  }

  const PALETTE = [['black', 0, 0, 0], ['black', 68, 68, 68], ['grey', 127, 127, 127], ['grey', 187, 187, 187], ['grey', 128, 128, 128],
    ['blue', 76, 120, 168], ['blue', 99, 110, 250], ['blue', 31, 119, 180], ['blue', 8, 48, 107], ['orange', 255, 176, 0],
    ['orange', 255, 127, 14], ['red', 214, 39, 40], ['red', 239, 85, 59], ['red', 225, 71, 37], ['green', 44, 160, 44],
    ['purple', 148, 103, 189]];
  function colourName(c) {
    if (typeof c !== 'string') return null;
    const named = { grey: 'grey', gray: 'grey', black: 'black', red: 'red', blue: 'blue', green: 'green', orange: 'orange' };
    if (named[c.toLowerCase()]) return named[c.toLowerCase()];
    let rgb = null;
    const hex = c.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      const h = hex[1].length === 3 ? hex[1].split('').map((x) => x + x).join('') : hex[1];
      rgb = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    const m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
    if (m) rgb = m.slice(1, 4).map(Number);
    if (!rgb) return null;
    let best = null;
    let dist = Infinity;
    for (const [name, r, g, b] of PALETTE) {
      const d = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2;
      if (d < dist) { dist = d; best = name; }
    }
    return best;
  }

  // what a chart contains: star, dashed/dotted lines, error bars, colours, crosses, colour scale
  function features(gd) {
    const f = new Set();
    const addColour = (c) => (Array.isArray(c) ? c : [c]).forEach((x) => { const n = colourName(x); if (n) f.add(n); });
    for (const t of gd.data || []) {
      const m = t.marker || {};
      const l = t.line || {};
      f.add(t.type);
      if (m.symbol === 'star') f.add('star');
      if (m.symbol === 'x') { f.add('cross'); f.add('black'); }
      if ((t.error_x && t.error_x.array) || (t.error_y && t.error_y.array)) f.add('errorbars');
      if (l.dash === 'dash') f.add('dashed');
      if (l.dash === 'dot') f.add('dotted');
      if (m.colorscale || m.showscale || t.colorscale) f.add('scale');
      addColour(m.color);
      addColour(l.color);
      if (t.type === 'waterfall') { f.add('red'); f.add('green'); }
      if (t.type === 'heatmap') f.add('blue');
    }
    for (const s of (gd.layout && gd.layout.shapes) || []) {
      const d = (s.line || {}).dash;
      f.add(d === 'dash' ? 'dashed' : d === 'dot' ? 'dotted' : 'solidline');
      addColour((s.line || {}).color);
    }
    return f;
  }

  // ---------------- the knowledge base: one entry per kind of chart ----------------
  const SETTINGS = [
    [/\bC\b|regularisation/, 'C controls how simple the Logistic Regression is: a small C keeps it simple and smooth, a large C lets it follow the training data more closely.'],
    [/max depth/, 'Max depth is how many yes/no questions deep a tree may go. Deeper trees can learn more detail but may memorise the training data.'],
    [/min samples leaf/, 'Min samples leaf is the smallest group allowed at the end of a branch. Bigger values make simpler, more general trees.'],
    [/number of neighbours|\bK\b/, 'K is how many of the most similar past people K-Nearest Neighbours looks at to make a prediction.'],
    [/weight type/, 'Weight type: "uniform" counts every neighbour equally; "distance" lets closer neighbours count more.'],
    [/distance \(p\)/, 'p is how distance is measured: p = 1 adds up the differences column by column, p = 2 is the straight-line distance.'],
    [/number of trees/, 'Number of trees is how many decision trees vote together in the Random Forest.'],
    [/hidden layer/, 'Hidden layer size is the number of neurons in the Neural Network\'s middle layer.'],
  ];

  function kb(kind, w, title) {
    const { P, O } = w;
    const K = {
      hyper: {
        name: 'a hyperparameter chart',
        what: `It shows how one setting of a model changes its score. ChurnSense tried several values and kept the best one. ${(SETTINGS.find(([re]) => re.test(title)) || [null, ''])[1]}`,
        read: 'Higher is better. Each point is the average cross-validated ROC-AUC for that value, averaged over the other settings tried.',
        parts: {
          main: 'The **blue line or blue bars** show the average score for each value of the setting, measured with cross-validation on the training data.',
          orange: 'The **orange** bar or star marks the best value of the setting.',
          star: 'The **star** marks the best value: the one with the highest average score. Its label gives the value and the score.',
          dashed: 'The **orange dashed line** is drawn at the best score, so you can see how far below it the other values are. If every point is close to it, this setting hardly matters.',
          x: 'The **bottom axis** lists the values of the setting that were tried.',
          y: 'The **left axis** is the mean cross-validated ROC-AUC (0.5 = guessing, 1 = perfect).',
        },
      },
      confusion: {
        name: 'a confusion matrix',
        what: `It shows how the model sorted the ${P} in the test set: what it predicted against what really happened.`,
        read: 'Rows are what really happened; columns are what the model predicted. Top-left and bottom-right are correct predictions.',
        parts: {
          main: `Each **square** counts ${P}. Top-left: did not ${w.leave ? 'leave' : 'join'} and predicted correctly. Top-right: wrongly flagged (false alarms). Bottom-left: ${O} but the model missed them. Bottom-right: ${O} and the model caught them.`,
          blue: 'The **blue shade** shows how many people are in each square: darker = more.',
          numbers: 'The **numbers** are counts of people in the test set (the 20% the model never saw while training).',
          star: `The **star** marks the bottom-right square: the ${P} who really ${O} and were correctly found.`,
          x: 'The **bottom axis** is what the model predicted.',
          y: 'The **left axis** is what really happened.',
        },
      },
      tree: {
        name: 'a decision tree diagram',
        what: 'It draws the best Decision Tree as a flowchart of yes/no questions.',
        read: `Start at the top. Each node asks about one column; follow "yes" or "no" down. The nodes at the bottom are the final groups, with the share who ${O}.`,
        parts: {
          main: `Each **node** is a group of ${P}; its label shows the question, the group size and the share who ${O}.`,
          scale: `Each **node's colour** shows the share who ${O} in that group: the warmer the colour, the higher the share.`,
          orange: `The **orange path** leads to the group with the highest share who ${O}.`,
          grey: 'The **grey lines** are the other branches of the tree.',
          star: `The **star** marks the final group with the highest share who ${O}.`,
        },
      },
      compare: {
        name: 'the model comparison',
        what: 'It compares all five models, scored on the test set.',
        read: 'A longer bar is a better model. The best one is used for every other chart.',
        parts: {
          main: 'Each **bar** is one model\'s score on the test set.',
          orange: 'The **orange bar** is the best model.',
          dashed: 'The **orange dashed line** marks the best model\'s score.',
          star: 'The **star** marks the best model and its score.',
          x: 'The **bottom axis** is ROC-AUC on the test set (0.5 = guessing, 1 = perfect).',
          y: 'The **left axis** lists the models.',
        },
      },
      roc: {
        name: 'the ROC curves',
        what: `It shows how well each model separates ${P} who ${O} from the others, at every possible threshold.`,
        read: 'A curve that bends up towards the top-left corner is better. The AUC in the legend sums each curve up in one number (the area under it).',
        parts: {
          main: 'Each **coloured line** is one model (see the legend for its AUC).',
          dashed: 'The **grey dashed diagonal** is a random guess (AUC 0.5). A useful model\'s curve stays well above it.',
          grey: 'The **grey dashed diagonal** is a random guess (AUC 0.5). A useful model\'s curve stays well above it.',
          star: 'The **star** marks where the best model sits at the current threshold (0.5): how many it finds against how many it wrongly flags.',
          x: `The **bottom axis** (false positive rate) is the share of ${P} who did not ${w.leave ? 'leave' : 'join'} but were wrongly flagged.`,
          y: `The **left axis** (true positive rate, or recall) is the share of ${P} who ${O} that the model found.`,
        },
      },
      reasons: {
        name: 'the top reasons chart (permutation importance)',
        what: 'It shows which columns matter most to the best model\'s predictions.',
        read: 'A longer bar means a more important column. Each bar shows how much the model\'s score drops when that column is shuffled (mixed up between people), so it no longer carries any information.',
        parts: {
          main: 'Each **bar** is one column; its length is how much worse the model gets without it.',
          orange: 'The **orange bar** is the top reason, the most important column.',
          black: 'The **thin black lines** at the end of the bars are **error bars**. The shuffling was repeated several times, and the error bar shows how much the result varied (one standard deviation either side). A long error bar means that column\'s importance is less certain; if it reaches zero, the column may not matter at all.',
          star: 'The **star** marks the top reason.',
          x: 'The **bottom axis** is the drop in ROC-AUC when the column is shuffled (higher = more important).',
          y: 'The **left axis** lists the columns, most important at the top.',
          note: 'This chart shows how much a column matters, not in which direction. The SHAP summary and the direction charts show the direction.',
        },
      },
      shap: {
        name: 'the SHAP summary',
        what: 'It shows how each column pushed the risk up or down for every single person.',
        read: `Each row is a column and each dot is one person. Dots to the right raised that person's chance of having ${O}; dots to the left lowered it. Rows are sorted by importance.`,
        parts: {
          main: 'Each **dot** is one person, placed by how much that column changed their predicted probability.',
          scale: 'The **dot colour** is that person\'s value for the column: blue = low, red = high (see the colour bar). For example, red dots on the right mean high values push the risk up.',
          red: 'A **red dot** is a person with a high value for that column (blue = low).',
          blue: 'A **blue dot** is a person with a low value for that column (red = high).',
          grey: 'The **grey vertical line** at 0 means "no effect". The further a dot is from it, the bigger that column\'s effect for that person.',
          black: 'The **vertical line** at 0 means "no effect". The further a dot is from it, the bigger that column\'s effect for that person.',
          star: 'The **star** marks the column with the strongest effect overall.',
          x: 'The **bottom axis** is the change in probability: 0.1 means that column added 10 percentage points for that person.',
          y: 'The **left axis** lists the columns, most important at the top.',
        },
      },
      rate: {
        name: 'a direction-of-effect chart',
        what: `It shows the share of ${P} who ${O} for each value or range of one column.`,
        read: `A taller bar means more ${P} in that group ${O}. Compare each bar with the dotted line for everyone.`,
        parts: {
          main: `Each **bar** is a group of ${P} with that value (numbers are split into ranges with similar numbers of people).`,
          red: `The **red bar** is the group with the highest share who ${O}.`,
          dotted: `The **grey dotted line** is the overall average for everyone, so you can see which groups are above or below normal.`,
          grey: `The **grey dotted line** is the overall average for everyone, so you can see which groups are above or below normal.`,
          dashed: 'The **orange dashed line** is at the level of the highest group.',
          orange: 'The **orange dashed line and star** mark the highest group.',
          star: `The **star** marks the group with the highest share who ${O}.`,
          numbers: `The **percentage on each bar** is the share of that group who ${O}.`,
          x: 'The **bottom axis** lists the values or ranges of the column.',
          y: `The **left axis** is the percentage who ${O}.`,
        },
      },
      elbow: {
        name: 'the elbow plot',
        what: 'It helps choose how many groups (segments) to split people into.',
        read: 'Inertia measures how spread out the groups are (lower = tighter). It always falls as you add groups. The "elbow", where it stops falling quickly, is a good number of groups.',
        parts: {
          main: 'The **blue line** shows the inertia for each number of groups.',
          star: 'The **star** marks the suggested number of groups, at the bend.',
          dashed: 'The **orange dashed line** is at the inertia for the suggested number of groups.',
          x: 'The **bottom axis** is the number of groups (k).',
          y: 'The **left axis** is the inertia (how spread out the groups are).',
        },
      },
      segments: {
        name: 'the segments chart',
        what: 'It shows people split into groups with similar values, found with K-Means clustering.',
        read: 'Each dot is a person, coloured by group and placed by two columns. Groups that sit apart are clearly different.',
        parts: {
          main: 'Each **dot** is one person; its **colour** is the group it belongs to.',
          cross: 'The **black X marks** are the group centres: the typical member of each group.',
          black: 'The **black X marks** are the group centres: the typical member of each group.',
          star: `The **star** marks the group with the highest share who ${O}.`,
          x: 'The **two axes** are the columns chosen for this view (change them on the Why page).',
          y: 'The **two axes** are the columns chosen for this view (change them on the Why page).',
        },
      },
      fairness: {
        name: 'the fairness check',
        what: 'It checks whether the best model treats groups (for example boys and girls, or age groups) differently.',
        read: 'Compare the bars between groups. Big differences are a warning sign. A warning appears when one group is flagged at less than 80% of another group\'s rate (the four-fifths rule), or when recall differs by more than 10 points.',
        parts: {
          blue: 'The **blue bars** are the average predicted chance for each group.',
          red: `The **red bars** are the share of each group flagged as likely to have ${O} at the current threshold.`,
          grey: `The **grey bars** are recall: of the people in that group who really ${O}, the share the model caught.`,
          star: 'The **star** names the largest gap between groups.',
          main: 'Each set of **bars** is one group; blue = average predicted %, red = % flagged, grey = recall %.',
        },
      },
      person: {
        name: 'a person\'s explanation (waterfall)',
        what: 'It shows why the model gave this one person their risk.',
        read: 'Start from the average risk, then each bar adds or removes a reason; the last bar is this person\'s risk.',
        parts: {
          red: 'The **red bars** are reasons that raise this person\'s risk.',
          green: 'The **green bars** are reasons that lower it.',
          main: 'Each **bar** is one reason (a column and this person\'s value). "All other columns" groups the rest together.',
          star: 'The **star** gives the final risk and the biggest reason.',
          x: 'The **bottom axis** is the probability, in percent.',
        },
      },
      histogram: {
        name: 'a histogram',
        what: 'It shows how the values of one column are spread out.',
        read: 'Each bar counts how many people have a value in that range. The colours split them by outcome.',
        parts: {
          main: 'Each **bar** counts the people whose value falls in that range.',
          blue: 'The **blue bars** are people for whom the outcome is "No".',
          red: 'The **red bars** are people for whom the outcome is "Yes".',
          dashed: 'The **orange dashed line** is the median: half the people are below it and half above.',
          star: 'The **star** label gives the median.',
        },
      },
      box: {
        name: 'a box plot',
        what: 'It compares how the values of a column are spread in different groups.',
        read: 'The box holds the middle half of the values; the line inside it is the median.',
        parts: {
          main: 'Each **box** holds the middle half of the values; the **line inside** is the median; the **whiskers** show the usual range; **separate dots** are unusual values (outliers).',
          blue: 'The **blue boxes** are people for whom the outcome is "No".',
          red: 'The **red boxes** are people for whom the outcome is "Yes".',
          star: 'The **star** label names the group with the highest median.',
        },
      },
      scatter: {
        name: 'a scatter plot',
        what: 'It shows two columns against each other, one dot per person.',
        read: 'If the dots form a rising band, the two columns go up together. The correlation r goes from -1 to 1 (0 = no straight-line link).',
        parts: {
          main: 'Each **dot** is one person.',
          blue: 'The **blue dots** are people for whom the outcome is "No".',
          red: 'The **red dots** are people for whom the outcome is "Yes".',
          star: 'The **star** label gives the correlation r between the two columns.',
        },
      },
      regression: {
        name: 'the regression chart',
        what: 'It checks a simple linear regression that predicts a number (for example income) from other columns.',
        read: 'Each dot compares the real value with the predicted one. The closer the dots sit to the dashed line, the better the prediction.',
        parts: {
          main: 'Each **dot** is a row from the test set: real value along the bottom, predicted value up the side.',
          dashed: 'The **orange dashed diagonal** is a perfect prediction (predicted = real).',
          orange: 'The **orange dashed diagonal** is a perfect prediction (predicted = real).',
          star: 'The **label** gives R² (the share of the variation explained, 1 = perfect) and MAE (the average error).',
          x: 'The **bottom axis** is the real value.',
          y: 'The **left axis** is the predicted value.',
        },
      },
    };
    return K[kind];
  }

  function kindOf(chart) {
    const types = new Set((chart.gd.data || []).map((t) => t.type));
    const t = `${chart.title} ${chart.plotTitle}`.toLowerCase();
    if (types.has('waterfall')) return 'person';
    if (types.has('heatmap') || /confusion matrix/.test(t)) return 'confusion';
    if (types.has('box')) return 'box';
    if (types.has('histogram')) return 'histogram';
    const rules = [['hyper', /effect of/], ['tree', /decision tree rules|tree rules/], ['compare', /model comparison/],
      ['roc', /roc curve/], ['shap', /shap/], ['reasons', /top reasons|permutation importance/],
      ['rate', /direction of effect|outcome rate|% who|rate by|driver:/], ['elbow', /elbow/], ['fairness', /fairness/],
      ['segments', /segment|cluster/], ['regression', /regression|actual vs predicted/], ['scatter', /scatter/]];
    const hit = rules.find(([, re]) => re.test(t));
    return hit ? hit[0] : null;
  }

  // ---------------- understanding the question ----------------
  const ORDINALS = { first: 1, '1st': 1, second: 2, '2nd': 2, third: 3, '3rd': 3, fourth: 4, '4th': 4, fifth: 5, '5th': 5,
    sixth: 6, '6th': 6, seventh: 7, '7th': 7, eighth: 8, '8th': 8, ninth: 9, '9th': 9, tenth: 10, '10th': 10 };
  const PARTS = [
    ['black', /\bblack\b|\bdark (line|lines)\b|whisker|error ?bars?|thin lines?|small lines?|lines? (on|at the end of|through|across) (the )?bars?/],
    ['star', /\bstars?\b|★/],
    ['dashed', /\bdash(ed)?\b/],
    ['dotted', /\bdott(ed)?\b|\bdot line\b/],
    ['cross', /\bx marks?\b|\bcross(es)?\b|\bx's\b/],
    ['orange', /\borange\b|\byellow\b|\bgold\b/],
    ['red', /\bred\b/], ['blue', /\bblue\b/], ['green', /\bgreen\b/], ['grey', /\bgr[ae]y\b/],
    ['scale', /colou?r ?(bar|scale)|\blegend\b|colou?rs? mean/],
    ['numbers', /\bnumbers? (on|in|inside)\b|percentages? on|labels? on|written on/],
    ['x', /x[- ]?axis|bottom axis|horizontal axis|along the bottom|across the bottom/],
    ['y', /y[- ]?axis|left axis|vertical axis|up the side|side axis/],
  ];
  const KIND_WORDS = [['roc', /\broc\b/], ['confusion', /confusion|matrix/], ['shap', /\bshap\b|summary plot/],
    ['elbow', /elbow|inertia/], ['segments', /segment|cluster/], ['fairness', /fairness|fair\b/], ['compare', /comparison|compare/],
    ['reasons', /top reasons?|importance/], ['rate', /direction|rate by|outcome rate/], ['tree', /tree (diagram|rules)|decision tree chart/],
    ['person', /waterfall|person'?s chart/], ['hyper', /effect of|hyperparameter|setting chart/], ['histogram', /histogram/],
    ['box', /box ?plot/], ['scatter', /scatter/], ['regression', /regression chart|actual vs predicted/]];

  function findChart(q, charts) {
    const num = q.match(/\b(?:graph|chart|plot|figure|diagram)\s*(?:no\.?|number|#)?\s*(\d+)\b/) || q.match(/\b(\d+)(?:st|nd|rd|th)?\s+(?:graph|chart|plot|figure)\b/);
    if (num) return charts.find((c) => c.n === Number(num[1])) || { missing: Number(num[1]) };
    if (/\blast (graph|chart|plot|figure|one)\b/.test(q)) return charts[charts.length - 1];
    const ord = q.match(/\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|seventh|7th|eighth|8th|ninth|9th|tenth|10th)\b/);
    if (ord) return charts.find((c) => c.n === ORDINALS[ord[1]]) || { missing: ORDINALS[ord[1]] };
    const kw = KIND_WORDS.find(([, re]) => re.test(q));
    if (kw) {
      const same = charts.filter((c) => kindOf(c) === kw[0]);
      if (same.length) return same.includes(H.chart) ? H.chart : same[0];
    }
    return null;
  }

  // ---------------- answers ----------------
  function describeChart(chart, w) {
    const kind = kindOf(chart);
    const k = kind && kb(kind, w, `${chart.title} ${chart.plotTitle}`);
    const f = features(chart.gd);
    const head = `**Chart ${chart.n}: ${chart.title}**`;
    if (!k) return `${head}\n${fallbackParts(chart, f)}`;
    const order = ['main', 'orange', 'red', 'blue', 'green', 'grey', 'scale', 'black', 'cross', 'dotted', 'dashed', 'star', 'numbers', 'x', 'y'];
    const needs = { orange: 'orange', red: 'red', blue: 'blue', green: 'green', grey: 'grey', scale: 'scale', black: kind === 'reasons' ? 'errorbars' : 'black',
                    cross: 'cross', dotted: 'dotted', dashed: 'dashed', star: 'star' };
    const seen = new Set();
    const lines = [];
    for (const key of order) {
      const text = k.parts[key];
      if (!text || seen.has(text) || (needs[key] && !f.has(needs[key]))) continue;
      seen.add(text);
      lines.push(`• ${text}`);
    }
    return `${head} is ${k.name}. ${k.what}\n**How to read it:** ${k.read}\n${lines.join('\n')}${k.parts.note ? `\n${k.parts.note}` : ''}`;
  }

  function fallbackParts(chart, f) {
    const has = [];
    if (f.has('star')) has.push('a star marking the best or most important value');
    if (f.has('dashed')) has.push('a dashed line marking a reference level');
    if (f.has('dotted')) has.push('a dotted line for the overall average');
    if (f.has('errorbars')) has.push('black error bars showing how certain each value is');
    return has.length ? `This chart has ${has.join(', ')}.` : 'Hover over any part of the chart to see its exact values.';
  }

  function answerPart(chart, part, w) {
    const kind = kindOf(chart);
    const k = kind && kb(kind, w, `${chart.title} ${chart.plotTitle}`);
    const f = features(chart.gd);
    const head = `In **chart ${chart.n}, "${chart.title}"**: `;
    if (k && k.parts[part]) {
      if (part === 'black' && kind === 'reasons' && !f.has('errorbars')) {
        return `${head}this chart has no black lines. ${k.parts.main}`;
      }
      return head + k.parts[part];
    }
    if (part === 'black' && f.has('errorbars')) return `${head}the thin black lines are **error bars**: they show how much the value varies, so a long one means the value is less certain.`;
    if (part === 'dashed' || part === 'dotted' || part === 'star' || ['orange', 'red', 'blue', 'green', 'grey', 'black'].includes(part)) {
      return `${head}I can't find a ${part === 'star' ? 'star' : `${part} line or mark`} there. ${k ? `${k.what} ${k.read}` : fallbackParts(chart, f)}`;
    }
    return `${head}${k ? `${k.what} ${k.read}` : fallbackParts(chart, f)}`;
  }

  const GLOSSARY = [
    [/roc[- ]?auc|\bauc\b/, '**ROC-AUC** measures how well a model ranks people: if you pick one person who {O} and one who did not, it is the chance the model gives the first a higher risk. 0.5 is guessing, 1.0 is perfect; around 0.8 is good.'],
    [/\brecall\b|true positive rate/, '**Recall** is the share of the people who really {O} that the model caught. 68% recall means it found about 2 in 3 of them.'],
    [/\bprecision\b/, '**Precision** is the share of the people the model flagged who really {O}. Low precision means many false alarms.'],
    [/\baccuracy\b|\baccurate\b/, '**Accuracy** is the share of all predictions that were right. It can mislead when few people {O}: always guessing "no" can be 84% accurate and still useless, so ChurnSense ranks models by ROC-AUC instead.'],
    [/\bf1\b/, '**F1-score** balances precision and recall in one number (their harmonic mean).'],
    [/threshold|cut-?off/, '**Threshold** is the cut-off (0.5 by default) above which someone counts as "predicted to {Ov}". Lower it on the Train page to catch more people, at the cost of more false alarms.'],
    [/cross[- ]?validation|\bcv\b/, '**Cross-validation** cuts the training data into 3 to 5 parts and lets each part take a turn as a practice test, so settings are chosen on data the model did not learn from.'],
    [/test set|unseen|held back/, 'The **test set** is the 20% of people hidden while the models train. All scores are measured on it, so they are honest.'],
    [/overfit/, '**Overfitting** is when a model memorises its training examples instead of learning general patterns. The test set and cross-validation catch it.'],
    [/hyper ?parameter|setting/, 'A **hyperparameter** is a setting chosen before training, such as tree depth or the number of neighbours. ChurnSense tries several values and keeps the best (the "effect of ..." charts).'],
    [/confusion matrix/, 'A **confusion matrix** is a 2x2 table of predictions against reality: correct, false alarms (flagged but did not {Ov}) and missed (did {Ov} but not flagged).'],
    [/false (positive|alarm)/, 'A **false positive** (false alarm) is someone the model flagged who did not actually {Ov}.'],
    [/false negative|missed/, 'A **false negative** is someone who really {O} but the model did not flag.'],
    [/permutation|importance/, '**Permutation importance** shuffles one column at a time and measures how much worse the model gets. A big drop means the column matters.'],
    [/\bshap\b/, '**SHAP** splits one person\'s risk into a contribution from each column ("5 failed subjects added 30 points"). The pieces add up exactly to their probability.'],
    [/k-?means|cluster|segment/, '**Segments** are groups of similar people found with **K-Means**, which places a centre for each group and assigns each person to the nearest centre.'],
    [/inertia|elbow/, '**Inertia** is how spread out the groups are (lower = tighter). The **elbow** is where adding more groups stops helping much.'],
    [/four[- ]fifths|proxy|fairness|\bbias/, 'The **fairness check** compares groups (e.g. by gender). It warns when one group is flagged at under 80% of another\'s rate (the **four-fifths rule**). Sensitive columns are left out of training, but other columns can act as a **proxy** for them, so the check still matters.'],
    [/standard deviation|error bars?|whisker/, '**Error bars** show how much a value varies when it is measured several times (one standard deviation either side). Longer = less certain.'],
    [/median/, 'The **median** is the middle value: half the people are below it and half above.'],
    [/correlation|\br =/, '**Correlation (r)** goes from -1 to 1: near 1, two columns rise together; near -1, one falls as the other rises; near 0, no straight-line link.'],
    [/r²|\br2\b|r-squared|\bmae\b/, '**R²** is the share of the variation a regression explains (1 = perfect). **MAE** is its average error, in the column\'s own units.'],
    [/probabilit|\brisk\b|high|medium|low level|levels?/, 'Each person gets a **probability** (risk) from 0% to 100%. Levels: **Low** below 30%, **Medium** 30 to 60%, **High** 60% and above.'],
    [/logistic regression/, '**Logistic Regression** learns a weighted formula: each column pushes the risk up or down by a learned amount.'],
    [/decision tree/, 'A **Decision Tree** is a flowchart of yes/no questions ("Attendance below 80%?") learned from the data.'],
    [/nearest neighbo|\bknn\b/, '**K-Nearest Neighbours** finds the K most similar past people and predicts what happened to most of them.'],
    [/random forest/, 'A **Random Forest** is hundreds of slightly different decision trees that vote together.'],
    [/neural network|\bmlp\b/, 'A **Neural Network** passes the data through layers of connected "neurons" to learn more complex patterns.'],
    [/one[- ]hot/, '**One-hot encoding** turns a word column (Contract = Monthly / Yearly) into yes/no number columns the models can use.'],
    [/class weight|imbalance|balanced/, '**Class imbalance** means one outcome is rare. ChurnSense uses balanced class weights so models pay extra attention to the smaller group.'],
    [/synthetic|test data|sample/, 'The **samples** are test data. Employees and Customers are public datasets; Students, Teachers and Applicants are **synthetic** (made up), so no real person is shown. Choose **My own data** to analyse your institution\'s spreadsheet.'],
    [/\bchurn\b/, '**Churn** means people leaving: students dropping out, staff resigning, customers cancelling.'],
  ];
  const HOWTO = [
    [/upload|add (a )?(file|spreadsheet)|drag|csv|excel|xlsx|my own data/, 'To use your own data, go to **Home**, drag a CSV or Excel file into the drop box (or click it to choose). It needs one row per person and a column saying whether they {O} (Yes/No). Then type a question and press **Run**.'],
    [/share|e-?mail|send|word|report|docx/, 'Click **Share results** (top right) after an analysis. It makes a **Word report** and a **CSV list**, then sends them by email or opens your email app. The Why page also has **Download the results (Word)**.'],
    [/guest|sign ?(in|out)|log ?(in|out)|saved?|keep my/, '**Signed in**, your details and spreadsheets are saved in this browser for next time; **Sign out** deletes them. As a **guest**, nothing is saved and closing the page clears everything.'],
    [/privacy|private|safe|secure|upload(ed)? anywhere|leave my computer/, 'Your spreadsheet **never leaves your computer**: all the analysis runs inside your browser. Nothing is uploaded unless you choose to email results.'],
    [/what can i ask|examples?|how do i ask|commands?/, 'Try: "Which students are most likely to drop out, and why?", "Explain student S01089", "What if OverTime = No for employee 1043", "Is the model fair?", "Show the segments", "Show all graphs". Type **help** in the main prompt box for the full list.'],
    [/quick|thorough|slow|speed|how long/, '**Quick** (the default) trains the five models in about 10 to 40 seconds. **Thorough** (on the Train page) tries many more settings and takes several minutes.'],
    [/best model|which model|why (this|that) model/, 'ChurnSense trains five models and picks the one with the highest cross-validated **ROC-AUC**. The Compare page shows them side by side.'],
    [/all graphs|every (graph|chart)/, '**All graphs** (in the sidebar) shows all 27 charts grouped by model, with the best value on each marked with a star.'],
    [/what[- ]if/, 'On **Predict & Explain**, load a person, change a value (for example OverTime = No) and click **What-if** to see how their risk changes.'],
  ];

  function fill(text, w) {
    return text.replace(/\{O\}/g, w.O).replace(/\{Ov\}/g, w.leave ? 'leave' : 'join');
  }

  function topicAnswers(q, w) {
    const out = [];
    for (const [re, text] of [...HOWTO, ...GLOSSARY]) {
      if (re.test(q) && !out.includes(text)) out.push(text);
      if (out.length >= 3) break;
    }
    return out.map((t) => fill(t, w));
  }

  function chartChips(charts) {
    return charts.slice(0, 6).map((c) => [`${c.n}. ${c.title.slice(0, 38)}`, `Explain chart ${c.n}`]);
  }

  function answer(raw) {
    const q = raw.toLowerCase().trim();
    const w = words();
    const charts = pageCharts();
    const parts = PARTS.filter(([, re]) => re.test(q)).map(([p]) => p);
    const chartWords = /\b(graph|chart|plot|figure|diagram|line|bars?|dots?|axis|colou?rs?|legend|star)\b/.test(q);
    const wantsWhole = /^(explain|describe|what is|what's|what does|read)\b.*\b(graph|chart|plot|figure|this)\b/.test(q) && !parts.length;
    let chart = findChart(q, charts);
    if (chart && chart.missing) {
      return { text: `There ${charts.length === 1 ? 'is only 1 chart' : `are ${charts.length} charts`} on this page, so there is no chart ${chart.missing}.`, chips: chartChips(charts) };
    }
    const topics = topicAnswers(q, w);
    // a pasted sentence ("what does this line mean: recall 68%") is about words, not a chart
    if (!chart && !parts.length && topics.length) return { text: topics.join('\n\n') };
    if (chart || parts.length || (chartWords && charts.length)) {
      if (!charts.length) return { text: 'There are no charts on this page yet. Press **Run** on the Home page (or open **All graphs**) and ask me again.' };
      chart = chart || H.chart || (charts.length === 1 ? charts[0] : null);
      if (!chart) {
        return { text: 'Which chart do you mean? Pick one below, or say "the first chart", "chart 3" or its name.', chips: chartChips(charts) };
      }
      H.chart = chart;
      flash(chart);
      const follow = [['Explain the whole chart', `Explain chart ${chart.n}`], ['What does the star mean?', `What does the star in chart ${chart.n} mean?`],
                      ['What do the axes show?', `What do the x axis and y axis of chart ${chart.n} show?`]];
      if (!parts.length || wantsWhole) return { text: describeChart(chart, w), chips: follow.slice(1) };
      return { text: parts.map((p) => answerPart(chart, p, w)).join('\n\n'), chips: follow };
    }
    return null;
  }

  // ---------------- AI fallback (only when the visitor has entered a Gemini key) ----------------
  async function askGemini(question) {
    const key = (document.querySelector('#ai-key') || {}).value;
    if (!key || !key.trim()) return null;
    const charts = pageCharts().map((c) => `Chart ${c.n}: ${c.title} (${kindOf(c) || 'chart'})`).join('\n');
    const model = ((document.querySelector('#ai-model') || {}).value || '').trim() || 'gemini-flash-latest';
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key.trim() },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: 'You are the help assistant inside ChurnSense, a website that predicts who will leave or join an organisation and explains why. Answer in 2 to 5 short, plain sentences for a school student. If unsure, say so.' }] },
        contents: [{ role: 'user', parts: [{ text: `Charts on the page:\n${charts}\n\nQuestion: ${question}` }] }],
        generationConfig: { temperature: 0.3 },
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return null;
    return ((data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [])
      .map((p) => p.text || '').join('').trim() || null;
  }

  // ---------------- the chat panel ----------------
  const $h = (s) => document.querySelector(s);
  function say(who, text, chips) {
    const log = $h('#help-log');
    const msg = document.createElement('div');
    msg.className = `hmsg ${who}`;
    msg.innerHTML = rich(text);
    log.append(msg);
    if (chips && chips.length) showChips(chips);
    log.scrollTop = log.scrollHeight;
  }

  function showChips(chips) {
    const box = $h('#help-chips');
    box.replaceChildren(...chips.map(([label, ask]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip ghost';
      b.textContent = label;
      b.addEventListener('click', () => ask === '__ai' ? null : send(ask));
      return b;
    }));
  }

  function flash(chart) {
    if (!chart || !chart.block) return;
    chart.block.scrollIntoView({ behavior: 'smooth', block: 'center' });
    chart.block.classList.remove('help-flash');
    void chart.block.offsetWidth;
    chart.block.classList.add('help-flash');
  }

  const START_CHIPS = [['What does the black line in the first chart mean?', 'What does the black line in the first chart mean?'],
    ['Explain the first chart', 'Explain the first chart'], ['What is ROC-AUC?', 'What is ROC-AUC?'],
    ['How do I share results?', 'How do I share results?'], ['Is my data private?', 'Is my data private?']];

  async function send(text) {
    const q = String(text || '').trim();
    if (!q) return;
    say('me', q);
    $h('#help-chips').replaceChildren();
    const a = answer(q);
    if (a) { say('bot', a.text, a.chips); return; }
    let ai = null;
    try { ai = await askGemini(q); } catch (e) { ai = null; }
    if (ai) { say('bot', `${ai}\n\n(AI answer from Gemini: check it against the charts.)`); return; }
    say('bot', 'I\'m not sure about that one. I can explain any chart on this page (its lines, colours, stars and axes), words like ROC-AUC, recall or SHAP, and how to use ChurnSense. Try one of these:', START_CHIPS);
  }

  function openHelp(opts = {}) {
    const panel = $h('#help-panel');
    panel.hidden = false;
    H.open = true;
    $h('#help-fab').setAttribute('aria-expanded', 'true');
    if (!$h('#help-log').childElementCount) {
      const state = typeof S !== 'undefined' ? S : null;
      // greet the signed-in person by their first name; guests get "Hi there"
      const first = state && state.profile && !state.profile.guest ? String(state.profile.name || '').trim().split(/\s+/)[0] : '';
      say('bot', `Hi ${first || 'there'}! How can I help you today?\n\nI can explain what a chart or line means, what a word like ROC-AUC means, or how to do something in ChurnSense.`, START_CHIPS);
    }
    if (opts.chart) {
      const chart = pageCharts().find((c) => c.gd === opts.chart);
      if (chart) {
        H.chart = chart;
        say('me', `Explain chart ${chart.n}: ${chart.title}`);
        $h('#help-chips').replaceChildren();
        const a = answer(`explain chart ${chart.n}`);
        say('bot', a.text, a.chips);
      }
    }
    if (!opts.chart) setTimeout(() => $h('#help-input').focus(), 50);
  }

  function closeHelp() {
    $h('#help-panel').hidden = true;
    H.open = false;
    $h('#help-fab').setAttribute('aria-expanded', 'false');
    $h('#help-fab').focus();
  }

  function wireHelp() {
    $h('#help-fab').addEventListener('click', () => (H.open ? closeHelp() : openHelp()));
    $h('#help-close').addEventListener('click', closeHelp);
    $h('#help-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const input = $h('#help-input');
      const text = input.value;
      input.value = '';
      send(text);
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && H.open) closeHelp(); });
  }

  window.openHelp = openHelp;
  window.helpAnswer = (q) => { const a = answer(q); return a ? a.text : null; };   // used by the tests
  wireHelp();
})();
