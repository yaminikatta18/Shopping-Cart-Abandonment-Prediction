const $ = id => document.getElementById(id);
const pct = x => (x * 100).toFixed(1) + '%';
const COL = {dt: '#f59e0b', rf: '#4f46e5'};

// slider labels
['browsing_time','products_viewed','cart_value','previous_purchases'].forEach(id => {
  $(id).addEventListener('input', () => $('v_' + id).textContent = $(id).value);
});

function show(prefix, r) {
  const ab = r.label === 'Cart Abandoned';
  $(prefix + '_box').className = 'pred ' + (ab ? 'abandon' : 'buy');
  $(prefix + '_label').textContent = (ab ? '🛒 ' : '✅ ') + r.label;
  $(prefix + '_bar').style.width = pct(r.probability);
  $(prefix + '_prob').textContent = 'Abandon probability: ' + pct(r.probability);
}

$('form').addEventListener('submit', async e => {
  e.preventDefault();
  $('error').textContent = '';
  const body = {
    browsing_time: $('browsing_time').value, products_viewed: $('products_viewed').value,
    cart_value: $('cart_value').value, previous_purchases: $('previous_purchases').value,
    discount_usage: $('discount_usage').checked ? 1 : 0
  };
  try {
    const res = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) { $('error').textContent = data.error; return; }
    show('rf', data.random_forest); show('dt', data.decision_tree);
    $('advice').textContent = data.advice;
    $('path').innerHTML = data.decision_tree.path.map(s => `<li>${s}</li>`).join('');
    $('result').classList.remove('hidden');
  } catch { $('error').textContent = 'Could not reach the server.'; }
});

function cmTable(name, cm) {
  const max = Math.max(...cm.flat());
  const cell = (v, ok) => `<td style="background:rgba(${ok ? '5,150,105' : '225,29,72'},${0.12 + 0.55 * v / max})"><b>${v}</b></td>`;
  return `<div class="cm"><h4>${name}</h4><table>
    <tr><th></th><th>Pred: Purchased</th><th>Pred: Abandoned</th></tr>
    <tr><th>Actual: Purchased</th>${cell(cm[0][0], true)}${cell(cm[0][1], false)}</tr>
    <tr><th>Actual: Abandoned</th>${cell(cm[1][0], false)}${cell(cm[1][1], true)}</tr></table></div>`;
}

fetch('/api/metrics').then(r => r.json()).then(m => {
  const d = m.dataset, M = m.models, names = Object.keys(M);
  $('dataset').innerHTML = [[d.rows, 'rows'], [d.train_rows + ' / ' + d.test_rows, 'train / test'],
    [d.features.length, 'features'], [pct(d.abandon_rate), 'abandon rate']]
    .map(([a, b]) => `<div><b>${a}</b><small>${b}</small></div>`).join('');
  $('rules').textContent = m.tree_rules;
  $('oob').textContent = pct(m.rf_oob_score);

  const keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
  const nice = {accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1-Score', roc_auc: 'ROC-AUC'};
  new Chart($('metricChart'), {type: 'bar', data: {labels: keys.map(k => nice[k]),
    datasets: names.map((n, i) => ({label: n, data: keys.map(k => M[n][k]), backgroundColor: i ? COL.rf : COL.dt}))},
    options: {scales: {y: {min: 0, max: 1}}, plugins: {title: {display: true, text: 'Test-set metrics'}}}});
  new Chart($('rocChart'), {type: 'line', data: {datasets: [
    ...names.map((n, i) => ({label: n, borderColor: i ? COL.rf : COL.dt, pointRadius: 0, tension: 0,
      data: M[n].roc.fpr.map((x, j) => ({x, y: M[n].roc.tpr[j]}))})),
    {label: 'Chance', borderColor: '#aaa', borderDash: [5, 5], pointRadius: 0, data: [{x: 0, y: 0}, {x: 1, y: 1}]}]},
    options: {parsing: false, scales: {x: {type: 'linear', min: 0, max: 1, title: {display: true, text: 'False positive rate'}},
      y: {min: 0, max: 1, title: {display: true, text: 'True positive rate'}}},
      plugins: {title: {display: true, text: 'ROC curve'}}}});

  $('metricTable').innerHTML = '<tr><th>Model</th>' + keys.map(k => `<th>${nice[k]}</th>`).join('') + '</tr>' +
    names.map(n => `<tr><td><b>${n}</b></td>${keys.map(k => `<td>${M[n][k].toFixed(3)}</td>`).join('')}</tr>`).join('');
  $('cms').innerHTML = names.map(n => cmTable(n, M[n].confusion_matrix)).join('');
  $('cv').innerHTML = Object.entries(m.cv_accuracy).map(([n, s]) => {
    const mean = s.reduce((a, b) => a + b) / s.length;
    const sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / s.length);
    return `<div><b>${pct(mean)} ± ${(sd * 100).toFixed(1)}</b><small>${n}</small></div>`;
  }).join('');

  const c = m.depth_curve;
  new Chart($('depthChart'), {type: 'line', data: {labels: c.depths, datasets: [
    {label: 'Train accuracy', data: c.train, borderColor: COL.rf, tension: .2},
    {label: 'Test accuracy', data: c.test, borderColor: COL.dt, tension: .2}]},
    options: {scales: {x: {title: {display: true, text: 'max_depth'}}, y: {min: 0.5, max: 1}}}});

  const im = m.importance, pretty = f => f.replace(/_/g, ' ');
  new Chart($('impChart'), {type: 'bar', data: {labels: im.features.map(pretty), datasets: [
    {label: 'Decision Tree (Gini)', data: im.decision_tree, backgroundColor: COL.dt},
    {label: 'Random Forest (Gini)', data: im.random_forest, backgroundColor: COL.rf},
    {label: 'RF permutation', data: im.rf_permutation, backgroundColor: '#10b981'}]},
    options: {indexAxis: 'y'}});
});
