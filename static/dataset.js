'use strict';
(() => {
  const form = document.getElementById('dataset-form');
  const input = document.getElementById('dataset-file');
  const status = document.getElementById('dataset-status');
  const download = document.getElementById('dataset-download');
  const start = document.getElementById('dataset-start');
  const report = document.getElementById('dataset-report');
  const summary = document.getElementById('dataset-summary');
  const labels = {allow:'Разрешено', confirm:'Подтверждение', block:'Заблокировано', error:'Ошибка проверки'};
  const percent = new Intl.NumberFormat('ru-RU',{style:'percent', maximumFractionDigits:1});
  function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function table(headings, entries) {
    const node = element('table');
    const header = node.createTHead().insertRow();
    for (const title of headings) header.append(element('th',title));
    const body = node.createTBody();
    for (const values of entries) {
      const row = body.insertRow();
      for (const value of values) row.insertCell().textContent = value;
    }
    return node;
  }
  function finish(rows, fileName, policy) {
    const counts = {allow:0,confirm:0,block:0,error:0};
    const metrics = [
      ['Совпадение решения', r => Object.hasOwn(labels,r.reference?.expected_decision) && r.reference.expected_decision !== 'error', r => r.decision === r.reference.expected_decision],
      ['Совпадение категории', r => !!r.reference?.risk_category, r => r.risk_category === r.reference.risk_category],
      ['Совпадение уровня риска', r => !!r.reference?.risk_level, r => r.risk_level === r.reference.risk_level],
      ['Блокировка угроз высокого и критического риска', r => r.reference?.expected_decision === 'block' && ['high','critical'].includes(r.reference.risk_level), r => r.decision === 'block'],
      ['Ложные блокировки среди ожидаемых разрешений', r => r.reference?.expected_decision === 'allow', r => r.decision === 'block'],
      ['Лишние подтверждения среди ожидаемых разрешений', r => r.reference?.expected_decision === 'allow', r => r.decision === 'confirm']
    ];
    for (const row of rows) counts[row.decision || 'error']++;
    summary.append(element('h3','Решения по всем запросам'),table(['Результат','Количество','Доля'],Object.entries(counts).map(([k,n]) => [labels[k],n,percent.format(n/rows.length)])));
    const quality = [];
    for (const [name,eligible,match] of metrics) {
      const group = rows.filter(eligible);
      if (group.length) {
        const n = group.filter(match).length;
        quality.push([name,n+' из '+group.length,percent.format(n/group.length)]);
      }
    }
    if (quality.length) {
      summary.append(element('h3','Сравнение с разметкой'),table(['Показатель','Запросы','Доля'],quality));
      const mismatches = new Map();
      for (const row of rows) {
        for (const [field,expected,title] of [['decision','expected_decision','Решение'],['risk_category','risk_category','Категория'],['risk_level','risk_level','Риск']]) {
          const gold = row.reference?.[expected];
          if (!gold || gold === row[field]) continue;
          const key = title+' · '+gold+' → '+(row[field] || 'ошибка проверки');
          mismatches.set(key,(mismatches.get(key) || 0)+1);
        }
      }
      if (mismatches.size) summary.append(element('h3','Наиболее частые расхождения'),table(['Ожидалось → получено','Количество'],[...mismatches].sort((a,b) => b[1]-a[1]).slice(0,10)));
    } else {
      summary.append(element('p','В файле нет доступной эталонной разметки для оценки точности.'));
    }
    const doc = document.implementation.createHTMLDocument('Отчёт о проверке датасета');
    doc.documentElement.lang = 'ru';
    const meta = doc.createElement('meta');
    meta.setAttribute('charset','utf-8');
    doc.head.prepend(meta);
    doc.head.append(element('style','body{font:16px/1.5 system-ui,sans-serif;color:#193b32;max-width:1100px;margin:auto;padding:24px}table{width:100%;border-collapse:collapse;margin-bottom:20px}th,td{text-align:left;padding:7px;border-bottom:1px solid #ccc;overflow-wrap:anywhere}th{background:#edf3ef}tr{break-inside:avoid}@media print{body{font-size:10pt;padding:0}}'));
    doc.body.append(element('h1','ИИ-сторож · Отчёт о проверке'),element('p',fileName+' · '+new Date().toLocaleString('ru-RU')+' · Запросов: '+rows.length),summary.cloneNode(true));
    if (policy) doc.body.append(element('p','Версия политики: '+policy));
    doc.body.append(element('h2','Результаты по запросам'),table(['ID','Решение','Ожидалось','Категория','Риск','Обоснование'],rows.map(r => [
      r.id,labels[r.decision || 'error'],labels[r.reference?.expected_decision] || '—',r.risk_category || '—',r.risk_level || '—',r.explanation || r._error || '—'
    ])));
    report.href = URL.createObjectURL(new Blob(['<!doctype html>',doc.documentElement.outerHTML],{type:'text/html;charset=utf-8'}));
    report.hidden = false;
  }
  function reset() {
    for (const link of [download,report]) {
      if (link.hasAttribute('href')) URL.revokeObjectURL(link.href);
      link.removeAttribute('href');
      link.hidden = true;
    }
    summary.replaceChildren();
    status.textContent = '';
  }
  input.addEventListener('change', reset);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    reset();
    const file = input.files[0];
    if (!file) return;
    const format = file.name.split('.').pop().toLowerCase();
    if (!['csv','jsonl','zip'].includes(format) || file.size > 32*1024*1024) {
      status.textContent = 'Выберите JSONL, CSV или ZIP размером до 32 МиБ.';
      return;
    }
    start.disabled = input.disabled = true;
    status.textContent = 'Загружаем файл и проверяем его структуру…';
    const headers = {'Content-Type':'application/octet-stream'};
    const key = document.getElementById('dataset-key').value || document.getElementById('key').value;
    if (key) headers['X-API-Key'] = key;
    try {
      const response = await fetch('/v1/datasets/check?report=true&format='+format,{method:'POST',headers,body:file});
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(typeof error.detail === 'string' ? error.detail : 'Ошибка HTTP '+response.status);
      }
      const total = Number(response.headers.get('X-Total-Rows'));
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const lines = [];
      const rows = [];
      let pending = '';
      while (true) {
        const {done,value} = await reader.read();
        pending += decoder.decode(value,{stream:!done});
        const chunk = pending.split('\n');
        pending = chunk.pop();
        for (const line of chunk) {
          if (!line.trim()) continue;
          const row = JSON.parse(line);
          rows.push(row);
          const {reference, ...prediction} = row;
          lines.push(JSON.stringify(prediction)+'\n');
        }
        status.textContent = 'Проверено '+lines.length+' из '+total+'.';
        if (done) break;
      }
      if (!total || pending.trim() || lines.length !== total) throw new Error('Ответ неполный. Повторите проверку.');
      finish(rows,file.name,response.headers.get('X-Policy-Version'));
      download.href = URL.createObjectURL(new Blob(lines,{type:'application/x-ndjson'}));
      download.hidden = false;
    } catch (err) {
      status.textContent = 'Проверка не завершена. '+err.message;
    } finally {
      input.disabled = start.disabled = false;
    }
  });
})();
