'use strict';
(() => {
  const form = document.getElementById('dataset-form');
  const input = document.getElementById('dataset-file');
  const status = document.getElementById('dataset-status');
  const download = document.getElementById('dataset-download');
  const start = document.getElementById('dataset-start');
  function reset() {
    if (download.hasAttribute('href')) URL.revokeObjectURL(download.href);
    download.removeAttribute('href');
    download.hidden = true;
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
      const response = await fetch('/v1/datasets/check?format='+format,{method:'POST',headers,body:file});
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(typeof error.detail === 'string' ? error.detail : 'Ошибка HTTP '+response.status);
      }
      const total = Number(response.headers.get('X-Total-Rows'));
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const lines = [];
      const counts = {allow:0,confirm:0,block:0,error:0};
      let pending = '';
      while (true) {
        const {done,value} = await reader.read();
        pending += decoder.decode(value,{stream:!done});
        const chunk = pending.split('\n');
        pending = chunk.pop();
        for (const line of chunk) {
          if (!line.trim()) continue;
          const row = JSON.parse(line);
          if (['allow','confirm','block'].includes(row.decision)) counts[row.decision]++;
          else counts.error++;
          lines.push(line+'\n');
        }
        status.textContent = 'Проверено '+lines.length+' из '+total+'.';
        if (done) break;
      }
      if (!total || pending.trim() || lines.length !== total) throw new Error('Ответ неполный. Повторите проверку.');
      status.textContent += ' Разрешено: '+counts.allow+' · Подтверждение: '+counts.confirm+
        ' · Заблокировано: '+counts.block+' · Ошибок: '+counts.error+'.';
      download.href = URL.createObjectURL(new Blob(lines,{type:'application/x-ndjson'}));
      download.hidden = false;
    } catch (err) {
      status.textContent = 'Проверка не завершена. '+err.message;
    } finally {
      input.disabled = start.disabled = false;
    }
  });
})();
