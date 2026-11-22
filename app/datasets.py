"""Read official dataset containers as data and return evaluator-compatible JSONL."""
import csv
import io
import json
import time
import zipfile
import zlib
from pathlib import PurePosixPath

from fastapi import HTTPException

from .contest import ContestRequest, inspect_request

MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 10000
csv.field_size_limit(MAX_BYTES)  # A serialized context can contain several attachments.


def unzip_dataset(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            files = [i for i in archive.infolist() if not i.is_dir() and
                     PurePosixPath(i.filename).suffix.lower() in {'.jsonl', '.csv'}]
            # The official archive includes two representations of the same dataset.
            if len(files) == 2 and len({PurePosixPath(i.filename).with_suffix('') for i in files}) == 1:
                files = [i for i in files if i.filename.lower().endswith('.jsonl')]
            if len(files) != 1:
                raise HTTPException(422, 'В ZIP нужен один набор JSONL или CSV.')
            entry = files[0]
            if entry.file_size > MAX_BYTES:
                raise HTTPException(413, 'Распакованный набор превышает 32 МиБ.')
            with archive.open(entry) as stream:
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, 'Распакованный набор превышает 32 МиБ.')
            return data, PurePosixPath(entry.filename).suffix.lower()[1:]
    except (zipfile.BadZipFile, RuntimeError, EOFError, NotImplementedError, zlib.error, ValueError):
        raise HTTPException(422, 'Не удалось прочитать ZIP-архив.')


def parse_dataset(raw, fmt, report=False):
    if fmt == 'zip':
        raw, fmt = unzip_dataset(raw)
    try:
        stream = io.StringIO(raw.decode('utf-8-sig'), newline='')
    except UnicodeError:
        raise HTTPException(422, 'Нужен файл UTF-8.')
    if fmt == 'csv':
        reader = csv.DictReader(stream, strict=True)
        required = {'id','user_prompt','context'}
        try:
            headers = reader.fieldnames or []
        except csv.Error:
            raise HTTPException(422, 'Некорректный заголовок CSV.')
        if not required.issubset(headers) or len(headers) != len(set(headers)):
            raise HTTPException(422, 'Проверьте заголовок CSV: обязательны id, user_prompt и context.')
    else:
        reader = stream
    packets = []
    seen = set()
    index = 0
    try:
        for index, item in enumerate(reader, start=2 if fmt=='csv' else 1):
            if fmt == 'jsonl' and not item.strip():
                continue
            if len(packets) >= MAX_ROWS:
                raise HTTPException(413, 'В одном наборе допускается до 10 000 примеров.')
            row = json.loads(item) if fmt == 'jsonl' else item
            if not isinstance(row, dict):
                raise ValueError
            rid = row['id']
            if not isinstance(rid, str) or not rid.strip() or len(rid)>256 or rid in seen:
                raise ValueError
            context = json.loads(row['context']) if fmt == 'csv' else row['context']
            req = ContestRequest(user_prompt=row['user_prompt'], context=context)
            seen.add(rid)
            reference = None
            if report:
                reference = {key: row[key] for key in ('expected_decision', 'risk_level', 'risk_category')
                             if isinstance(row.get(key), str) and row[key].strip()}
            packets.append((rid, req, reference))
    except (ValueError, KeyError, TypeError, csv.Error, RecursionError):
        raise HTTPException(422, f'Запись {index}: проверьте уникальный id, текст и структуру context.')
    if not packets:
        raise HTTPException(422, 'В наборе нет примеров.')
    return packets


def predictions(packets, raw_policy, thresholds):
    for rid, req, reference in packets:
        start = time.perf_counter()
        try:
            row = inspect_request(req, thresholds, raw_policy)
        except Exception:
            # Invalid per-row predictions must remain invalid for the official evaluator.
            row = {'_error':'inference_failed'}
        row.update(id=rid, latency_ms=round((time.perf_counter()-start)*1000, 3))
        if reference is not None:
            row['reference'] = reference
        yield json.dumps(row,ensure_ascii=False)+'\n'
