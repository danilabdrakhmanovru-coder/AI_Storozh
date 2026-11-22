"""Durable request lifecycle, review and reports. Legacy log table is retained."""
import json
import sqlite3
import threading
import time
from collections import Counter

class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS contest_policy_events (author TEXT NOT NULL, ts REAL NOT NULL, version TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS contest_checks (
                id TEXT PRIMARY KEY, ts REAL NOT NULL, owner TEXT NOT NULL, role TEXT NOT NULL,
                decision TEXT NOT NULL, category TEXT NOT NULL, latency_ms REAL NOT NULL,
                policy_version TEXT NOT NULL, fingerprint TEXT NOT NULL, state TEXT NOT NULL,
                received_at REAL);
            CREATE INDEX IF NOT EXISTS contest_owner_time ON contest_checks(owner, ts);
            CREATE TABLE IF NOT EXISTS requests (
              id TEXT PRIMARY KEY, ts REAL NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL,
              source TEXT NOT NULL, text TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL,
              decision TEXT NOT NULL, state TEXT NOT NULL, review TEXT, feedback TEXT,
              dispatch TEXT, expires REAL NOT NULL, policy_version TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS request_time ON requests(ts);
            CREATE INDEX IF NOT EXISTS request_user ON requests(user_id,ts);
        ''')
        self.db.commit()

    def add(self, payload, result, version):
        with self.lock, self.db:
            self.db.execute('INSERT INTO requests(id,ts,user_id,role,source,text,payload,result,decision,state,expires,policy_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                            (result['request_id'], time.time(), payload['user_id'], payload['role'], payload['source'], payload['text'],
                             json.dumps(payload, ensure_ascii=False), json.dumps(result, ensure_ascii=False), result['decision'],
                             {'allow':'ready','confirm':'pending','block':'blocked'}[result['decision']], time.time()+3600, version))

    def decode(self, row):
        if row is None: return None
        d = dict(row)
        for k in ['payload','result','review','feedback','dispatch']:
            d[k] = json.loads(d[k]) if d[k] else None
        d['expired'] = d['expires'] < time.time()
        return d

    def get(self, rid):
        with self.lock:
            return self.decode(self.db.execute('SELECT * FROM requests WHERE id=?', (rid,)).fetchone())

    def rows(self, start=0, end=None, decision=None, limit=None, offset=0):
        sql = 'SELECT * FROM requests WHERE ts>=? AND ts<?'
        args = [start, end if end is not None else time.time()+1]
        if decision:
            sql += ' AND decision=?'; args.append(decision)
        sql += ' ORDER BY ts DESC'
        if limit is not None:
            sql += ' LIMIT ? OFFSET ?'; args += [limit,offset]
        with self.lock:
            return [self.decode(r) for r in self.db.execute(sql, args).fetchall()]

    def blocks(self, uid):
        with self.lock:
            return self.db.execute("SELECT count(*) FROM requests WHERE user_id=? AND decision='block' AND ts>?", (uid,time.time()-600)).fetchone()[0]

    def review_request(self, rid, approve, actor):
        with self.lock, self.db:
            r = self.get(rid)
            if not r: raise ValueError('Запрос не найден')
            if r['state'] != 'pending' or r['expired']: raise ValueError('Запрос уже обработан или срок подтверждения истёк')
            if r['result']['model']['status'] == 'error' or 'internal_error' in r['result']['categories']:
                raise ValueError('После сбоя нужна повторная проверка; разрешить запрос нельзя')
            review = {'approve': approve, 'actor': actor, 'ts': time.time()}
            self.db.execute('UPDATE requests SET state=?,review=? WHERE id=?',
                            ('approved' if approve else 'rejected',json.dumps(review),rid))
        return self.get(rid)

    def claim(self, rid, uid, admin, version):
        with self.lock, self.db:
            r = self.get(rid)
            if not r or (not admin and r['user_id'] != uid): raise ValueError('Запрос недоступен')
            if r['state'] not in ['ready','approved'] or r['expired']: raise ValueError('Передача запрещена, уже выполнена или срок истёк')
            if r['policy_version'] != version: raise ValueError('Политика изменилась. Проверьте запрос заново')
            if r['payload']['tools']: raise ValueError('Демонстрационный агент не выполняет инструменты. Используйте интеграционный API')
            self.db.execute("UPDATE requests SET state='dispatching' WHERE id=?", (rid,))
        return r

    def dispatched(self, rid, answer=None):
        with self.lock, self.db:
            self.db.execute('UPDATE requests SET state=?,dispatch=? WHERE id=?',
                            ('delivered' if answer is not None else 'delivery_error', json.dumps({'answer':answer,'ts':time.time()},ensure_ascii=False),rid))

    def feedback(self, rid, label, note):
        with self.lock, self.db:
            if not self.get(rid): raise ValueError('Запрос не найден')
            self.db.execute('UPDATE requests SET feedback=? WHERE id=?', (json.dumps({'label':label,'note':note,'ts':time.time()},ensure_ascii=False),rid))

    def report(self, start, end):
        rows = self.rows(start,end)
        decisions = Counter(r['decision'] for r in rows)
        categories = Counter(c for r in rows for c in set(r['result']['categories']))
        errors = sum(r['result']['model']['status']=='error' or 'internal_error' in r['result']['categories'] for r in rows)
        feedback = Counter(r['feedback']['label'] for r in rows if r['feedback'])
        latency = sorted(r['result']['latency_ms'] for r in rows)
        daily = {}
        for r in rows:
            day = time.strftime('%Y-%m-%d',time.gmtime(r['ts']))
            daily.setdefault(day,Counter())[r['decision']] += 1
        total=len(rows)
        return {'start':start,'end':end,'total':total,'decisions':dict(decisions),'categories':dict(categories.most_common()),
                'technical_errors':errors,'feedback':dict(feedback),'reviewed':sum(bool(r['review']) for r in rows),
                'pending':sum(r['state']=='pending' and not r['expired'] for r in rows),
                'delivered':sum(r['state']=='delivered' for r in rows),
                'p95_ms':latency[min(len(latency)-1,int(len(latency)*.95))] if latency else 0,
                'daily':[{'date':d,**dict(v)} for d,v in sorted(daily.items())],
                'summary':f'Проверено {total} запросов. Разрешено: {decisions["allow"]}, заблокировано: {decisions["block"]}, направлено на подтверждение: {decisions["confirm"]}. Технических сбоев проверки: {errors}.',
                'quality_note':'Ошибки классификации отмечаются оператором. Без разметки нельзя оценить точность или считать все блокировки атаками.'}
