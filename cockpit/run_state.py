"""一轮共享的执行预算与消费授权；重规划不能重置或复制权限。"""

import json
from copy import deepcopy
from contextlib import contextmanager
from threading import Lock
from time import monotonic
from uuid import uuid4

from cockpit.task_auth import write_grants


class BudgetExceeded(RuntimeError):
    pass


def same_value(left, right):
    if isinstance(left, bool) or isinstance(right, bool): return type(left) is type(right) and left == right
    if type(left) in {int,float} and type(right) in {int,float}: return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_value(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


class RunState:
    def __init__(self, user_text, *, budget_seconds=120, max_model_calls=40, max_replans=2, max_instances=24):
        self.id = uuid4().hex
        self.user_text = user_text
        self.started = monotonic()
        self.deadline = self.started + budget_seconds
        self.max_model_calls, self.max_replans, self.max_instances = max_model_calls, max_replans, max_instances
        self.model_calls, self.replans = 0, 0
        self.task_instances = 0
        self.grants = write_grants(user_text)
        self.network_failed, self.retry_used = set(), set()
        self.write_ledger, self.versions, self.audit = [], [], []
        self.clarification = ''
        self.lock = Lock()
        self.network_locks = {}

    def elapsed_ms(self): return round((monotonic()-self.started)*1000, 2)

    def check(self):
        if monotonic() >= self.deadline: raise BudgetExceeded('本轮调度时间预算已用完，未开始新的调用。')

    def invoke(self, model, messages, *, final=False):
        with self.lock:
            self.check()  # 回答整理也不得在整轮预算耗尽后发起新的网络请求。
            if self.model_calls >= self.max_model_calls: raise BudgetExceeded('本轮模型请求次数已达上限。')
            self.model_calls += 1
        return model.invoke(messages)

    def consume_write(self, tool, arguments, task_id):
        with self.lock:
            for index, grant in enumerate(self.grants):
                if not grant['used'] and grant['tool'] == tool and same_value(grant['arguments'], arguments):
                    grant['used'] = True
                    entry = {'grant_index':index, 'task_id':task_id, 'tool':tool, 'arguments':deepcopy(arguments),
                             'source':grant['source'], 'status':'started'}
                    self.write_ledger.append(entry)
                    return entry
        return None

    @staticmethod
    def fingerprint(tool, arguments):
        return tool + ':' + json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)

    def permit_network_retry(self, tool, arguments):
        key = self.fingerprint(tool, arguments)
        with self.lock:
            if key in self.retry_used: return False
            if key in self.network_failed: self.retry_used.add(key)
            return True

    def network_failure(self, tool, arguments):
        with self.lock: self.network_failed.add(self.fingerprint(tool, arguments))

    @contextmanager
    def network_slot(self, tool, arguments):
        key = self.fingerprint(tool,arguments)
        with self.lock:
            gate = self.network_locks.setdefault(key,Lock())
        with gate:
            yield self.permit_network_retry(tool,arguments)

    def snapshot(self):
        return {'run_id':self.id, 'model_calls':self.model_calls, 'replans':self.replans,
                'task_instances':self.task_instances,
                'versions':deepcopy(self.versions), 'write_ledger':deepcopy(self.write_ledger),
                'audit':deepcopy(self.audit), 'clarification':self.clarification, 'duration_ms':self.elapsed_ms()}
