from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class CodingTask:
    id: str
    category: str
    prompt: str
    solution: str = ""
    test_code: str = ""
    expected_output: str = ""
    timeout: int = 5
    metadata: dict = field(default_factory=dict)


ALGORITHM_TASKS = [
    CodingTask(
        id="fibonacci",
        category="algorithms",
        prompt="Write a function fibonacci(n) that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
        solution="""
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
""",
        test_code="""def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
assert fibonacci(20) == 6765
print("fibonacci: PASS")
""",
        expected_output="fibonacci: PASS\n",
    ),
    CodingTask(
        id="factorial",
        category="algorithms",
        prompt="Write a function factorial(n) that returns n! (n factorial).",
        test_code="""def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

assert factorial(0) == 1
assert factorial(5) == 120
assert factorial(10) == 3628800
print("factorial: PASS")
""",
        expected_output="factorial: PASS\n",
    ),
    CodingTask(
        id="is_prime",
        category="algorithms",
        prompt="Write a function is_prime(n) that returns True if n is prime, False otherwise.",
        test_code="""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

assert is_prime(2) == True
assert is_prime(17) == True
assert is_prime(1) == False
assert is_prime(4) == False
assert is_prime(97) == True
print("is_prime: PASS")
""",
        expected_output="is_prime: PASS\n",
    ),
    CodingTask(
        id="binary_search",
        category="algorithms",
        prompt="Write a function binary_search(arr, target) that returns the index of target in sorted arr, or -1 if not found.",
        test_code="""def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

assert binary_search([1, 2, 3, 4, 5], 3) == 2
assert binary_search([1, 2, 3, 4, 5], 6) == -1
assert binary_search([], 1) == -1
print("binary_search: PASS")
""",
        expected_output="binary_search: PASS\n",
    ),
    CodingTask(
        id="merge_sort",
        category="algorithms",
        prompt="Write a function merge_sort(arr) that returns a sorted copy of the list using merge sort.",
        test_code="""def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

assert merge_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
assert merge_sort([]) == []
assert merge_sort([1]) == [1]
print("merge_sort: PASS")
""",
        expected_output="merge_sort: PASS\n",
    ),
    CodingTask(
        id="dfs_traversal",
        category="algorithms",
        prompt="Write a function dfs(graph, start) that performs DFS traversal and returns visited nodes in order. graph is an adjacency dict.",
        test_code="""def dfs(graph, start):
    visited = []
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            visited.append(node)
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in seen:
                    stack.append(neighbor)
    return visited

graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
result = dfs(graph, 'A')
assert result == ['A', 'B', 'D', 'C', 'E'] or result == ['A', 'C', 'E', 'B', 'D']
print("dfs_traversal: PASS")
""",
        expected_output="dfs_traversal: PASS\n",
    ),
    CodingTask(
        id="bfs_traversal",
        category="algorithms",
        prompt="Write a function bfs(graph, start) that performs BFS traversal and returns visited nodes in order. graph is an adjacency dict.",
        test_code="""from collections import deque

def bfs(graph, start):
    visited = []
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        visited.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return visited

graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
result = bfs(graph, 'A')
assert result == ['A', 'B', 'C', 'D', 'E']
print("bfs_traversal: PASS")
""",
        expected_output="bfs_traversal: PASS\n",
    ),
]


DATA_STRUCTURE_TASKS = [
    CodingTask(
        id="stack_implementation",
        category="data_structures",
        prompt="Implement a Stack class with push, pop, peek, and is_empty methods.",
        test_code="""class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    def is_empty(self):
        return len(self.items) == 0

s = Stack()
assert s.is_empty() == True
s.push(1)
s.push(2)
assert s.pop() == 2
assert s.peek() == 1
assert s.is_empty() == False
print("stack: PASS")
""",
        expected_output="stack: PASS\n",
    ),
    CodingTask(
        id="queue_implementation",
        category="data_structures",
        prompt="Implement a Queue class with enqueue, dequeue, peek, and is_empty methods.",
        test_code="""from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    def enqueue(self, item):
        self.items.append(item)
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
    def peek(self):
        if not self.is_empty():
            return self.items[0]
    def is_empty(self):
        return len(self.items) == 0

q = Queue()
assert q.is_empty() == True
q.enqueue(1)
q.enqueue(2)
assert q.dequeue() == 1
assert q.peek() == 2
print("queue: PASS")
""",
        expected_output="queue: PASS\n",
    ),
    CodingTask(
        id="linked_list",
        category="data_structures",
        prompt="Implement a LinkedList class with append, prepend, delete, and display methods (returns list of values).",
        test_code="""class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    def append(self, val):
        if not self.head:
            self.head = Node(val)
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = Node(val)
    def prepend(self, val):
        n = Node(val)
        n.next = self.head
        self.head = n
    def delete(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        cur = self.head
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
                return
            cur = cur.next
    def display(self):
        result = []
        cur = self.head
        while cur:
            result.append(cur.val)
            cur = cur.next
        return result

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.prepend(0)
assert ll.display() == [0, 1, 2]
ll.delete(1)
assert ll.display() == [0, 2]
print("linked_list: PASS")
""",
        expected_output="linked_list: PASS\n",
    ),
    CodingTask(
        id="hash_map",
        category="data_structures",
        prompt="Implement a SimpleHashMap class with put, get, and delete methods.",
        test_code="""class SimpleHashMap:
    def __init__(self):
        self.data = {}
    def put(self, key, val):
        self.data[key] = val
    def get(self, key):
        return self.data.get(key, None)
    def delete(self, key):
        if key in self.data:
            del self.data[key]

m = SimpleHashMap()
m.put('a', 1)
m.put('b', 2)
assert m.get('a') == 1
assert m.get('c') is None
m.delete('a')
assert m.get('a') is None
print("hash_map: PASS")
""",
        expected_output="hash_map: PASS\n",
    ),
]


PROGRAMMING_TASKS = [
    CodingTask(
        id="file_read_write",
        category="programming",
        prompt="Write a function copy_file(src, dst) that reads src and writes its contents to dst.",
        test_code="""def copy_file(src, dst):
    with open(src) as f:
        content = f.read()
    with open(dst, 'w') as f:
        f.write(content)

import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('hello world')
    src = f.name
dst = src + '.copy'
copy_file(src, dst)
with open(dst) as f:
    assert f.read() == 'hello world'
os.unlink(src)
os.unlink(dst)
print("file_read_write: PASS")
""",
        expected_output="file_read_write: PASS\n",
    ),
    CodingTask(
        id="json_parsing",
        category="programming",
        prompt="Write a function parse_json_string(s) that parses a JSON string and returns the resulting dict. Handle errors gracefully.",
        test_code="""import json

def parse_json_string(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}

assert parse_json_string('{"a": 1}') == {"a": 1}
assert parse_json_string('invalid') == {}
print("json_parsing: PASS")
""",
        expected_output="json_parsing: PASS\n",
    ),
    CodingTask(
        id="regex_extraction",
        category="programming",
        prompt="Write a function extract_emails(text) that returns all email addresses found in the text.",
        test_code="""import re

def extract_emails(text):
    return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.]+', text)

result = extract_emails('Contact us at support@example.com or sales@test.org')
assert 'support@example.com' in result
assert 'sales@test.org' in result
print("regex_extraction: PASS")
""",
        expected_output="regex_extraction: PASS\n",
    ),
    CodingTask(
        id="string_parsing",
        category="programming",
        prompt="Write a function parse_csv_line(line) that parses a CSV line and returns a list of fields.",
        test_code="""def parse_csv_line(line):
    return [f.strip() for f in line.split(',')]

assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
assert parse_csv_line('hello, world, test') == ['hello', 'world', 'test']
assert parse_csv_line('single') == ['single']
print("string_parsing: PASS")
""",
        expected_output="string_parsing: PASS\n",
    ),
    CodingTask(
        id="class_design",
        category="programming",
        prompt="Design a BankAccount class with deposit, withdraw, and get_balance methods. Prevent overdraft.",
        test_code="""class BankAccount:
    def __init__(self, initial=0):
        self.balance = initial
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    def get_balance(self):
        return self.balance

acct = BankAccount(100)
acct.deposit(50)
assert acct.get_balance() == 150
assert acct.withdraw(30) == True
assert acct.get_balance() == 120
assert acct.withdraw(200) == False
print("class_design: PASS")
""",
        expected_output="class_design: PASS\n",
    ),
]


DEBUGGING_TASKS = [
    CodingTask(
        id="fix_broken_function",
        category="debugging",
        prompt="The following function should return the sum of a list. Fix the bugs:\n\ndef buggy_sum(arr):\n    total = 0\n    for i in len(arr):\n        total = total + i\n    return total",
        test_code="""def buggy_sum(arr):
    total = 0
    for i in arr:
        total = total + i
    return total

assert buggy_sum([1, 2, 3]) == 6
assert buggy_sum([]) == 0
assert buggy_sum([-1, 1]) == 0
print("fix_broken_function: PASS")
""",
        expected_output="fix_broken_function: PASS\n",
    ),
    CodingTask(
        id="fix_syntax_error",
        category="debugging",
        prompt="Fix the syntax error in this function:\n\ndef greet(name):\n    print('Hello' name)\n",
        test_code="""def greet(name):
    print(f'Hello {name}')

assert greet('World') is None
print("fix_syntax_error: PASS")
""",
        expected_output="fix_syntax_error: PASS\n",
    ),
    CodingTask(
        id="fix_logic_bug",
        category="debugging",
        prompt="This function should return True if the list is sorted ascending. Fix the logic:\n\ndef is_sorted(arr):\n    for i in range(len(arr)):\n        if arr[i] > arr[i + 1]:\n            return False\n    return True",
        test_code="""def is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

assert is_sorted([1, 2, 3]) == True
assert is_sorted([3, 1, 2]) == False
assert is_sorted([]) == True
assert is_sorted([1]) == True
print("fix_logic_bug: PASS")
""",
        expected_output="fix_logic_bug: PASS\n",
    ),
    CodingTask(
        id="refactor_code",
        category="debugging",
        prompt="Refactor this messy code into a clean, readable function:\n\ndef f(x):\n    a=[]\n    for i in range(x):\n        if i%2==0:\n            a.append(i)\n    return a",
        test_code="""def f(x):
    return [i for i in range(x) if i % 2 == 0]

assert f(10) == [0, 2, 4, 6, 8]
assert f(0) == []
print("refactor_code: PASS")
""",
        expected_output="refactor_code: PASS\n",
    ),
]


PROJECT_TASKS = [
    CodingTask(
        id="cli_calculator",
        category="projects",
        prompt="Write a calculator function that takes a string like '2 + 3' and returns the result. Support +, -, *, /.",
        test_code="""def calculator(expr):
    parts = expr.split()
    if len(parts) != 3:
        return None
    a, op, b = float(parts[0]), parts[1], float(parts[2])
    if op == '+': return a + b
    elif op == '-': return a - b
    elif op == '*': return a * b
    elif op == '/': return a / b if b != 0 else None
    return None

assert calculator('2 + 3') == 5.0
assert calculator('10 * 5') == 50.0
assert calculator('10 / 2') == 5.0
print("cli_calculator: PASS")
""",
        expected_output="cli_calculator: PASS\n",
    ),
    CodingTask(
        id="todo_list",
        category="projects",
        prompt="Implement a TodoList class with add_task, complete_task, and list_tasks methods.",
        test_code="""class TodoList:
    def __init__(self):
        self.tasks = []
    def add_task(self, task):
        self.tasks.append({'task': task, 'done': False})
    def complete_task(self, idx):
        if 0 <= idx < len(self.tasks):
            self.tasks[idx]['done'] = True
    def list_tasks(self):
        return [t['task'] for t in self.tasks if not t['done']]

t = TodoList()
t.add_task('buy milk')
t.add_task('write code')
assert len(t.list_tasks()) == 2
t.complete_task(0)
assert len(t.list_tasks()) == 1
assert t.list_tasks()[0] == 'write code'
print("todo_list: PASS")
""",
        expected_output="todo_list: PASS\n",
    ),
    CodingTask(
        id="log_parser",
        category="projects",
        prompt="Write a function parse_logs(log_text) that counts log levels (INFO, WARN, ERROR) and returns a dict.",
        test_code="""def parse_logs(log_text):
    counts = {'INFO': 0, 'WARN': 0, 'ERROR': 0}
    for line in log_text.split('\\n'):
        for level in counts:
            if level in line:
                counts[level] += 1
    return counts

logs = 'INFO: started\\nERROR: failed\\nINFO: done'
assert parse_logs(logs) == {'INFO': 2, 'WARN': 0, 'ERROR': 1}
print("log_parser: PASS")
""",
        expected_output="log_parser: PASS\n",
    ),
]


ALL_TASKS_BY_CATEGORY = {
    "algorithms": ALGORITHM_TASKS,
    "data_structures": DATA_STRUCTURE_TASKS,
    "programming": PROGRAMMING_TASKS,
    "debugging": DEBUGGING_TASKS,
    "projects": PROJECT_TASKS,
}

ALL_TASKS = ALGORITHM_TASKS + DATA_STRUCTURE_TASKS + PROGRAMMING_TASKS + DEBUGGING_TASKS + PROJECT_TASKS
TASK_IDS = {t.id: t for t in ALL_TASKS}


def get_task(task_id: str) -> Optional[CodingTask]:
    return TASK_IDS.get(task_id)


def get_tasks_by_category(category: str) -> list[CodingTask]:
    return ALL_TASKS_BY_CATEGORY.get(category, [])


def get_categories() -> list[str]:
    return list(ALL_TASKS_BY_CATEGORY.keys())
