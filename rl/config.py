TASKS = [
    {
        "id": "factorial",
        "prompt": "Write a function that computes the factorial of n recursively.",
        "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
        "difficulty": 0.2,
    },
    {
        "id": "fibonacci",
        "prompt": "Write a function that returns the nth Fibonacci number using recursion.",
        "solution": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "tests": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765",
        "difficulty": 0.3,
    },
    {
        "id": "is_palindrome",
        "prompt": "Write a function that checks if a string is a palindrome.",
        "solution": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "tests": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal panama') == True\nassert is_palindrome('') == True",
        "difficulty": 0.3,
    },
    {
        "id": "reverse_string",
        "prompt": "Write a function that reverses a string without using slicing or reversed().",
        "solution": "def reverse_string(s):\n    chars = list(s)\n    left, right = 0, len(chars) - 1\n    while left < right:\n        chars[left], chars[right] = chars[right], chars[left]\n        left += 1\n        right -= 1\n    return ''.join(chars)",
        "tests": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('abc') == 'cba'\nassert reverse_string('a') == 'a'\nassert reverse_string('') == ''",
        "difficulty": 0.4,
    },
    {
        "id": "is_prime",
        "prompt": "Write a function that checks if a number is prime.",
        "solution": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "tests": "assert is_prime(2) == True\nassert is_prime(17) == True\nassert is_prime(4) == False\nassert is_prime(1) == False\nassert is_prime(97) == True",
        "difficulty": 0.4,
    },
    {
        "id": "binary_search",
        "prompt": "Write a binary search function that returns the index of target in sorted array, or -1 if not found.",
        "solution": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "tests": "assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1, 3, 5, 7, 9], 1) == 0\nassert binary_search([1, 3, 5, 7, 9], 10) == -1\nassert binary_search([], 5) == -1",
        "difficulty": 0.5,
    },
    {
        "id": "two_sum",
        "prompt": "Write a function that returns indices of two numbers that add up to a target.",
        "solution": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
        "tests": "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nassert two_sum([3, 2, 4], 6) == [1, 2]\nassert two_sum([3, 3], 6) == [0, 1]",
        "difficulty": 0.5,
    },
    {
        "id": "count_vowels",
        "prompt": "Write a function that counts the number of vowels in a string.",
        "solution": "def count_vowels(s):\n    vowels = set('aeiou')\n    return sum(1 for c in s.lower() if c in vowels)",
        "tests": "assert count_vowels('hello') == 2\nassert count_vowels('sky') == 0\nassert count_vowels('AEIOU') == 5\nassert count_vowels('') == 0",
        "difficulty": 0.2,
    },
    {
        "id": "find_max",
        "prompt": "Write a function that returns the maximum element in a list without using max().",
        "solution": "def find_max(lst):\n    if not lst:\n        return None\n    m = lst[0]\n    for x in lst:\n        if x > m:\n            m = x\n    return m",
        "tests": "assert find_max([1, 5, 3, 9, 2]) == 9\nassert find_max([-5, -2, -10]) == -2\nassert find_max([7]) == 7\nassert find_max([]) is None",
        "difficulty": 0.2,
    },
    {
        "id": "merge_sorted",
        "prompt": "Write a function that merges two sorted lists into one sorted list.",
        "solution": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
        "tests": "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1, 2], []) == [1, 2]\nassert merge_sorted([1], [1]) == [1, 1]",
        "difficulty": 0.5,
    },
]

BUGGY_TASKS = [
    {
        "id": "bug_factorial",
        "prompt": "Fix this factorial function. It returns 0 for all inputs.",
        "buggy_code": "def factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result",
        "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "bug_explanation": "result starts at 0 instead of 1, and multiplication by 0 always yields 0",
        "tests": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
        "difficulty": 0.3,
    },
    {
        "id": "bug_binary_search",
        "prompt": "Fix this binary search function. It never finds the target.",
        "buggy_code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "solution": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "bug_explanation": "while left < right should be while left <= right, otherwise single-element arrays are skipped",
        "tests": "assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1], 1) == 0\nassert binary_search([1, 3, 5], 10) == -1",
        "difficulty": 0.4,
    },
    {
        "id": "bug_is_palindrome",
        "prompt": "Fix this palindrome checker. It raises an error for some inputs.",
        "buggy_code": "def is_palindrome(s):\n    s = s.lower()\n    return s == s.reverse()",
        "solution": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "bug_explanation": "strings don't have a .reverse() method, and spaces aren't stripped",
        "tests": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal panama') == True",
        "difficulty": 0.4,
    },
]

# RL hyperparameters
N_SOLUTIONS_PER_TASK = 3
FAILED_PRIORITY = 3.0
NEAR_MISS_PRIORITY = 1.5
MAX_BUFFER_SIZE = 1000
FT_EPOCHS = 2
FT_LR = 5e-5
FT_BATCH_SIZE = 2
FT_MAX_STEPS = 100
EVAL_TIMEOUT = 5
