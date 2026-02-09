import re


def main(arg1):
    pattern = r"【判断是否为(极端客诉|多轮纠缠|用户表达不懂|人工诉求)】结果：(\d+)"
    matches = re.findall(pattern, arg1)

    if not matches:
        return "NONE"

    results = {label: int(value) for label, value in matches}

    strong_labels = ("极端客诉", "用户表达不懂", "人工诉求")
    if any(results.get(label, 0) == 1 for label in strong_labels):
        return "STRONG_RE"

    if results.get("多轮纠缠", 0) == 1:
        return "WEAK"

    return "NONE"
