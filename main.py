import re


def main(arg1):
    pattern = (
        r"【判断是否为极端客诉】结果：(\d+).*?"
        r"【判断是否为多轮纠缠】结果：(\d+).*?"
        r"【判断是否为用户表达不懂】结果：(\d+).*?"
        r"【判断是否为人工诉求】结果：(\d+)"
    )
    match = re.search(pattern, arg1, re.DOTALL)

    # 默认结果
    result = "NONE"

    if match:
        extreme_complaint, multi_round, user_confused, human_request = match.groups()

        extreme_complaint = int(extreme_complaint)
        multi_round = int(multi_round)
        user_confused = int(user_confused)
        human_request = int(human_request)

        if extreme_complaint == 1 or user_confused == 1 or human_request == 1:
            return "STRONG_RE"

        if multi_round == 1:
            return "WEAK"

        # 如果以上条件都不满足，返回默认的 'NONE'
        return result
    else:
        # 如果没有匹配到，返回 'NONE'
        return "NONE"
