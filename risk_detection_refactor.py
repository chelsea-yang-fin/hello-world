import argparse
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix

# -----------------------------
# User Config (edit here once)
# -----------------------------
DEFAULT_INPUT_PATH = (
    "/apdcephfs/ycx4/apdcephfs_nj7/share_303360414/chunxueyang/Reflection/Test_sample.xlsx"
)
DEFAULT_SHEET_NAME = "评测集_0915"
DEFAULT_OUTPUT_PATH_TEMPLATE = (
    "/apdcephfs/share/apdcephfs_nj7/share_303360414/apdcephfs_nj7/share_303360414/"
    "chunxueyang/Reflection/risk_detection_results_v{version}.xlsx"
)
DEFAULT_MODEL_NAME = "Qwen2.5-72B-Instruct"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_WORKERS = 5
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_QPS = None
DEFAULT_NAMESPACE = "Development"
DEFAULT_SERVICE_NAME = "trpc.fdd_llm.llm_model.HttpService"
DEFAULT_URL = None
DEFAULT_STRICT_LABELS = False

# -----------------------------
# LLM Integration Setup
# -----------------------------
try:
    from polaris.api.consumer import GetOneInstanceRequest, create_consumer_by_default_config_file

    POLARIS_AVAILABLE = True
except ImportError:
    print(
        "Warning: polaris modules not found. LLM calls will not work unless you provide a direct URL."
    )
    POLARIS_AVAILABLE = False


PROMPT_TEMPLATE = """

现在你是一位专业的对话风险评估专家，你需要完成3个风险类别的判断。请注意输入内容的格式：**【前序对话】中包含了"用户:"和"客服:"的混合内容，而【当前对话】**中仅包含当前这一轮用户的发言。

请你根据我提供的对话内容，参考推理过程指导，按照要求进行推理判断，再按照最后要求的格式输出。

【推理过程指导：判断是否为极端客诉】
第一步：整合用户全部发言

1.从【前序对话】中，提取所有以"用户:"开头的行。

2.将上一步提取出的历史发言与【当前对话】的内容合并，形成一个完整的"用户发言列表"。

第二步：检查关键词

1.检查整个"用户发言列表"中是否包含骂人如“垃圾”、投诉、骗子、骗人、虚假宣传、找领导、曝光、银监会、12378、退钱、赔偿、精神损失等关键词。

2.只要列表中任何一处用户的发言包含上述关键词，结果即为1，否则为0。

【推理过程指导：判断是否涉及内容安全风险】
第一步：整合用户全部发言

1. 从【前序对话】中，提取所有以"用户:"开头的行。

2.将上一步提取出的历史发言与【当前对话】的内容合并，形成一个完整的"用户发言列表"。

第二步：检查关键词

1. 检查整个"用户发言列表"中是否包含洗钱、套现、虚假、诈骗、转钱、打款、安全账户、验证资金等关键词。

2. 只要列表中任何一处用户的发言包含上述关键词，结果即为1，否则为0。

【推理过程指导：判断是否为多轮纠缠】
第一步：检查前置条件

1. 统计【前序对话】中以"用户:"开头的行数，记为 N。

2. 用户的总提问轮次 T = N + 1 (其中+1代表【当前对话】)。

3. 如果 T < 3，则不满足多轮纠缠的最低轮数要求，直接判断结果为 0，推理结束。

第二步：判断当前问题与前序问题的重复次数

1. 识别【当前对话】内容的语义核心。这一步是为了理解用户当前的核心意图。

2. 遍历【前序对话】中所有以"用户:"开头的行，以理解历史对话上下文，并统计有多少行的语义核心与【当前对话】的语义核心相同或高度相似，记为 C。

3. 计算当前这个问题的总重复次数 R = C + 1。

4. 如果 R >= 3，即用户就当前这个问题（或高度相似的问题）累计提问了至少3次，则结果为 1。否则，结果为 0。

【最终输出格式要求】
#part1 推理过程
【判断是否为极端客诉】推理过程：xxx
【判断是否涉及内容安全风险】推理过程：xxx
【判断是否为多轮纠缠】推理过程：xxx
#part2 最终结果，三个判断结果都只用输出1个数字，不需要输出额外的内容
【判断是否为极端客诉】结果：x
【判断是否涉及内容安全风险】结果：x
【判断是否为多轮纠缠】结果：x

前序对话：
'''
{history}
'''

当前对话：
'''
{current_dialog}
'''

"""


@dataclass
class AppConfig:
    input_path: str = DEFAULT_INPUT_PATH
    sheet_name: str = DEFAULT_SHEET_NAME
    output_path_template: str = DEFAULT_OUTPUT_PATH_TEMPLATE
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_workers: int = DEFAULT_MAX_WORKERS
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    qps: Optional[float] = DEFAULT_QPS
    namespace: str = DEFAULT_NAMESPACE
    service_name: str = DEFAULT_SERVICE_NAME
    url: Optional[str] = DEFAULT_URL
    strict_labels: bool = DEFAULT_STRICT_LABELS


class RateLimiter:
    def __init__(self, qps: Optional[float]) -> None:
        self.qps = qps
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def acquire(self) -> None:
        if not self.qps or self.qps <= 0:
            return
        interval = 1.0 / self.qps
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + interval
        if wait > 0:
            time.sleep(wait)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def test_ip(url: str, timeout: int = 10) -> bool:
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": "你是谁？"}],
        "user": "sherryzxguo",
        "model": "Qwen2.5-72B-Instruct",
        "stream": False,
        "stopTokenIds": [151645],
        "temperature": 0.1,
        "topP": 1,
        "topK": 40,
        "repetitionPenalty": 1,
        "maxTokens": 2048,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(data), timeout=timeout
        )
        if response.status_code == 200:
            logging.info("✅ 已获取可用的北极星ip: %s", url)
            return True
        logging.warning(
            "北极星ip请求失败，状态码：%s, 响应: %s...",
            response.status_code,
            response.text[:100],
        )
        return False
    except requests.exceptions.RequestException as exc:
        logging.warning("北极星ip请求异常: %s", exc)
        return False


def get_polaris_api():
    if not POLARIS_AVAILABLE:
        return None
    try:
        return create_consumer_by_default_config_file()
    except Exception as exc:
        logging.warning("Error creating polaris consumer: %s", exc)
        return None


def get_instances(namespace: str, service_name: str):
    if not POLARIS_AVAILABLE:
        logging.warning("Polaris not available, cannot get instances.")
        return None, None

    api = get_polaris_api()
    if not api:
        logging.warning("Failed to get polaris API instance.")
        return None, None

    for attempt in range(5):
        try:
            request = GetOneInstanceRequest(namespace=namespace, service=service_name)
            instance = api.get_one_instance(request)
            host = instance.get_host()
            port = instance.get_port()
            url = f"http://{host}:{port}/llm/chat"
            if test_ip(url):
                logging.info("✅ 使用 host:%s port:%s", host, port)
                return host, port
        except Exception as exc:
            logging.warning("Attempt %s to get instance failed: %s", attempt + 1, exc)
            time.sleep(1)
    logging.error("❌ 多次尝试获取北极星ip失败！")
    return None, None


class LLMServerParallel:
    def __init__(self, config: AppConfig) -> None:
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.repetition_penalty = config.repetition_penalty
        self.namespace = config.namespace
        self.service_name = config.service_name
        self.timeout = config.request_timeout
        self.max_retries = config.max_retries
        self.backoff_base = config.backoff_base
        self.rate_limiter = RateLimiter(config.qps)
        self.session = requests.Session()

        if config.url and config.url != "null":
            self.url = config.url
            logging.info("✅ 使用直接提供的 URL: %s", self.url)
        else:
            host, port = get_instances(self.namespace, self.service_name)
            if host and port:
                self.url = f"http://{host}:{port}/llm/chat"
            else:
                raise RuntimeError("无法获取有效的 LLM 服务地址 (URL)")

    def chat_completions(self, messages, user="default_user", max_tokens=800) -> Optional[str]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]

        data = {
            "messages": messages,
            "user": user,
            "model": self.model_name,
            "stream": False,
            "stopTokenIds": [151645],
            "temperature": self.temperature,
            "topP": 1,
            "topK": 40,
            "repetitionPenalty": self.repetition_penalty,
            "maxTokens": max_tokens,
        }

        return self._post_with_retries(data)

    def _post_with_retries(self, data: Dict) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                response = self.session.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    try:
                        return response.json()["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, json.JSONDecodeError) as exc:
                        logging.warning(
                            "❌ 解析 LLM 响应失败: %s, 响应内容: %s...",
                            exc,
                            response.text[:200],
                        )
                        return None

                should_retry = response.status_code >= 500 or response.status_code == 429
                logging.warning(
                    "❌ LLM 请求失败, 状态码: %s, 响应: %s...",
                    response.status_code,
                    response.text[:200],
                )
                if not should_retry:
                    return None
            except requests.exceptions.RequestException as exc:
                logging.warning("❌ LLM 请求异常: %s", exc)

            if attempt < self.max_retries:
                backoff = self.backoff_base * (2 ** attempt) + random.random() * 0.1
                time.sleep(backoff)
        return None


def build_prompt(history_text: str, current_dialog: str) -> str:
    return PROMPT_TEMPLATE.format(
        history=history_text if pd.notna(history_text) else "",
        current_dialog=current_dialog if pd.notna(current_dialog) else "",
    )


def call_llm_for_risk_detection(
    history_text: str, current_dialog: str, llm_server: LLMServerParallel, max_tokens: int
) -> Optional[str]:
    prompt = build_prompt(history_text, current_dialog)
    try:
        return llm_server.chat_completions(prompt, max_tokens=max_tokens)
    except Exception as exc:
        logging.warning("LLM调用失败: %s", exc)
        return None


def extract_risk_results(llm_response: Optional[str]) -> Dict[str, int]:
    if not llm_response:
        return {"极端客诉": 0, "内容安全": 0, "多轮纠缠": 0}

    def _extract(pattern: str) -> int:
        match = re.search(pattern, llm_response)
        if not match:
            return 0
        value = match.group(1).strip()
        return 1 if value == "1" else 0

    results = {
        "极端客诉": _extract(r"【判断是否为极端客诉】结果：\s*([01])"),
        "内容安全": _extract(r"【判断是否涉及内容安全风险】结果：\s*([01])"),
        "多轮纠缠": _extract(r"【判断是否为多轮纠缠】结果：\s*([01])"),
    }
    return results


def batch_detect_all_risks(
    df: pd.DataFrame, llm_server: LLMServerParallel, max_workers: int, max_tokens: int
) -> Dict[int, Dict[str, int]]:
    target_indices = df.index.tolist()
    logging.info("🔍 需要检测风险的样本数: %s", len(target_indices))

    risk_results: Dict[int, Dict[str, int]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for idx in target_indices:
            history = df.loc[idx, "prior_query"] if "prior_query" in df.columns else ""
            current_dialog = df.loc[idx, "query"]
            if pd.notna(current_dialog):
                future = executor.submit(
                    call_llm_for_risk_detection, history, current_dialog, llm_server, max_tokens
                )
                future_to_index[future] = idx

        completed = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                response = future.result()
                results = extract_risk_results(response)
                risk_results[idx] = results
                completed += 1
                if completed % 10 == 0:
                    logging.info("✅ 已完成 %s/%s 个样本的风险检测", completed, len(target_indices))
            except Exception as exc:
                logging.warning("❌ 样本 %s 处理失败: %s", idx, exc)
                risk_results[idx] = {"极端客诉": 0, "内容安全": 0, "多轮纠缠": 0}

    return risk_results


def detect_all_risks_with_llm(
    df: pd.DataFrame, llm_server: LLMServerParallel, max_workers: int, max_tokens: int
) -> pd.DataFrame:
    df = df.copy()
    df["pred_极端客诉"] = 0
    df["pred_内容安全"] = 0
    df["pred_多轮纠缠"] = 0

    risk_results = batch_detect_all_risks(df, llm_server, max_workers, max_tokens)
    result_df = pd.DataFrame.from_dict(risk_results, orient="index")
    result_df = result_df.reindex(df.index).fillna(0).astype(int)

    df["pred_极端客诉"] = result_df["极端客诉"]
    df["pred_内容安全"] = result_df["内容安全"]
    df["pred_多轮纠缠"] = result_df["多轮纠缠"]
    return df


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else np.nan
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def evaluate_predictions(df: pd.DataFrame, strict_labels: bool = False) -> None:
    required_columns = ["is_bad", "极端客诉", "内容安全", "多轮纠缠"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.warning("⚠️  警告：缺少真实标签列 %s，无法计算准确率和召回率。", missing_columns)
        if strict_labels:
            raise ValueError("Missing required label columns.")
        return

    print("\n📈 评估指标与混淆矩阵:")

    y_true_bad = df["is_bad"]
    y_pred_bad = df["pred_bad"]

    total_accuracy = (y_pred_bad == y_true_bad).mean()
    print("\n--- 总体 (pred_bad vs is_bad) ---")
    print(f"   - 总体准确率: {total_accuracy:.2%}")

    cm_bad = confusion_matrix(y_true_bad, y_pred_bad, labels=[0, 1])
    print("   - 总体混淆矩阵:")
    cm_bad_df = pd.DataFrame(
        cm_bad,
        index=["实际 Negative (0)", "实际 Positive (1)"],
        columns=["预测 Negative (0)", "预测 Positive (1)"],
    )
    print(cm_bad_df.to_string())

    risk_categories = [
        ("极端客诉", "pred_极端客诉"),
        ("内容安全", "pred_内容安全"),
        ("多轮纠缠", "pred_多轮纠缠"),
    ]

    all_metrics = {}
    for true_col, pred_col in risk_categories:
        print(f"\n--- {true_col} ---")
        metrics = compute_metrics(df[true_col], df[pred_col])
        all_metrics[true_col] = metrics

        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1_score = metrics["f1_score"]

        print(f"   - 准确率: {accuracy:.2%}" if not np.isnan(accuracy) else "   - 准确率: NaN")
        print(
            f"   - 精确率 (Precision): {precision:.2%}"
            if not np.isnan(precision)
            else "   - 精确率 (Precision): NaN"
        )
        print(
            f"   - 召回率 (Recall): {recall:.2%}"
            if not np.isnan(recall)
            else "   - 召回率 (Recall): NaN"
        )
        print(
            f"   - F1 分数: {f1_score:.2%}"
            if not np.isnan(f1_score)
            else "   - F1 分数: NaN"
        )

        cm = confusion_matrix(df[true_col], df[pred_col], labels=[0, 1])
        print("   - 混淆矩阵:")
        cm_df = pd.DataFrame(
            cm,
            index=["实际 Negative (0)", "实际 Positive (1)"],
            columns=["预测 Negative (0)", "预测 Positive (1)"],
        )
        print(cm_df.to_string())

    print("\n📊 主要指标汇总:")
    print(f"   - 总体准确率 (pred_bad == is_bad): {total_accuracy:.2%}")

    print("\n🔍 各风险类别主要指标:")
    for cat, metrics in all_metrics.items():
        valid_metrics = {
            k: f"{v:.2%}" for k, v in metrics.items() if not np.isnan(v)
        }
        recall_str = valid_metrics.get("recall", "NaN")
        print(
            f"   - {cat}：准确率 = {valid_metrics.get('accuracy', 'NaN')}, 召回率 = {recall_str}"
        )


def build_output_path(template: str) -> str:
    version = datetime.now().strftime("%m%d%H%M")
    return template.format(version=version)


def print_summary(df: pd.DataFrame, output_file: str) -> None:
    print("\n✅ 风险检测完成！")
    print(f"   - 总样本数: {len(df)}")
    print(f"   - pred_bad: {df['pred_bad'].sum()} 条")
    print(f"   - pred_极端客诉: {(df['pred_极端客诉'] == 1).sum()} 条")
    print(f"   - pred_内容安全: {(df['pred_内容安全'] == 1).sum()} 条")
    print(f"   - pred_多轮纠缠: {(df['pred_多轮纠缠'] == 1).sum()} 条")
    print(f"   - 输出文件: {output_file}")

    print("\n📊 风险分布统计:")
    print(f"   - pred_极端客诉: {(df['pred_极端客诉'] == 1).sum()} 条")
    print(f"   - pred_内容安全: {(df['pred_内容安全'] == 1).sum()} 条")
    print(f"   - pred_多轮纠缠: {(df['pred_多轮纠缠'] == 1).sum()} 条")

    print("\n🔄 多风险重叠情况:")
    df["预测风险总数"] = df[["pred_极端客诉", "pred_内容安全", "pred_多轮纠缠"]].sum(axis=1)
    overlap_stats = df["预测风险总数"].value_counts().sort_index()
    for risk_count, count in overlap_stats.items():
        if risk_count > 0:
            print(f"   - 同时命中 {risk_count} 个风险类别: {count} 条")


def run_pipeline(config: AppConfig) -> None:
    df = pd.read_excel(config.input_path, sheet_name=config.sheet_name)
    llm_server = LLMServerParallel(config)
    df = detect_all_risks_with_llm(
        df, llm_server, max_workers=config.max_workers, max_tokens=config.max_tokens
    )

    df["pred_bad"] = (
        (df["pred_极端客诉"] == 1)
        | (df["pred_内容安全"] == 1)
        | (df["pred_多轮纠缠"] == 1)
    ).astype(int)

    evaluate_predictions(df, strict_labels=config.strict_labels)

    output_file = build_output_path(config.output_path_template)
    df.to_excel(output_file, index=False)
    print_summary(df, output_file)


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="LLM 风险检测")
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument(
        "--output-path-template",
        default=DEFAULT_OUTPUT_PATH_TEMPLATE,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--backoff-base", type=float, default=DEFAULT_BACKOFF_BASE)
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--strict-labels", action="store_true", default=DEFAULT_STRICT_LABELS)

    args = parser.parse_args()
    return AppConfig(**vars(args))


def main() -> None:
    configure_logging()
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
