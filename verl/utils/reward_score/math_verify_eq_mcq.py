import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def extract_boxed_content(text: str) -> str:
    """提取最后一个 \\boxed{} 的内容，支持多层嵌套花括号"""
    pattern = r'boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return ""
    
    last_match = matches[-1]
    start_pos = last_match.end()
    
    brace_count = 1
    i = start_pos
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        return text[start_pos:i-1].strip()
    return ""


def extract_choice_letter(text: str) -> str:
    """
    从文本中提取选择题的选项字母
    返回空字符串表示不是选择题格式
    """
    if not text:
        return ""
    
    # 清理 LaTeX 命令
    cleaned = text
    cleaned = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\text[a-z]*\s*\{([^}]*)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', cleaned)
    cleaned = cleaned.strip()
    
    # 模式1: 开头是单个字母 A-Z（后跟 . 、 : 空格等或结尾）
    match = re.match(r'^([A-Za-z])(?:[.\s、:）)\]]|$)', cleaned)
    if match:
        return match.group(1).upper()
    
    # 模式2: 括号包围的字母 (A) 或 （A）
    match = re.match(r'^[（(]\s*([A-Za-z])\s*[）)]', cleaned)
    if match:
        return match.group(1).upper()
    
    # 模式3: 整个文本就是单个字母
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()
    
    return ""


def normalize_model_output(model_output: str) -> str:
    """
    标准化模型输出
    如果 boxed 内容是选择题格式，将其替换为纯字母形式
    """
    boxed_content = extract_boxed_content(model_output)
    if not boxed_content:
        return model_output
    
    # 尝试提取选择题字母
    choice_letter = extract_choice_letter(boxed_content)
    if choice_letter:
        # 是选择题格式，替换 boxed 内容为纯字母
        return model_output.replace(
            boxed_content, 
            choice_letter
        )
    
    # 不是选择题格式，保持原样
    return model_output


def compute_score(model_output: str, ground_truth: str, equivalent_answers: list = None, timeout_score: float = 0) -> float:
    """
    计算模型输出与标准答案的匹配分数
    
    Args:
        model_output: 模型输出的答案
        ground_truth: 主标准答案
        equivalent_answers: 等价答案列表（可选）
        timeout_score: 超时时返回的分数
    
    Returns:
        匹配分数 (0.0 或 1.0)
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()),
    )
    
    # 处理占位符
    if equivalent_answers and "__EMPTY__" in equivalent_answers:
        equivalent_answers = None
    
    # 标准化模型输出（处理选择题格式）
    normalized_output = normalize_model_output(model_output)
    
    # 构建所有可能的正确答案列表
    all_valid_answers = [ground_truth]
    if equivalent_answers:
        all_valid_answers.extend(equivalent_answers)
    
    # 依次尝试匹配每个正确答案
    for answer in all_valid_answers:
        try:
            answer_boxed = "\\boxed{" + str(answer) + "}"
            score, _ = verify_func([answer_boxed], [normalized_output])
            if score > 0:
                return 1.0
        except TimeoutException:
            return timeout_score
        except Exception:
            continue
    
    return 0.0
