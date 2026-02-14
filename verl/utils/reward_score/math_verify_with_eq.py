try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


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
    
    # 构建所有可能的正确答案列表
    all_valid_answers = [ground_truth]
    if equivalent_answers:
        all_valid_answers.extend(equivalent_answers)
    
    # 依次尝试匹配每个正确答案，任意一个匹配成功即返回满分
    for answer in all_valid_answers:
        try:
            answer_boxed = "\\boxed{" + str(answer) + "}"
            score, _ = verify_func([answer_boxed], [model_output])
            if score > 0:
                return 1.0
        except TimeoutException:
            return timeout_score
        except Exception:
            continue
    
    return 0.0
