
import json
import re
import argparse
from tqdm import tqdm

# 导入 math-verify 相关模块
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
    exit(1)

def extract_boxed_content(text: str) -> str:
    """提取最后一个 \boxed{} 的内容，支持多层嵌套花括号"""
    pattern = r'boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return "No \\boxed{} found"
    
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
    return "Unmatched braces in \\boxed{}"

def compute_score(model_output: str, ground_truth, timeout_score: float = 0) -> float:
    """
    使用 math-verify 计算模型输出与标准答案的匹配分数
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()),
    )
    ret_score = 0.0
    # 🔥 修复：将 ground_truth 转换为字符串
    ground_truth_str = str(ground_truth)
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth_str + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in verification: {e}")
        ret_score = 0.0
    return ret_score


def evaluate_jsonl(file_path: str, output_file: str = None, show_samples: bool = True, export_errors: bool = True):
    """
    读取 JSONL 文件并评估模型回答的正确率
    
    Args:
        file_path: 输入JSONL文件路径
        output_file: 评估结果JSON文件路径
        show_samples: 是否显示样例
        export_errors: 是否导出错误case
    """
    correct_count = 0
    total_count = 0
    error_count = 0
    results = []
    error_cases = []  # 🔥 新增：存储错误case
    
    print(f"Reading file: {file_path}")
    
    # 读取 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total samples: {len(lines)}")
    print("Starting evaluation...")
    
    # 处理每一行数据
    for idx, line in enumerate(tqdm(lines, desc="Evaluating")):
        try:
            data = json.loads(line.strip())
            
            # 获取模型输出和标准答案
            model_response = data.get('model_response', data.get('response', ''))
            ground_truth = data.get('gdt', data.get('ground_truth', data.get('answer', '')))
            
            if not model_response or not ground_truth:
                print(f"Warning: Missing data at line {idx + 1}")
                error_count += 1
                continue
            
            # 计算分数
            score = compute_score(model_response, ground_truth)
            
            # 提取 boxed 内容用于展示
            model_answer = extract_boxed_content(model_response)
            
            # 记录结果
            is_correct = score > 0.5  # math-verify 返回 0 或 1
            if is_correct:
                correct_count += 1
            else:
                # 🔥 新增：保存错误case的完整原始数据
                error_case = data.copy()  # 保留原始数据的所有字段
                error_case['_eval_info'] = {  # 添加评估信息
                    'score': score,
                    'extracted_answer': model_answer,
                    'ground_truth': ground_truth,
                    'line_number': idx + 1
                }
                error_cases.append(error_case)
            
            total_count += 1
            
            results.append({
                'index': idx,
                'score': score,
                'is_correct': is_correct,
                'model_answer': model_answer,
                'ground_truth': ground_truth,
                'original_data': data  # 🔥 保存原始数据
            })
            
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON at line {idx + 1}")
            error_count += 1
        except Exception as e:
            print(f"Error processing line {idx + 1}: {e}")
            error_count += 1
    
    # 计算正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 打印结果统计
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples processed: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Wrong answers: {total_count - correct_count}")
    print(f"Errors/Skipped: {error_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*50)
    
    # 保存详细结果到文件
    if output_file is None:
        output_file = file_path.replace('.jsonl', '_evaluation.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_samples': total_count,
                'correct_count': correct_count,
                'wrong_count': total_count - correct_count,
                'error_count': error_count,
                'accuracy': accuracy
            },
            'details': results[:10]  # 保存前10个样本的详细信息
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # 🔥 新增：导出错误case到JSONL
    if export_errors and error_cases:
        error_file = file_path.replace('.jsonl', '_mathverify_error.jsonl')
        with open(error_file, 'w', encoding='utf-8') as f:
            for case in error_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        print(f"\n🔥 Exported {len(error_cases)} error cases to: {error_file}")
        print(f"   Error rate: {len(error_cases)/total_count*100:.2f}%")
    
    # 显示样例（可选）
    if show_samples:
        # 显示一些正确的案例
        print("\n" + "="*50)
        print("SAMPLE CORRECT PREDICTIONS")
        print("="*50)
        correct_cases = [r for r in results if r['is_correct']][:3]
        for i, case in enumerate(correct_cases, 1):
            print(f"\nCorrect Case {i}:")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  Model Answer: {case['model_answer']}")
        
        # 显示一些错误的案例
        print("\n" + "="*50)
        print("SAMPLE INCORRECT PREDICTIONS")
        print("="*50)
        wrong_cases = [r for r in results if not r['is_correct']][:3]
        for i, case in enumerate(wrong_cases, 1):
            print(f"\nIncorrect Case {i}:")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  Model Answer: {case['model_answer']}")
        
        # 显示答案分布
        print("\n" + "="*50)
        print("ANSWER DISTRIBUTION (First 20 samples)")
        print("="*50)
        for i, result in enumerate(results[:20], 1):
            status = "✓" if result['is_correct'] else "✗"
            print(f"{i:3d}. [{status}] Model: {result['model_answer'][:50]:<50} | GT: {result['ground_truth'][:50]}")
    
    return accuracy, results, error_cases  # 🔥 返回error_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate math problem solving results')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to input JSONL file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output evaluation JSON file (default: input_file_evaluation.json)')
    parser.add_argument('--no-samples', action='store_true',
                        help='Do not show sample predictions')
    parser.add_argument('--no-export-errors', action='store_true',  # 🔥 新增参数
                        help='Do not export error cases to JSONL')
    
    args = parser.parse_args()
    
    # 执行评估
    accuracy, results, error_cases = evaluate_jsonl(
        args.input, 
        output_file=args.output,
        show_samples=not args.no_samples,
        export_errors=not args.no_export_errors  # 🔥 传递参数
    )
    
    # 🔥 新增：显示错误case统计
    if error_cases:
        print(f"\n📊 Error cases exported: {len(error_cases)}")
        print(f"   File: {args.input.replace('.jsonl', '_mathverify_error.jsonl')}")
