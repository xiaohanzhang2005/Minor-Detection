"""
序号升序排序脚本
"""
import json
import re

def sort_jsonl_by_dataset_id(input_file, output_file):
    """
    Sorts a .jsonl file based on the numeric part of the 'dataset_id' field.

    Args:
        input_file (str): The path to the input .jsonl file.
        output_file (str): The path to the output .jsonl file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_objects = []
        for line in lines:
            try:
                data_objects.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告: 发现无法解析的行，已跳过: {line.strip()}")
                continue

        # 定义一个函数来从 dataset_id 中提取数字
        def get_id_number(item):
            dataset_id = item.get("dataset_id", "")
            # 兼容 "gemini-3-flash-preview_Seed_XX" 和 "gemini-3-flash-preview_XX" 两种格式
            match = re.search(r'_(\d+)$', dataset_id)
            if not match:
                match = re.search(r'_Seed_(\d+)$', dataset_id)
            
            if match:
                return int(match.group(1))
            return -1 # 如果没有找到数字，则排在最前面

        # 排序和统一格式
        sorted_data = sorted(data_objects, key=get_id_number)
        
        # 在写入前，统一 dataset_id 的格式
        for item in sorted_data:
            id_num = get_id_number(item)
            if id_num != -1:
                model_name_match = re.match(r'[^_]+', item.get("dataset_id", ""))
                model_name = model_name_match.group(0) if model_name_match else "gemini-3-flash-preview"
                item["dataset_id"] = f"{model_name}_{id_num}"

        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sorted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"排序完成，输出文件已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 at {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    input_filename = "semantic_data.jsonl"
    output_filename = "semantic_data_v2.jsonl"
    sort_jsonl_by_dataset_id(input_filename, output_filename)
