import json
import re
from num2words import num2words
import os
from unicodedata import normalize
from typing import List, Tuple

# Các ký tự đặc biệt cần tách
SPECIAL_CHARS = r"!?\:;,\"'()\[\]/\.-$&\*"

def replace_numbers_with_words(text):
    """Chuyển đổi chuỗi số thành chữ tiếng Việt."""
    def replace(match):
        try:
            number = float(match.group()) if '.' in match.group() else int(match.group())
            return num2words(number, lang='vi')
        except:
            return match.group()
    return re.sub(r'-?\d+(\.\d+)?', replace, text)

def separate_special_chars(token: str) -> str:
    """Tách các ký tự đặc biệt bằng space."""
    if len(token) <= 1:
        return token
    # Thêm space xung quanh các ký tự đặc biệt
    for char in r"!?:;,\"'()[]/.−$&*":
        token = token.replace(char, f" {char} ")
    return token

def process_single_token(token: str, task: str, preserve_case: bool = False) -> List[str]:
    """
    Xử lý một token: chuẩn hóa, tách ký tự đặc biệt, (tùy chọn) đổi số.
    
    Args:
        token: token cần xử lý
        task: 'text_classification' hoặc 'seq_labeling'
        preserve_case: False = chữ thường, True = giữ nguyên (cho seq_labeling)
    """
    if not preserve_case:
        token = token.lower()
    
    token = normalize("NFKC", token)
    token = separate_special_chars(token)
    sub_tokens = token.strip().split()
    
    # Chỉ chuyển số thành chữ cho classification
    if not preserve_case:
        sub_tokens = [replace_numbers_with_words(sub) for sub in sub_tokens]
        sub_tokens = " ".join(sub_tokens).split()
    
    return sub_tokens

def preprocess_token_list(word_list: List[str], task: str) -> Tuple[List[str], List[int]]:
    """
    Xử lý danh sách các từ: chuẩn hóa, tách ký tự đặc biệt, lập map với token gốc.
    """
    new_words = []
    original_indices = []
    preserve_case = (task == "seq_labeling")
    
    for i, token in enumerate(word_list):
        processed_tokens = process_single_token(token, task, preserve_case=preserve_case)
        new_words.extend(processed_tokens)
        original_indices.extend([i] * len(processed_tokens))
    
    return new_words, original_indices

def convert_numbers_in_file(input_path, output_path, task, text_key, label_key):
    # Đọc file
    try:
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Thử đọc toàn bộ file như dict JSON
            try:
                json_data = json.loads(content)
                # Nếu là dict, lấy values
                if isinstance(json_data, dict):
                    data = list(json_data.values())
                else:
                    # Nếu là list, dùng trực tiếp
                    data = json_data
            except json.JSONDecodeError:
                # Nếu không phải JSON hợp lệ, thử đọc từng dòng
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Lỗi format JSON ở dòng: {line[:50]}...")
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return

    new_data = []
    skipped_count = 0

    for index, sample in enumerate(data):
        if text_key not in sample:
            continue

        raw_words = sample[text_key]
        
        # Kiểm tra kiểu dữ liệu để đảm bảo đúng format
        if isinstance(raw_words, str):
            raw_words = raw_words.split()

        # Xử lý dựa trên task type
        if task == "seq_labeling":
            # Cho seq_labeling: tách token + lặp lại nhãn gốc
            new_words, original_indices = preprocess_token_list(raw_words, task)
            
            if label_key in sample:
                raw_tags = sample[label_key]
                new_tags = []
                
                # Mapping tag dựa trên index gốc
                valid_sample = True
                for original_idx in original_indices:
                    if original_idx < len(raw_tags):
                        new_tags.append(raw_tags[original_idx])
                    else:
                        print(f"Warning: Index {original_idx} out of range for tags at sample {index}")
                        valid_sample = False
                        break
                
                if valid_sample:
                    sample[text_key] = new_words
                    sample[label_key] = new_tags
                    new_data.append(sample)
                else:
                    skipped_count += 1
            else:
                # Nếu không có label thì chỉ lưu words (dùng cho inference)
                sample[text_key] = new_words
                new_data.append(sample)
        
        else:
            # Cho text_classification và aspect_based: xử lý đầy đủ
            new_words, original_indices = preprocess_token_list(raw_words, task)
            
            if task == "text_classification" or task == "aspect_based":
                # Với classification, nối lại thành câu string
                sample[text_key] = " ".join(new_words)
                new_data.append(sample)

    # Lưu file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # Ghi theo format JSON Lines (mỗi dòng 1 object) cho đúng chuẩn đầu vào Dataset
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Done: {input_path} -> {output_path}")
    print(f"Số mẫu: {len(new_data)}, Bỏ qua: {skipped_count}")

if __name__ == "__main__":
    # Cấu hình đường dẫn
    base_dir = "data"
    dataset_folder = r"E:\DS201.Q11\DS201\Lab_5\data\UIT-ViOCD"
    
    text_key = "review"
    label_key = "domain"
    task = "text_classification"

    input_paths = {
        "train": os.path.join(dataset_folder, "train.json"),
        "dev":   os.path.join(dataset_folder, "dev.json"),
        "test":  os.path.join(dataset_folder, "test.json")
    }
    
    # Sửa lại đường dẫn output cho gọn hoặc giữ nguyên tùy bạn
    output_paths = {
        "train": os.path.join(dataset_folder, "train_preprocessed.json"),
        "dev":   os.path.join(dataset_folder, "dev_preprocessed.json"),
        "test":  os.path.join(dataset_folder, "test_preprocessed.json")
    }

    for split, path in input_paths.items():
        if os.path.exists(path):
            convert_numbers_in_file(path, output_paths[split], task, text_key, label_key)
        else:
            print(f"Không tìm thấy file: {path}")