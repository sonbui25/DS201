import json
import re
from num2words import num2words
import os
from unicodedata import normalize
from typing import List, Tuple

def replace_numbers_with_words(text):
    """Chuyển đổi chuỗi số thành chữ tiếng Việt."""
    def replace(match):
        number_str = match.group()
        try:
            # Ưu tiên float nếu có dấu chấm, ngược lại là int
            number = float(number_str) if '.' in number_str else int(number_str)
            return num2words(number, lang='vi')
        except:
            return number_str
    # Regex bắt số nguyên hoặc số thập phân
    return re.sub(r'-?\d+(\.\d+)?', replace, text)

def process_single_token(token: str) -> List[str]:
    """
    Xử lý một token đơn lẻ: chuẩn hóa, tách ký tự đặc biệt, đổi số.
    Trả về danh sách các sub-tokens.
    """
    token = token.lower()
    token = normalize("NFKC", token)
    
    # 1. Tách các ký tự đặc biệt bằng cách thêm khoảng trắng
    if len(token) > 1:
        token = re.sub(r"!", " ! ", token)
        token = re.sub(r"\?", " ? ", token)
        token = re.sub(r":", " : ", token)
        token = re.sub(r";", " ; ", token)
        token = re.sub(r",", " , ", token)
        token = re.sub(r"\"", " \" ", token)
        token = re.sub(r"'", " ' ", token)
        token = re.sub(r"\(", " ( ", token)
        token = re.sub(r"\[", " [ ", token)
        token = re.sub(r"\)", " ) ", token)
        token = re.sub(r"\]", " ] ", token)
        token = re.sub(r"/", " / ", token)
        token = re.sub(r"\.", " . ", token)
        token = re.sub(r"-", " - ", token)
        token = re.sub(r"\$", " $ ", token)
        token = re.sub(r"\&", " & ", token)
        token = re.sub(r"\*", " * ", token)

    # 2. Split ra thành các sub-token (ví dụ "COVID-19" -> ["covid", "-", "19"])
    sub_tokens = token.strip().split()

    # 3. Đổi số thành chữ cho từng sub-token
    final_tokens = []
    for sub in sub_tokens:
        wordified = replace_numbers_with_words(sub)
        # num2words có thể trả về chuỗi có khoảng trắng (ví dụ "mười chín")
        # nên cần split tiếp để đảm bảo cấu trúc phẳng
        final_tokens.extend(wordified.split())
        
    return final_tokens

def preprocess_token_list(word_list: List[str]) -> Tuple[List[str], List[int]]:
    """
    Xử lý danh sách các từ (input list) và trả về danh sách mới + mapping index gốc.
    """
    new_words = []
    original_indices = []

    for i, token in enumerate(word_list):
        # Xử lý token hiện tại
        processed_tokens = process_single_token(token)
        
        # Thêm vào danh sách mới
        new_words.extend(processed_tokens)
        
        # Mapping: Các token mới sinh ra đều trỏ về index i của token gốc
        # Ví dụ: token cũ là "19" (index 5) -> "mười", "chín" -> cả 2 đều có index 5
        original_indices.extend([i] * len(processed_tokens))

    return new_words, original_indices

def convert_numbers_in_file(input_path, output_path, task, text_key, label_key):
    # Đọc file
    try:
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
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

        raw_words = sample[text_key] # Đây là List: ["Từ", "Từ", ...]
        
        # Kiểm tra kiểu dữ liệu để đảm bảo đúng format
        if isinstance(raw_words, str):
            # Fallback nếu dữ liệu lỡ là string (tách tạm bằng split)
            raw_words = raw_words.split() 

        # --- XỬ LÝ CHÍNH ---
        # Hàm mới trả về list words đã xử lý và mapping index
        new_words, original_indices = preprocess_token_list(raw_words)
        
        # Cập nhật lại words mới vào sample
        sample[text_key] = new_words 

        # Xử lý Labels (Tags) cho bài toán Seq Labeling
        if task == "seq_labeling":
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
                    sample[label_key] = new_tags
                    new_data.append(sample)
                else:
                    skipped_count += 1
            else:
                # Nếu không có label thì chỉ lưu words (dùng cho inference)
                new_data.append(sample)
        
        elif task == "text_classification" or task == "aspect_based":
            # Với classification,nối lại thành câu string
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
    dataset_folder = r"E:\DS201.Q11\DS201\Lab_3\data\PhoNER_COVID19\data\syllable"
    
    text_key = "words"
    label_key = "tags"
    task = "seq_labeling"

    input_paths = {
        "train": os.path.join(dataset_folder, "train_syllable.json"),
        "dev":   os.path.join(dataset_folder, "dev_syllable.json"),
        "test":  os.path.join(dataset_folder, "test_syllable.json")
    }
    
    # Sửa lại đường dẫn output cho gọn hoặc giữ nguyên tùy bạn
    output_paths = {
        "train": os.path.join(dataset_folder, "train_syllable_preprocessed.json"),
        "dev":   os.path.join(dataset_folder, "dev_syllable_preprocessed.json"),
        "test":  os.path.join(dataset_folder, "test_syllable_preprocessed.json")
    }

    for split, path in input_paths.items():
        if os.path.exists(path):
            convert_numbers_in_file(path, output_paths[split], task, text_key, label_key)
        else:
            print(f"Không tìm thấy file: {path}")