import json
import re
import os
from unicodedata import normalize
try:
    from num2words import num2words
except ImportError:
    print("Vui lòng cài đặt thư viện num2words: pip install num2words")
    num2words = None

def detect_language_simple(text: str) -> str:
    """Nhận diện ngôn ngữ đơn giản: 'vi' hoặc 'en'."""
    vietnamese_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    if any(char in vietnamese_chars for char in text.lower()):
        return 'vi'
    return 'en'

def replace_numbers(text: str, lang: str) -> str:
    """Chuyển đổi số thành chữ (giữ nguyên logic ưu tiên float/int)."""
    if num2words is None: return text 

    def replace(match):
        number_str = match.group()
        try:
            # Ưu tiên float nếu có dấu chấm, ngược lại là int
            number = float(number_str) if '.' in number_str else int(number_str)
            # num2words trả về chuỗi (vd: "mười chín"), ta trả về luôn để regex thay thế
            return num2words(number, lang=lang)
        except:
            return number_str
            
    # Regex bắt số nguyên hoặc số thập phân
    return re.sub(r'-?\d+(\.\d+)?', replace, text)

def preprocess_sentence(sentence: str, lang: str = None) -> str:
    """
    Tiền xử lý câu:
    1. Chuẩn hóa & Lowercase
    2. Tách các ký tự đặc biệt
    3. Chuyển số thành chữ
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return ""
    
    # 1. Chuẩn hóa & Lowercase
    sentence = normalize("NFKC", sentence).lower().strip()
    
    # 2. Tách các ký tự đặc biệt
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    # Lưu ý: Dấu chấm có thể là kết thúc câu hoặc số thập phân. 
    sentence = re.sub(r"(?<!\d)\.(?!\d)", " . ", sentence) 
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)

    # 3. Nhận diện ngôn ngữ (nếu chưa có)
    if lang is None:
        lang = detect_language_simple(sentence)

    # 4. Chuyển số thành chữ (xử lý trên từng từ sau khi đã tách đặc biệt)
    # Split ra để xử lý số, tránh bị dính các ký tự lạ còn sót
    tokens = sentence.split()
    final_tokens = []
    for token in tokens:
        # Gọi hàm đổi số
        wordified = replace_numbers(token, lang)
        final_tokens.append(wordified)
    
    # Nối lại thành chuỗi
    return " ".join(final_tokens)

def process_translation_file(input_path, output_path, src_key="english", tgt_key="vietnamese"):
    if not os.path.exists(input_path):
        print(f"[BỎ QUA] Không tìm thấy: {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[LỖI] Đọc file thất bại: {e}")
        return

    processed_data = []
    
    for item in data:
        src_text = item.get(src_key, "")
        tgt_text = item.get(tgt_key, "")
        
        # Xử lý với ngôn ngữ cụ thể
        clean_src = preprocess_sentence(src_text, lang='en')
        clean_tgt = preprocess_sentence(tgt_text, lang='vi')
        
        if clean_src and clean_tgt:
            processed_data.append({
                src_key: clean_src,
                tgt_key: clean_tgt
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"[XONG] {input_path} -> {output_path} ({len(processed_data)} mẫu)")

if __name__ == "__main__":
    #  CẤU HÌNH 
    dataset_folder = r".\kaggle\input\small-phomt" 
    
    files_to_process = {
        "train": ("train.json", "train_preprocessed.json"),
        "dev":   ("dev.json",   "dev_preprocessed.json"),
        "test":  ("test.json",  "test_preprocessed.json")
    }

    print(" BẮT ĐẦU TIỀN XỬ LÝ (REGEX ĐẶC BIỆT + SỐ -> CHỮ) ")
    for split, (in_f, out_f) in files_to_process.items():
        process_translation_file(
            os.path.join(dataset_folder, in_f),
            os.path.join(dataset_folder, out_f)
        )