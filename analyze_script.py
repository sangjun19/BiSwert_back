import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os

LABEL_NAMES = [
    "switch_origin",
    "switch_flat",
    "switch_opaque",
    "switch_vir",
    "non_switch_origin",
    "non_switch_flat",
    "non_switch_opaque",
    "non_switch_vir"
]

def analyze_file(file_path):
    # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 모델 및 토크나이저 경로 설정
    model_path = os.path.join(base_dir, "results-checkpoint-1400")
    tokenizer_path = os.path.join(base_dir, "results-checkpoint-1400")  # 토크나이저도 같은 위치에 있다고 가정
    
    print(f"Loading model from: {model_path}")  # 디버깅용 출력
    
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model.eval()

    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # 토크나이징 및 예측
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    print(f"Predicted: {LABEL_NAMES[predicted_class]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_script.py <file_path>")
        sys.exit(1)
    
    analyze_file(sys.argv[1]) 