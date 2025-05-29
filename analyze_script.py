import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os
import logging
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 모델 관련 상수 정의
BASE_DIR = "/home/yj-noh-3060/Desktop/hy-workspace/hyenv/switch_classification"
MODEL_PATH = os.path.join(BASE_DIR, "final_model/8class")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer/8class")

def get_latest_checkpoint(model_dir: str) -> str:
    """8class 디렉토리에서 사용 가능한 체크포인트 폴더를 찾는 함수"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")
    
    # 디렉토리 내의 모든 폴더 가져오기
    checkpoints = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not checkpoints:
        raise FileNotFoundError(f"사용 가능한 체크포인트 폴더가 없습니다: {model_dir}")
    
    # 첫 번째 체크포인트 선택
    selected_checkpoint = checkpoints[0]
    logger.info(f"선택된 체크포인트 폴더: {selected_checkpoint}")
    return os.path.join(model_dir, selected_checkpoint)

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

def ensure_model_paths():
    """모델과 토크나이저 경로를 확인하는 함수"""
    model_path = get_latest_checkpoint(MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다. 경로: {model_path}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"토크나이저 파일이 없습니다. 경로: {TOKENIZER_PATH}")
    return model_path, TOKENIZER_PATH

def read_file_content(file_path: str) -> Optional[str]:
    """파일 내용을 읽는 함수"""
    encodings = ['utf-8', 'cp949', 'euc-kr', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"파일 읽기 오류: {str(e)}")
            raise
    
    raise UnicodeDecodeError(f"지원되는 인코딩으로 파일을 읽을 수 없습니다: {file_path}")

def analyze_file(file_path: str) -> str:
    try:
        # 모델과 토크나이저 경로 확인
        model_path, tokenizer_path = ensure_model_paths()
        
        # 토크나이저 로드 (bert-base-uncased 사용)
        logger.info("토크나이저를 로드합니다...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 모델 로드
        logger.info(f"모델을 로드합니다: {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # 파일 읽기
        test_text = read_file_content(file_path)
        if not test_text:
            raise ValueError("파일이 비어있습니다")

        # 토크나이징
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        result = LABEL_NAMES[predicted_class]
        logger.info(f"분석 결과: {result}")
        return result

    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            logger.error("Usage: python analyze_script.py <file_path>")
            sys.exit(1)
        
        if not os.path.exists(sys.argv[1]):
            logger.error(f"파일이 존재하지 않습니다: {sys.argv[1]}")
            sys.exit(1)
            
        result = analyze_file(sys.argv[1])
        print(f"Predicted: {result}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1) 