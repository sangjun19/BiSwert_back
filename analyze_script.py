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
MODEL_DIR = "results/8class"  # 모델 파일이 저장될 디렉토리
MODEL_NAME = "checkpoint-1400"  # 모델 체크포인트 이름

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

def ensure_model_path():
    """모델 디렉토리 경로를 확인하고 생성하는 함수"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, MODEL_DIR)
    model_path = os.path.join(model_dir, MODEL_NAME)
    
    if not os.path.exists(model_path):
        logger.error(f"""
모델 파일을 찾을 수 없습니다. 다음 경로에 모델 파일이 필요합니다:
{model_path}

다음 단계를 따라 모델 파일을 설정하세요:
1. '{MODEL_DIR}' 디렉토리 구조를 생성하세요: mkdir -p {MODEL_DIR}
2. 학습된 모델 파일을 '{MODEL_DIR}/{MODEL_NAME}' 위치에 복사하세요.
""")
        raise FileNotFoundError(f"모델 파일이 없습니다. 경로: {model_path}")
    
    return model_path

def load_model_and_tokenizer(model_path: str, tokenizer_path: str) -> tuple[BertForSequenceClassification, BertTokenizer]:
    """모델과 토크나이저를 로드하는 함수"""
    try:
        logger.info(f"모델을 로드합니다: {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # GPU 사용 가능 시 활용
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"사용 중인 디바이스: {device}")
        model = model.to(device)
        
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
        raise

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
        # 모델 경로 확인 및 설정
        model_path = ensure_model_path()
        
        # 모델과 토크나이저 로드
        model, tokenizer, device = load_model_and_tokenizer(model_path, model_path)
        
        # 파일 읽기
        test_text = read_file_content(file_path)
        if not test_text:
            raise ValueError("파일이 비어있습니다")

        # 토크나이징 및 예측
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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