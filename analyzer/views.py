from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import subprocess
import os
import tempfile
import sys
import logging

logger = logging.getLogger(__name__)

# Create your views here.

class AnalyzeFileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def create_error_response(self, message, status_code=500):
        return Response({
            'success': False,
            'containsSwitch': False,
            'predictedLabel': 'error',
            'message': message
        }, status=status_code)

    def create_success_response(self, predicted_label):
        contains_switch = 'switch' in predicted_label.lower()
        return Response({
            'success': True,
            'containsSwitch': contains_switch,
            'predictedLabel': predicted_label,
            'message': f'분석 완료: {predicted_label}'
        })

    def post(self, request):
        temp_path = None
        try:
            # 파일 검증
            if 'file' not in request.FILES:
                return self.create_error_response('파일이 제공되지 않았습니다.', 400)

            file = request.FILES['file']
            
            # 파일 크기 제한 체크 (예: 10MB)
            if file.size > 10 * 1024 * 1024:
                return self.create_error_response('파일 크기가 너무 큽니다. 10MB 이하의 파일만 허용됩니다.', 400)

            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.s', mode='wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            # analyze_script.py의 절대 경로 계산
            script_path = os.path.join(settings.BASE_DIR, 'analyze_script.py')
            
            if not os.path.exists(script_path):
                return self.create_error_response('분석 스크립트를 찾을 수 없습니다.')

            # Python 스크립트 실행
            try:
                result = subprocess.run(
                    [sys.executable, script_path, temp_path],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=30  # 30초 타임아웃 설정
                )
            except subprocess.TimeoutExpired:
                return self.create_error_response('분석 시간이 초과되었습니다.')
            
            if result.returncode != 0:
                logger.error(f"분석 스크립트 오류: {result.stderr}")
                return self.create_error_response(f'분석 중 오류 발생: {result.stderr}')

            # 출력에서 예측 레이블 파싱
            output = result.stdout.strip()
            if not output or "Predicted: " not in output:
                return self.create_error_response('분석 결과를 파싱할 수 없습니다.')

            predicted_label = output.split("Predicted: ")[-1].strip()
            return self.create_success_response(predicted_label)

        except Exception as e:
            logger.exception("파일 분석 중 예외 발생")
            return self.create_error_response(f'예기치 않은 오류가 발생했습니다: {str(e)}')

        finally:
            # 임시 파일 정리
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {str(e)}")
