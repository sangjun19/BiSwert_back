from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import subprocess
import os
import tempfile
import sys

# Create your views here.

class AnalyzeFileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        try:
            if 'file' not in request.FILES:
                return Response({
                    'containsSwitch': False,
                    'predictedLabel': 'error',
                    'message': '파일이 제공되지 않았습니다.'
                }, status=400)

            file = request.FILES['file']
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.s', mode='wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            try:
                # analyze_script.py의 절대 경로 계산
                script_path = os.path.join(settings.BASE_DIR, 'analyze_script.py')
                
                # Python 스크립트 실행
                result = subprocess.run([sys.executable, script_path, temp_path], 
                                     capture_output=True, 
                                     text=True,
                                     encoding='utf-8')
                
                if result.returncode == 0:
                    # 출력에서 예측 레이블 파싱
                    output = result.stdout.strip()
                    predicted_label = output.split("Predicted: ")[-1].strip()
                    contains_switch = 'switch' in predicted_label.lower()
                    
                    return Response({
                        'containsSwitch': contains_switch,
                        'predictedLabel': predicted_label,
                        'message': f'분석 완료: {predicted_label}'
                    })
                else:
                    return Response({
                        'containsSwitch': False,
                        'predictedLabel': 'error',
                        'message': f'분석 중 오류 발생: {result.stderr}'
                    }, status=500)
                    
            finally:
                # 임시 파일 삭제
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
        except Exception as e:
            return Response({
                'containsSwitch': False,
                'predictedLabel': 'error',
                'message': f'분석 중 오류가 발생했습니다: {str(e)}'
            }, status=500)
