from django.urls import path
from .views import AnalyzeFileView

urlpatterns = [
    path('analyze/', AnalyzeFileView.as_view(), name='analyze'),
] 