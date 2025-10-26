# rag/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('document/', views.upload_document, name='upload_document'),
    path('session/create/', views.create_session, name='create_session'),
    path('session/<int:session_id>/messages/', views.get_session, name='get_session'),
    path('chat/', views.chat_stream, name='chat_stream'),
    path('health/', views.health_check, name='health_check'),
]
