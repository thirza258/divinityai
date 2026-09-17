from django.urls import path

from .views import (
    ConversationDetailView, ConversationListView, LoginView, LogoutView,
    MemoryDetailView, MemoryListView, ProfileView, RegisterView, SessionView,
)

urlpatterns = [
    path('auth/session', SessionView.as_view()),
    path('auth/register', RegisterView.as_view()),
    path('auth/login', LoginView.as_view()),
    path('auth/logout', LogoutView.as_view()),
    path('auth/profile', ProfileView.as_view()),
    path('conversations', ConversationListView.as_view()),
    path('conversations/<uuid:pk>', ConversationDetailView.as_view()),
    path('memories', MemoryListView.as_view()),
    path('memories/<uuid:pk>', MemoryDetailView.as_view()),
]
