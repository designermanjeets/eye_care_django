from django.urls import path

from pages import views

urlpatterns = [
    # path("", views.home, name="home"),
    path("", views.home, name="home"),
    # path("practice2", views.home2, name="home"),
    # path("practice1", views.home, name="home"),
    # path("handle_jquery_response", views.func, name="jquerry"),
    path("chatbot_view", views.chatbot_view, name="handle_user_query"),
]
