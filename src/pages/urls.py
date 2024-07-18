from django.urls import path

from pages import views

urlpatterns = [
    path("", views.home, name="home"),
    path("handle_user_query", views.handle_user_query, name="handle_user_query"),
]
