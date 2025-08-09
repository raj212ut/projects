from django.urls import path
from home import views


urlpatterns = [
    path('', views.index, name="home"),
    path('about', views.about, name="about"),
    path('contact', views.contact, name="contacts"),
    path('registration', views.registration, name="registration"),
    path('login/', views.user_login, name="login"),
    path('prediction', views.prediction, name="prediction"),
    path('diabete', views.diabete, name="diabete"),
    path('heart', views.heart, name="heart"),
    path('disease', views.disease, name="disease")



]