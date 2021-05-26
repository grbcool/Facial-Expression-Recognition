from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path(route='',view=views.base,name='base'),
    path(route='facial_expressions',view=views.facial_expression_detection,name='facial_expressions'),
    path('live',views.live,name='live'),
    
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
