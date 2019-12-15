import os
import sys
import django


pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyGraduationProject.settings')
django.setup()
