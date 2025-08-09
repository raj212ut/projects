#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from flask import Flask, request, render_template
from django.urls import path
import pandas as pd

app = Flask(__name__)  # Add Flask app instance

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diab.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


@app.route('/disease', methods=['POST'])
def disease():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        
        # Add symptoms_df definition before using it
        symptoms_df = pd.DataFrame([user_symptoms], columns=SYMPTOMS_COLUMNS)  # Replace SYMPTOMS_COLUMNS with your actual columns
        
        predicted_disease = get_predicted_value(symptoms_df)  # Pass the DataFrame

        desc, pre, med, die, wrkout = helper(predicted_disease)

        return render_template('disease.html', 
                               predicted_disease=predicted_disease, 
                               dis_des=desc, 
                               dis_pre=pre, 
                               dis_med=med, 
                               dis_diet=die, 
                               dis_wrkout=wrkout)
    return render_template('index.html')  # Redirect to index if not POST


if __name__ == '__main__':
    main()

# Update the urlpatterns to include the new route
urlpatterns = [
    path('', index, name='index'),
    path('predict/', predict, name='predict'),
    path('disease/', disease, name='disease'),  # New route for disease prediction
]
