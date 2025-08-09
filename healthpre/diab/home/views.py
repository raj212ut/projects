from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
import numpy as np
import pandas as pd
import pickle

from joblib import load
mod = load('./static/Mod.joblib')


# Create your views here.
def index(request):
    # return HttpResponse("this is the home")
    return render(request, "index.html")

def about(request):
    return render(request, "about.html")

def user_login(request):
    error_message = None  # Initialize a variable to hold the error message
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect("diabete")
        else:
            error_message = "Invalid username or password. Please try again."  # Set the error message

    return render(request, "login.html", {'error_message': error_message})  # Pass the message to the template

def registration(request):
    success_message = None  # Initialize a variable to hold the success message
    if request.method == "POST":
        username = request.POST.get("username")
        Email = request.POST.get("email")
        Password = request.POST.get("password")
        Confirm_Password = request.POST.get("confirm_password")
        
        if Password != Confirm_Password:
            return render(request, 'registration.html', {'error_message': "Passwords do not match"})  # Show error on the same page
        
        my_user = User.objects.create_user(username=username, email=Email, password=Password)
        my_user.save()
        success_message = "User created successfully!"  # Set the success message

    return render(request, 'registration.html', {'success_message': success_message})  # Pass the message to the template

def contact(request):
    return render(request, "contact.html")

def prediction(request):
    return render(request, "prediction.html")
    
def disease(request):
    if request.method == "POST":
        try:
            # Load datasets
            sym_des = pd.read_csv("static/data/symtoms_df.csv")
            precautions = pd.read_csv("static/data/precautions_df.csv")
            workout = pd.read_csv("static/data/workout_df.csv")
            description = pd.read_csv("static/data/description.csv")
            medications = pd.read_csv('static/data/medications.csv')
            diets = pd.read_csv("static/data/diets.csv")

            # Load model
            svc = pickle.load(open('static/svc.pkl', 'rb'))

            # Get symptoms from form
            symptoms = request.POST.get('symptoms', '')
            symptom_list = [s.strip() for s in symptoms.split(',')]

            symptoms_dict = {...}  # Replace with the original dictionary
            diseases_list = {...}
            # Create input vector
            input_vector = np.zeros(len(symptoms_dict))
            for symptom in symptom_list:
                if symptom in symptoms_dict:
                    input_vector[symptoms_dict[symptom]] = 1

            # Make prediction
            predicted_disease = diseases_list[svc.predict([input_vector])[0]]

            # Get additional information
            def helper(dis):
                desc = description[description['Disease'] == dis]['Description'].values
                desc = " ".join([str(w) for w in desc])

                pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
                pre = [col for col in pre.values]

                med = medications[medications['Disease'] == dis]['Medication'].values
                med = [str(m) for m in med]

                die = diets[diets['Disease'] == dis]['Diet'].values
                die = [str(d) for d in die]

                wrkout = workout[workout['disease'] == dis]['workout'].values
                wrkout = [str(w) for w in wrkout]

                return desc, pre, med, die, wrkout

            desc, pre, med, die, wrkout = helper(predicted_disease)
            precautions = [item for sublist in pre for item in sublist]

            # Prepare context for template
            context = {
                "Disease": predicted_disease,
                "Description": desc,
                "Precautions": precautions,
                "Medications": med,
                "Diets": die,
                "Workouts": wrkout
            }

            return render(request, 'disease_result.html', context)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render(request, 'disease.html', {'error_message': error_message})

    # If GET request, show the form
    return render(request, 'disease.html')






def heart(request):
    output = None
    error_message = None
    
    if request.method == "POST":
        try:
            # Get form values with proper type conversion and validation
            def get_float_value(field, min_val, max_val):
                value = request.POST.get(field)
                if not value:
                    raise ValueError(f"{field} is required")
                try:
                    float_val = float(value)
                    if not (min_val <= float_val <= max_val):
                        raise ValueError(f"{field} must be between {min_val} and {max_val}")
                    return float_val
                except ValueError:
                    raise ValueError(f"Invalid {field} value")

            def get_int_value(field, valid_values=None):
                value = request.POST.get(field)
                if not value:
                    raise ValueError(f"{field} is required")
                try:
                    int_val = int(value)
                    if valid_values and int_val not in valid_values:
                        raise ValueError(f"Invalid {field} value")
                    return int_val
                except ValueError:
                    raise ValueError(f"Invalid {field} value")

            # Get values in the same order as training
            features = [
                get_int_value('age', range(1, 121)),
                get_int_value('sex', [0, 1]),
                get_int_value('cp', [0, 1, 2, 3]),
                get_float_value('trestbps', 50, 250),
                get_float_value('chol', 100, 600),
                get_int_value('fbs', [0, 1]),
                get_int_value('restecg', [0, 1, 2]),
                get_float_value('thalach', 60, 220),
                get_int_value('exang', [0, 1]),
                get_float_value('oldpeak', 0, 6.2),
                get_int_value('slope', [0, 1, 2]),
                get_int_value('ca', range(0, 4)),
                get_int_value('thal', [0, 1, 2, 3])
            ]

            # Load and verify heart disease model
            heart_mod = load('./static/M.joblib')
            if not hasattr(heart_mod, 'classes_'):
                raise ValueError("Heart disease model is not trained. Please train the model first.")
            
            # Make prediction with correct feature count
            prediction = heart_mod.predict([features])
            output = 'No Heart Disease' if prediction[0] == 0 else 'Heart Disease Detected'
                
        except ValueError as e:
            error_message = str(e)
            return render(request, "heart.html", {'error': error_message})
            
        except Exception as e:
            print("Unexpected error:", str(e))
            error_message = f"An error occurred: {str(e)}"
            return render(request, "heart.html", {'error': error_message})

    return render(request, "heart.html", {
        'output': output,
        'error_message': error_message
    })


def diabete(request):
    output = None
    
    if request.method == "POST":
        try:
            # # Get form values
            pregnancies = request.POST.get('pregnancies')
            glucose = request.POST.get('glucose')
            blood_pressure = request.POST.get('bloodPressure')
            skin_thickness = request.POST.get('skinThickness')
            insulin = request.POST.get('insulin')
            bmi = request.POST.get('bmi')
            diabetes_pedigree = request.POST.get('diabetesPedigreeFunction')
            age = request.POST.get('age')

            # Convert form values to float
            # pregnancies = float(pregnancies)
            # glucose = float(glucose)
            # blood_pressure = float(blood_pressure)
            # skin_thickness = float(skin_thickness)
            # insulin = float(insulin)
            # bmi = float(bmi)
            # diabetes_pedigree = float(diabetes_pedigree)
            # age = float(age)

            y_pred = mod.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            if y_pred[0]==0:
                y_pred = 'non-diabetic'
            else:
                y_pred = 'diabetic'
            return render(request, "diabete.html", {'output': y_pred})

        except (ValueError, TypeError) as e:
            error_message = "Please enter valid numeric values for all fields"
            return render(request, "diabete.html", {'error': error_message})
            
        except Exception as e:
            print("Unexpected error:", str(e))
            error_message = f"An error occurred: {str(e)}"
            return render(request, "diabete.html", {'error': error_message})

    return render(request, "diabete.html")
