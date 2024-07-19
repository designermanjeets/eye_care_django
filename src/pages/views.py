import os
from django import get_version
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_POST

import openai
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import time

import json
import requests

from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI client
open_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=open_api_key)

# Initialize the API credentials
vendor_id = os.getenv("VendorId")
vendor_password = os.getenv("VendorPassword")
account_id = os.getenv("AccountId")
account_password = os.getenv("AccountPassword")

# Print the model structure for debugging
def print_model_structure(model):
    for name, module in model.named_modules():
        print(name)
print("Base model structure:")

# Define the function to call the Hugging Face endpoint
def call_huggingface_endpoint(prompt, api_url, api_token, retries=3, backoff_factor=0.3):
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,
            "num_return_sequences": 1,
        }
    }
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise e

# Function to identify intent
def identify_intent(user_query):
    prompt = (
       f"""Identify the intent of the following query: "{user_query}".
       Is it related to rescheduling an appointment, canceling an appointment, booking an appointment or something else?"""
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
    )

    # Extract the intent from the response
    intent = chat_completion.choices[0].message.content.strip().lower()
    print(intent)
    return intent

# Function to extract required information
def fetch_info(response):
    model = ChatOpenAI(temperature=0, openai_api_key=open_api_key)

    class Req_info(BaseModel):
        Lastname: str = Field(
            description="The value that identifies lastname of the user")
        DateOfBirth: str = Field(
            description="The value that represents date of birth")

    parser = PydanticOutputParser(pydantic_object=Req_info)
    prompt = PromptTemplate(
        template="Extract the information that defined.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    final_results = chain.invoke({"query": response})
    final_results1 = dict(final_results)
    print(type(final_results1))
    return final_results1

# Function to get authentication token
def get_auth_token(vendor_id, vendor_password, account_id, account_password):
    auth_url = "https://iochatbot.maximeyes.com/api/v2/account/authenticate"
    auth_payload = {
        "VendorId": vendor_id,
        "VendorPassword": vendor_password,
        "AccountId": account_id,
        "AccountPassword": account_password
    }
    headers = {'Content-Type': 'application/json'}
    try:
        auth_response = requests.post(auth_url, json=auth_payload, headers=headers)
        auth_response.raise_for_status()
        response_json = auth_response.json()

        if response_json.get('IsToken'):
            return response_json.get('Token')
        else:
            print("Error message:", response_json.get('ErrorMessage'))
            return None
    except requests.RequestException as e:
        print(f"Authentication failed: {str(e)}")
        return None
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return None

# Function to reschedule appointment
def reschedule_appointment(auth_token, Lastname, DateOfBirth, IsActive="true"):
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {auth_token}'  # Corrected this line
    }

    # Step 1: Get the patient number
    get_patient_url = "https://iochatbot.maximeyes.com/api/patient/getlistofpatient"
    patient_payload = {
        "Lastname": Lastname,
        "DateOfBirth": DateOfBirth,
        "IsActive": IsActive
    }

    try:
        patient_response = requests.post(get_patient_url, json=patient_payload, headers=headers)
        patient_response.raise_for_status()
        print(f"URL: {get_patient_url}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return

    if patient_response.status_code != 200:
        return f"Failed to get patient details. Status code: {patient_response.status_code}"
    else:
        pass

    patient_number = input("enter the patient_number: ")
    # Step 1: Get the appointment details
    get_appointment_url = f"https://iochatbot.maximeyes.com/api/appointment/getappointmentchatbot?PatientNumber={patient_number}"

    try:
        appointment_response = requests.get(get_appointment_url, headers=headers)
        appointment_response.raise_for_status()

        print(f"URL: {get_appointment_url}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    if appointment_response.status_code != 200:
        return f"Failed to get appointment details. Status code: {appointment_response.status_code}"

    try:
        appointment_details = appointment_response.json()
    except ValueError:
        return "Failed to parse appointment details response as JSON."

    if not appointment_details:
        return "No appointments found for the given patient number."

    # Extract required IDs from the appointment details
    appointment_id = appointment_details[0].get('AppointmentId')
    patient_id = appointment_details[0].get('PatientId')
    from_date = appointment_details[0].get('FromDate')

    # Step 2: Cancel the current appointment
    cancel_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/cancelappoinmentschatbot"
    cancel_payload = {
        "PatientId": patient_id,
        "AppointmentId": appointment_id,
        "from_date": from_date
    }
    cancel_response = requests.post(cancel_appointment_url, json=cancel_payload, headers=headers)

    if cancel_response.status_code != 200:
        return f"Failed to cancel the appointment. Status code: {cancel_response.status_code}"

    # Step 3: Update the appointment status
    update_status_url = f"https://iochatbot.maximeyes.com/api/appointment/updateappointmentstatus/{appointment_id}"
    update_status_payload = {
        "Status": "Cancelled"
    }
    update_status_response = requests.put(update_status_url, json=update_status_payload, headers=headers)

    if update_status_response.status_code != 200:
        return f"Failed to update appointment status. Status code: {update_status_response.status_code}"

    # Step 4: Get new slots for rescheduling
    open_slots_url = f"https://iochatbot.maximeyes.com/api/appointment/openslotforchatbot?fromDate={from_date}&isOpenSlotsOnly=true"
    open_slots_response = requests.get(open_slots_url, headers=headers)

    if open_slots_response.status_code != 200:
        return f"Failed to get open slots. Status code: {open_slots_response.status_code}"

    try:
        open_slots = open_slots_response.json()
    except ValueError:
        return "Failed to parse open slots response as JSON."

    if not open_slots:
        return "No open slots available for rescheduling."

    return open_slots

# Function to generate response using Hugging Face endpoint
def generate_response(prompt, max_length=512, num_return_sequences=1):
    api_url = "https://tpfuzx0pqdencyjo.us-east-1.aws.endpoints.huggingface.cloud"
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    return call_huggingface_endpoint(prompt, api_url, api_token)

# Function to handle missing information
def ask_for_missing_info(request, required_info):
    if 'missing_info' not in request.session:
        request.session['missing_info'] = required_info
    else:
        request.session['missing_info'].update(required_info)
    
    missing_info = request.session['missing_info']
    missing_keys = [key for key, value in missing_info.items() if value is None]

    if missing_keys:
        missing_key = missing_keys[0]
        return f"Please provide your {missing_key.replace('_', ' ')}."
    return None

# Main function to handle user query
def handle_user_query_me(request, user_query):
    # Identify the user's intent
    print("user_query", user_query)
    intent = identify_intent(user_query)
    print(f"Identified intent: {intent}")

    # Initialize required information if not already in session
    if 'required_info' not in request.session:
        request.session['required_info'] = {
            'first_name': None,
            'last_name': None,
            'dob': None,
            'mobile_number': None,
            'email': None
        }

    # If the intent is to reschedule an appointment
    if intent == "rescheduling":
        missing_info = ask_for_missing_info(request, request.session['required_info'])
        if missing_info:
            return missing_info

        # Extract required information
        extracted_info = fetch_info(user_query)
        print(type(extracted_info))
        print(f"Extracted info: {extracted_info}")

        # Update session with extracted information
        request.session['required_info'].update(extracted_info)

        # Check for any missing information
        missing_info = ask_for_missing_info(request, request.session['required_info'])
        if missing_info:
            return missing_info

        # Get authentication token
        auth_token = get_auth_token(vendor_id, vendor_password, account_id, account_password)
        if not auth_token:
            return "Failed to authenticate."

        # Reschedule the appointment and get open slots
        required_info = request.session['required_info']
        open_slots = reschedule_appointment(auth_token, required_info['last_name'], required_info['dob'])
        return open_slots

    # If the intent is not related to rescheduling, generate a response
    else:
        response = generate_response(user_query)
        return response


def home(request):
    context = {
        "debug": settings.DEBUG,
        "django_ver": get_version(),
        # "python_ver": os.environ["PYTHON_VERSION"],
    }

    return render(request, "pages/home.html", context)

@csrf_exempt
@require_POST
def handle_user_query(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        print(f"Received data: {data}")
        
        if not isinstance(data, dict):
            return JsonResponse({"error": "Invalid JSON format, expected a JSON object"}, status=400)

        input_message = data["message"]
        if not input_message:
            return JsonResponse({"error": "Missing 'message' in request data"}, status=400)

        response = handle_user_query_me(request, input_message)
        return JsonResponse({"response": response})
    
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except KeyError:
        return JsonResponse({"error": "Missing 'message' in request data"}, status=400)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse({"error": "An internal error occurred"}, status=500)
