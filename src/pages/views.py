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
import random
import json
import requests
from datetime import datetime, timedelta
import re

from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI client
# open_api_key = os.getenv("OPENAI_API_KEY")
open_api_key = ''

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


def convert_date_format(date_str):
    # Define date formats
    formats = [
        "%Y-%m-%d",        # ISO format
        "%d-%m-%Y",        # European format
        "%m-%d-%Y",        # US format
        "%B %d, %Y",       # Full month name with day and year
        "%b %d, %Y",       # Abbreviated month name with day and year
        "%d %B %Y"         # Day with full month name and year
    ]

    # Remove ordinal suffixes (e.g., "st", "nd", "rd", "th")
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    # Try parsing with each format
    for fmt in formats:
        try:
            # Parse the date
            parsed_date = datetime.strptime(date_str, fmt)
            # Return in YYYY-MM-DD format
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # If no formats match, return the original string
    return date_str


def fetch_info_openai(response):
    prompt =(
        f"Extract the following information from the text if available else give empty value:\n"
        f"Text: {response}\n"
        f"Extracted Information:\n"
        f"FirstName:\n"
        f"LastName:\n"
        f"DateOfBirth:\n"
        f"PhoneNumber:\n"
        f"Email:\n"
        f"Preferred date or time:\n"

    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
          {
              "role": "user",
              "content": prompt,
          }
          ]
        )
        # result = completion.choices[0].text.strip()
        result = completion.choices[0].message.content.strip()

        info = {}
        for line in result.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()

        return info
    except Exception as e:
        print(f"Error extracting information: {e}")
        return {}


# Function to identify intent
def identify_intent(user_query):
    prompt = (
        f"""Identify the intent of the following query: "{user_query}".
        Is it related to book an appointment, rescheduling an appointment, canceling an appointment, or something else?  and also check the appointment date and time """
    )
    chat_completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ]
    )
    # Extract the intent from the response
    intent = chat_completion.choices[0].message.content.strip()
    return intent


def validation_check(user_query):
    prompt = (
        f"""Identify the user input: "{user_query}".
        its related to user repose its validate the data is correct or not correct  if correct and return True and if not correct then return False"""
    )
    chat_completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ]
    )
    # Extract the intent from the response
    intent = chat_completion.choices[0].message.content.strip()
    return intent


# # Function to extract required information
# def fetch_info(response):
#     model = ChatOpenAI(temperature=0, openai_api_key=open_api_key)

#     class Req_info(BaseModel):
#         Lastname: str = Field(
#             description="The value that identifies lastname of the user")
#         DateOfBirth: str = Field(
#             description="The value that represents date of birth")

#     parser = PydanticOutputParser(pydantic_object=Req_info)
#     prompt = PromptTemplate(
#         template="Extract the information that defined.\n{format_instructions}\n{query}\n",
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     chain = prompt | model | parser
#     final_results = chain.invoke({"query": response})
#     final_results1 = dict(final_results)
#     print(type(final_results1))
#     return final_results1

def transform_input(input_text):
    # Define a list of prompts to transform the input text
    prompts = [

        f"Rephrase the following request in a more engaging way related to appintment: '{input_text}'",

        f"How would you ask this question in a friendly and conversational tone related to appintment: '{input_text}'?",

        f"Make this request sound more personable and interesting related to appintment: '{input_text}'",

        f"Convert this question into a warm and inviting request related to appintment: '{input_text}'",

        f"Turn the following statement into a casual and friendly question related to appintment: '{input_text}'"

    ]

    # Select a random prompt from the list
    prompt = random.choice(prompts)

    # Call the API using the latest method
    response = client.chat.completions.create(

        model="gpt-3.5-turbo",

        messages=[

            {"role": "system", "content": "You are a creative apointment booking assistant .Start Chat with Greetings.\n not add too much words simple greeting and assist"},

            {"role": "user", "content": prompt}

        ],

        max_tokens=60,
        temperature=0.9

    )

    # Extract the response text
    transformed_text = response.choices[0].message.content.strip()
    return transformed_text

def update_info(extracted_info, additional_info):
    for key, value in additional_info.items():
        if value and value.lower() != "none":
            extracted_info[key] = value

# Function to get authentication token
# def get_auth_token(vendor_id, vendor_password, account_id, account_password):
#     auth_url = "https://iochatbot.maximeyes.com/api/v2/account/authenticate"
#     auth_payload = {
#         "VendorId": vendor_id,
#         "VendorPassword": vendor_password,
#         "AccountId": account_id,
#         "AccountPassword": account_password
#     }
#     headers = {'Content-Type': 'application/json'}
#     try:
#         auth_response = requests.post(auth_url, json=auth_payload, headers=headers)
#         auth_response.raise_for_status()
#         response_json = auth_response.json()

#         if response_json.get('IsToken'):
#             return response_json.get('Token')
#         else:
#             print("Error message:", response_json.get('ErrorMessage'))
#             return None
#     except requests.RequestException as e:
#         print(f"Authentication failed: {str(e)}")
#         return None
#     except json.JSONDecodeError:
#         print("Failed to decode JSON response")
#         return None


# Tool to get authentication token
def get_auth_token(vendor_id, vendor_password, account_id, account_password) -> str:
    """

    Get authentication token using vendor and account credentials.
    
    """
    auth_url = "https://iochatbot.maximeyes.com/api/v2/account/authenticate"
    auth_payload = { "VendorId": "e59ec838-2fc5-4639-b761-78e3ec55176c", "VendorPassword": "password@123", "AccountId": "chatbot1", "AccountPassword": "sJ0Y0oniZb6eoBMETuxUNy0aHf6tD6z3wynipZEAxcg=" }
    headers = {'Content-Type': 'application/json'}
    try:
        auth_response = requests.post(auth_url, json=auth_payload, headers=headers)
        auth_response.raise_for_status()
        response_json = auth_response.json()
 
        if response_json.get('IsToken'):
            return response_json.get('Token')
        else:
            return f"Error message: {response_json.get('ErrorMessage')}"
    except requests.RequestException as e:
        return f"Authentication failed: {str(e)}"
    except json.JSONDecodeError:
        return "Failed to decode JSON response"

def prefred_date_time_fun(response):
  #print("response12",response)
  if "next" in response.lower() or "comming" in response.lower() or "upcomming" in response.lower() or "tomorrow" in response.lower() or "next day" in response.lower():
        def get_next_weekday(day_name, use_next=False):
            # Dictionary to convert day names to weekday numbers
            days_of_week = {
                'monday': 0,
                'tuesday': 1,
                'wednesday': 2,
                'thursday': 3,
                'friday': 4,
                'saturday': 5,
                'sunday': 6
            }

            # Get today's date and the current weekday
            today = datetime.now()
            current_weekday = today.weekday()

            # Convert the day name to a weekday number
            target_weekday = days_of_week[day_name.lower()]

            # Calculate the number of days until the next target weekday
            days_until_target = (target_weekday - current_weekday + 7) % 7
            if days_until_target == 0 or use_next:
                days_until_target += 7

            # Calculate the date for the next target weekday
            next_weekday = today + timedelta(days=days_until_target)

            return next_weekday

        def get_relative_day(keyword):
            today = datetime.now()
            if keyword == "tomorrow":
                return today + timedelta(days=1)
            elif keyword == "day after tomorrow":
                return today + timedelta(days=2)
            return None

        keywords = ["next", "coming", "upcoming", "tomorrow", "day after tomorrow"]
        use_next = any(keyword in response for keyword in ["next", "coming", "upcoming"])

        # Check for "tomorrow" and "day after tomorrow"
        relative_day = None
        for keyword in ["tomorrow", "day after tomorrow"]:
            if keyword in response:
                relative_day = get_relative_day(keyword)
                response = response.replace(keyword, "").strip()
                break

        if relative_day:
            if response:
                # If there's a specific day mentioned, calculate from the relative day
                next_day = get_next_weekday(response)
                if next_day <= relative_day:
                    next_day += timedelta(days=7)
                return (next_day.strftime("%Y-%m-%dT%H:%M:%S"))
            else:
                return (relative_day.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            # Remove the keyword from the input if it exists
            for keyword in ["next", "coming", "upcoming"]:
                response = response.replace(keyword, "").strip()

            # Get the next occurrence of the specified day
            next_day = get_next_weekday(response, use_next)
            return next_day.strftime("%Y-%m-%dT%H:%M:%S")


  else:
    date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4}), (Morning|Afternoon|Evening|Night)\b'
    # Search for the pattern in the text
    
    try:
      match = re.search(date_pattern, response)
      # print("value in the date_time",match.groups())
      month, day, year, time_of_day = match.groups()
      date_str = f"{month} {day}, {year}"

      # Convert month_day_year to a datetime object
      datetime_obj = datetime.datetime.strptime(date_str, '%B %d, %Y')

      # Define time mappings
      time_mappings = {"Morning": 10,
                       "Afternoon": 15,
                       "Evening": 18,
                       "Night": 21}

      # Add the appropriate hour to the datetime object
      hour = time_mappings.get(time_of_day, 12)  # Default to noon if not found
      datetime_obj = datetime_obj.replace(hour=hour)

      # Convert to the target datetime
      target_datetime = datetime(2024, 7, 21, 8, 0, 0)

      # Print the results in ISO 8601 format
      print(f"Extracted datetime: {datetime_obj.isoformat()}")
      print(f"Target datetime: {target_datetime.isoformat()}")
      return datetime_obj.isoformat()

    except AttributeError:
      date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})\b'
      time_pattern = r'\b(\d{1,2}) (AM|PM)\b'
      period_pattern = r'\b(Morning|Afternoon|Evening|Night)\b'

      # Search for the date
      date_match = re.search(date_pattern, response)
      # Search for the time
      time_match = re.search(time_pattern, response)
      # Search for the period
      period_match = re.search(period_pattern, response)

      if date_match:
          month, day, year = date_match.groups()
          date_str = f"{month} {day}, {year}"

          # Convert month_day_year to a datetime object
          datetime_obj = datetime.strptime(date_str, '%B %d, %Y')

          if time_match:
              hour, am_pm = time_match.groups()
              hour = int(hour)
              if am_pm == 'PM' and hour != 12:
                  hour += 12
              elif am_pm == 'AM' and hour == 12:
                  hour = 0

              datetime_obj = datetime_obj.replace(hour=hour)
          else:
              # Define time mappings if no explicit time is provided
              time_mappings = {
                  "Morning": 9,
                  "Afternoon": 15,
                  "Evening": 18,
                  "Night": 21}

              if period_match:
                  period = period_match.group(0)
                  hour = time_mappings.get(period, 12)  # Default to noon if not found
                  datetime_obj = datetime_obj.replace(hour=hour)

          return datetime_obj.isoformat()
      else:
        return None


# Tool to book appointment
def book_appointment(request,auth_token, FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time):
    try:
        request.session['book_appointment']
    except:
        request.session['book_appointment']='True'
    


    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {auth_token}'}

    # Step 1: Get the list of locations
    get_locations_url = "https://iochatbot.maximeyes.com/api/location/GetLocationsChatBot"
    try:
        locations_response = requests.get(get_locations_url, headers=headers)
        locations_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching locations: {e}")
        return
    if locations_response.status_code != 200:
        return f"Failed to get locations. Status code: {locations_response.status_code}"
    try:
        locations = locations_response.json()
    except ValueError:
        return "Failed to parse locations response as JSON."

    print("Available locations:")
    result=''
    for idx, location in enumerate(locations):
        result+=f"{idx + 1}: {location['Name']} (ID: {location['LocationId']})\n"
        #print(f"{idx + 1}: {location['Name']} (ID: {location['LocationId']})")
    if request.session['book_appointment']=='True':  
       result=f" Choose a location by entering the ID: {result}"
       return result
    # location_id = transform_input(f"{request.session['book_appointment']}")
    location_id = request.session['book_appointment']
    #print(request.session['book_appointment'],'===============================',location_id)

    
    if location_id:
      print(transform_input('Thanks for providing location'))
    # Step 2: Get the list of providers for the selected location
    get_providers_url = f"https://iochatbot.maximeyes.com/api/scheduledresource/GetScheduledResourcesChatBot?LocationId={location_id}"
    try:
        providers_response = requests.get(get_providers_url, headers=headers)
        providers_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching providers: {e}")
        return
    if providers_response.status_code != 200:
        return f"Failed to get providers. Status code: {providers_response.status_code}"
    try:
        providers = providers_response.json()
    except ValueError:
        return "Failed to parse providers response as JSON."
    try:
        print(request.session['provider_id'],'-----------------------------')
    except:
        request.session['provider_id']='True'
    
    print("Available providers:")
    result=''
    for idx, provider in enumerate(providers):
        #print('----------',f"{idx + 1}: {provider['Name']} (ID: {provider['ScheduleResourceId']})")
        result+=f"{idx + 1}: {provider['Name']} (ID: {provider['ScheduleResourceId']})\n"
    #print(request.session['provider_id'],'request.session["provider_id"]----------------')

    if request.session['provider_id']=='True':  
       
       result=f" {transform_input('Choose a provider by entering the ID: ')} {result}"
       return result
    # provider_id = input(transform_input("Choose a provider by entering the ID: "))
    provider_id = request.session['provider_id']

    # Step 3: Get the appointment reasons for the selected provider and location
    get_reasons_url = f"https://iochatbot.maximeyes.com/api/appointment/appointmentreasonsForChatBot?LocationId={location_id}&SCHEDULE_RESOURCE_ID={provider_id}"
    try:
        reasons_response = requests.get(get_reasons_url, headers=headers)
        reasons_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching appointment reasons: {e}")
        return
    if reasons_response.status_code != 200:
        return f"Failed to get appointment reasons. Status code: {reasons_response.status_code}"
    try:
        reasons = reasons_response.json()
    except ValueError:
        return "Failed to parse appointment reasons response as JSON."
    try:
        request.session['reason_id']
    except:
        request.session['reason_id']='True'
    print("Available reasons:")
    result=''
    for idx, reason in enumerate(reasons):
        #print(f"{idx + 1}: {reason['ReasonName']} (ID: {reason['ReasonId']})")
        result+=f"{idx + 1}: {reason['ReasonName']} (ID: {reason['ReasonId']})\n"
    if request.session['reason_id']=='True':  
       result=f" {transform_input('Choose a reason by entering the ID: ')} {result}"
       return result

    reason_id = request.session['reason_id']
    # reason_id = input(transform_input("Choose a reason by entering the ID: "))

    # Step 4: Get the open slots for the selected location, provider, and reason
    preferred = prefred_date_time_fun(prefred_date_time)
    print("prefred date time",preferred)

    # from_date = "2024-07-20T15:30:00"
    from_date = preferred
    print("from_date",from_date)
    get_open_slots_url = f"https://iochatbot.maximeyes.com/api/appointment/openslotforchatbot?fromDate={from_date}&isOpenSlotsOnly=true"
    try:
        open_slots_response = requests.get(get_open_slots_url, headers=headers)
        open_slots_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching open slots: {e}")
        return
    if open_slots_response.status_code != 200:
        return f"Failed to get open slots. Status code: {open_slots_response.status_code}"
    try:
        open_slots = open_slots_response.json()
    except ValueError:
        return "Failed to parse open slots response as JSON."
    try:
        request.session['slot_id']
    except:
        request.session['slot_id']='True'
    #print("Available open slots:")
    result=''
    for idx, slot in enumerate(open_slots):
        #print(f"{idx + 1}: {slot['ApptStartDateTime']} - {slot['ApptEndDateTime']} (ID: {slot['OpenSlotId']})")
        result+=f"{idx + 1}: {slot['ApptStartDateTime']} - {slot['ApptEndDateTime']} (ID: {slot['OpenSlotId']})"
    if request.session['slot_id']=='True':  
       result=f" {transform_input('Choose an open slot by entering the ID:  ')} {result}"
       return result
    open_slot_id = request.session['slot_id']
    # open_slot_id = input(transform_input("Choose an open slot by entering the ID: "))

    # Step 5: Send OTP
    send_otp_url = "https://iochatbot.maximeyes.com/api/common/sendotp"
    otp_payload = {
        "FirstName": FirstName,
        "LastName": LastName,
        "DOB": DOB,
        "PhoneNumber": PhoneNumber,
        "Email": Email}
    try:
        otp_response = requests.post(send_otp_url, json=otp_payload, headers=headers)
        otp_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending OTP: {e}")
        return
    if otp_response.status_code != 200:
        return f"Failed to send OTP. Status code: {otp_response.status_code}"
    
    try:
        request.session['otp']
    except:
        request.session['otp']='True'
    result=''
    if request.session['otp']=='True':  
       result=f" {transform_input('Enter the OTP received: ')} "
       return result
    open_slot_id = request.session['slot_id']
    #print("open_slot_id",open_slot_id)
    otp = request.session['otp']
    # otp = input(transform_input("Enter the OTP received: "))

    # Step 6: Validate OTP
    validate_otp_url = "https://iochatbot.maximeyes.com/api/common/checkotp"

    #otp velidation

    validate_otp_payload = otp_payload.copy()
    validate_otp_payload["OTP"] = otp
    try:
        validate_otp_response = requests.post(validate_otp_url, json=validate_otp_payload, headers=headers)
        validate_otp_response.raise_for_status()
        print(validate_otp_response)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while validating OTP: {e}")
        return
    if validate_otp_response.status_code != 200:
        return f"Failed to validate OTP. Status code: {validate_otp_response.status_code}"
    try:
        validation_result = validate_otp_response.json()
    except ValueError:
        return "Failed to parse OTP validation response as JSON."
    # if not validation_result.get("Isvalidated"):
    #     return "Invalid OTP. Please try again."

    # Step 7: Book the appointment
    # book_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/onlinescheduling"
    # Convert ApptDate to 'MM/DD/YYYY' format
  #  print()
    #appointment_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S").strftime("%m/%d/%Y")
    # parsed_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")

    # # Convert the datetime object to the desired format
    # appointment_date = parsed_date.strftime("%m/%d/%Y")
    # print(appointment_date)
    # book_appointment_payload = {
    #     "OpenSlotId": open_slot_id,
    #     "ApptDate": appointment_date,
    #     "ReasonId": reason_id,
    #     "FirstName": FirstName,
    #     "LastName": LastName,
    #     "PatientDob": DOB,
    #     "MobileNumber": PhoneNumber,
    #     "EmailId": Email}
    # print(book_appointment_payload,'book_appointment_payload')
    # try:
    #     book_appointment_response = requests.post(book_appointment_url, json=book_appointment_payload, headers=headers)
    #     book_appointment_response.raise_for_status()
    #     return book_appointment_response.json()
    # except requests.exceptions.RequestException as e:
    #     print(f"An error occurred while booking the appointment: {e}")
    #     if book_appointment_response.status_code != 200:
    #         return f"Failed to book appointment. Status code: {book_appointment_response.status_code}"

    return "Appointment scheduled successfully."

# Function to generate response using Hugging Face endpoint
def generate_response(prompt, max_length=512, num_return_sequences=1):
    api_url = "https://tpfuzx0pqdencyjo.us-east-1.aws.endpoints.huggingface.cloud"
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    return call_huggingface_endpoint(prompt, api_url, api_token)

# Function to interactively handle the user query
def verification_check(FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time):
  a=f"FirstName: {FirstName}\nLastName: {LastName}\nDOB: {DOB}\nPhoneNumber: {PhoneNumber}\nEmail: {Email}\nprefred_date_time: {prefred_date_time}"
  print(a)

def update_field(extracted_info, field, value):
    if value and value.lower() != "none":
        extracted_info[field] = value
    return extracted_info

def handle_user_query1(request,user_query):
    # Identify the user's intent
    print("user_query:", user_query)
    intent = identify_intent(user_query)
    print(f"Identified intent: {intent}")

    # If the intent is to book an appointment
    if "booking an appointment" in intent.lower() or "schedule appointment" in intent.lower() or "book" in intent.lower():
      extracted_info = fetch_info_openai(user_query)
      print(f"Extracted info: {extracted_info}")


      fields = ['FirstName', 'LastName', 'DateOfBirth', 'PhoneNumber', 'Email','Preferred date or time']
      # missing_fields = [field for field in fields if not extracted_info.get(field)]

      missing_fields = [field for field in fields if not extracted_info.get(field) or extracted_info.get(field).lower() == "none"]
      print("missing fields",missing_fields)
      if missing_fields:

        prompt = f"Please provide your {missing_fields}: " 
        user_response = transform_input(prompt)
        
        return user_response 
      
     
      while missing_fields or len(missing_fields)==0:
        while missing_fields:
            #print("missing fileds in userhandel",missing_fields)
            prompt = f"Please provide your {missing_fields}: " 
            user_response = transform_input(prompt)
            additional_info = fetch_info_openai(user_response)

            for key in extracted_info:
              
              if not extracted_info[key] and key in additional_info:
                  extracted_info[key] = additional_info[key]

            # if field == "Email" and not user_response.strip():
            #     extracted_info["Email"] = "test@gmail.com"
            # else:
            #     additional_info = fetch_info_openai(user_response)
            #     if field in additional_info:
            #         extracted_info = update_field(extracted_info, field, additional_info[field])

            if 'DateOfBirth' in extracted_info and extracted_info['DateOfBirth']:
                extracted_info['DateOfBirth'] = convert_date_format(extracted_info['DateOfBirth'])

            missing_fields = [field for field in fields if not extracted_info.get(field) or extracted_info.get(field).lower() == "none"]
            print(extracted_info)
            

        FirstName = extracted_info.get('FirstName')
        LastName = extracted_info.get('LastName')
        DOB = extracted_info.get('DateOfBirth')
        PhoneNumber = extracted_info.get('PhoneNumber')
        Email = extracted_info.get('Email')
        prefred_date_time = extracted_info.get('Preferred date or time')



        verification_check(FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time)


        # Get authentication token
        auth_token = get_auth_token(vendor_id, vendor_password, account_id, account_password)
        if not auth_token:
            return "Failed to authenticate."


        # Book the appointment
        
        book_appt = book_appointment(request,auth_token, FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time)
        
        return book_appt

  #If the intent is not related to booking, generate a response using the fine-tuned model
    else:
        response = generate_response(user_query)
        return response

# Main function to handle user query
# def handle_user_query_me(request, user_query):
#     # Identify the user's intent
#     print("user_query", user_query)
#     intent = identify_intent(user_query)
#     print(f"Identified intent: {intent}")

#     # Initialize required information if not already in session
#     if 'required_info' not in request.session:
#         request.session['required_info'] = {
#             'first_name': None,
#             'last_name': None,
#             'dob': None,
#             'mobile_number': None,
#             'email': None
#         }

#     # If the intent is to reschedule an appointment
#     if intent == "rescheduling":
#         missing_info = ask_for_missing_info(request, request.session['required_info'])
#         if missing_info:
#             return missing_info

#         # Extract required information
#         extracted_info = fetch_info(user_query)
#         print(type(extracted_info))
#         print(f"Extracted info: {extracted_info}")

#         # Update session with extracted information
#         request.session['required_info'].update(extracted_info)

#         # Check for any missing information
#         missing_info = ask_for_missing_info(request, request.session['required_info'])
#         if missing_info:
#             return missing_info

#         # Get authentication token
#         auth_token = get_auth_token(vendor_id, vendor_password, account_id, account_password)
#         if not auth_token:
#             return "Failed to authenticate."

#         # Reschedule the appointment and get open slots
#         required_info = request.session['required_info']
#         open_slots = reschedule_appointment(auth_token, required_info['last_name'], required_info['dob'])
#         return open_slots

#     # If the intent is not related to rescheduling, generate a response
#     else:
#         response = generate_response(user_query)
#         return response


def home(request):
    context = {
        "debug": settings.DEBUG,
        "django_ver": get_version(),
        # "python_ver": os.environ["PYTHON_VERSION"],
    }
    if request.method=='GET':
        try:
            del request.session['context']
            del request.session['book_appointment']
            del request.session['provider_id']
            del request.session['reason_id']
            del request.session['slot_id']
            del request.session['otp']
            del request.session['return_response']
        except:
            pass
    return render(request, "pages/home.html", context)
def func(request):
    resp=''
    
   
    if request.method=='POST':
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('input', '')
        querry=message
        print("querry",querry)
        if  request.session.get('book_appointment')=='True':
            try:
                request.session['book_appointment']=message
            except:
                print('Not able select location')
        if  request.session.get('provider_id')=='True':
            try:
                request.session['provider_id']=message
            except:
                print('Not able select privider')
        if  request.session.get('reason_id')=='True':
            try:
                request.session['reason_id']=message
            except:
                print('Not able select reason')
        if  request.session.get('slot_id')=='True':
            try:
                request.session['slot_id']=message
            except:
                print('Not able select slot')
        if  request.session.get('otp')=='True':
            try:
                request.session['otp']=message
            except:
                print('Not able select otp')
        resp=str(handle_user_query(request))
       
    # return render(request, "pages/home.html",{'resp':resp,'return_response':return_response})
    return JsonResponse({"response": resp})

def func1(request):
    data = json.loads(request.body.decode('utf-8'))
    message = data.get('input', '')
    #print(message,'----------------')
    return JsonResponse({"response": 'success'})

@csrf_exempt
@require_POST
def handle_user_query(request):
    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('input', '')
    except:
        message=''
    if request.method != 'POST':
        
        return JsonResponse({"error": "Invalid request method"}, status=405)
    try:
        request.session['context']= request.session['context'] +' '+message
    except:
        request.session['context']=message
    
    try:
        # data = json.loads(request.body)
        # print(f"Received data: {data}")
 
        # query = data.get("query")
        # if not query:
        #     return JsonResponse({"error": "Missing 'query' in request data"}, status=400)
 
        # if not isinstance(query, dict):
        #     return JsonResponse({"error": "Invalid JSON format, expected a JSON object for 'query'"}, status=400)
 
        input_message = request.session['context']
        if not input_message:
            return JsonResponse({"error": "Missing 'message' in 'query' data"}, status=400)

        response = handle_user_query1(request,input_message)
        return response
        # return JsonResponse({"response": response})
   
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except KeyError as e:
        print(f"Key error: {e}")
        return JsonResponse({"error": "Missing 'message' in 'query' data"}, status=400)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return JsonResponse({"error": "An internal error occurred"}, status=500)