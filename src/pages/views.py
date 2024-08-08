import os
from django import get_version
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from langchain.chat_models import ChatOpenAI
import time
import json
import requests
from datetime import datetime, timedelta,date
import re
import uuid
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import json
import requests

load_dotenv('.env.development')
memory = ConversationBufferMemory()
vendor_id = os.getenv("VendorId")
vendor_password = os.getenv("VendorPassword")
account_id = os.getenv("AccountId")
account_password = os.getenv("AccountPassword")
hugging_face_api_token = os.getenv("hugging_face_api_token")
# print(hugging_face_api_token,'hugging_face_api_token')
api_url = "https://dgltkszlxd0qaoge.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Authorization": "Bearer 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",  
    "Content-Type": "application/json"
}

# Define the function to call the Hugging Face endpoint
def call_huggingface_endpoint(prompt, api_url, api_token,  max_new_tokens,  do_sample, temperature, top_p ,max_length=512,retries=1, backoff_factor=0.3):
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "parameters": {
            "max_length":max_length,
             "max_new_tokens":max_new_tokens,
             "do_sample":do_sample,
            "temperature":temperature,
            "top_p":top_p,
            
        }
    }
    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            try:
                response=(response.json()[0]["generated_text"]).split('Response:')[1]
            except:
                response=(response.json()[0]["generated_text"])
            return response
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise e


# Function to convert datetime to desired formant for api call
# def convert_date_format(date_str):
#     # Define date formats
#     formats = [
#         "%Y-%m-%d",        # ISO format
#         "%d-%m-%Y",        # European format
#         "%m-%d-%Y",        # US format
#         "%B %d, %Y",       # Full month name with day and year
#         "%b %d, %Y",       # Abbreviated month name with day and year
#         "%d %B %Y"         # Day with full month name and year
#     ]
#     # Remove ordinal suffixes (e.g., "st", "nd", "rd", "th")
#     date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

#     # Try parsing with each format
#     for fmt in formats:
#         try:
#             # Parse the date
#             parsed_date = datetime.strptime(date_str, fmt)
#             # Return in YYYY-MM-DD format
#             return parsed_date.strftime("%Y-%m-%d")
#         except ValueError:
#             continue

#     # If no formats match, return the original string
#     return date_str

# function to extract user info


def fetch_info(response):
    modelPromptForAppointment = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        text: {response}
        Extract the following information from given text : FirstName, LastName, DateOfBirth, Email, PhoneNumber and PreferredDateOrTime  if available ,determine what could be the information
        also note that if any fields in the information is not there return it as empty field. 
        instruction:
        -read the full information carefully
        -please provide indexing
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    try:
        result = call_huggingface_endpoint(modelPromptForAppointment, api_url, hugging_face_api_token,256 ,False  ,0.1 ,0.9)
        result=result[len(modelPromptForAppointment):].strip().replace('*','').replace('Not available','').replace('not available','').replace('(Not available)','')
        print('fetched info srting',result)
        data_dict = {}
        for line in result.split('\n'):
            if ':' in line:
                pattern = r"(\d+)\.\s*([a-zA-Z\s]+):\s*(.+)"
                matches = re.findall(pattern, line)
                if matches:
                    key, value = matches[0][1].strip(), matches[0][2].strip()
                    data_dict[key] = (value).replace('(empty field)','').replace('not Available','').replace('(not available)','').replace('Not mentioned','').replace('Not Mentioned','')
 
        
        return data_dict
    except Exception as e:
        print(f"Error extracting information: {e}")
        return {}
# Function to identify intent
def identify_intent(user_query):
    
    model_prompt_for_intent = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        "Identify the primary intent of the following user query:n"
        The intent must be one of the following options:
        - Greeting
        - Booking an appointment
        - Rescheduling an appointment
        - Canceling an appointment
        - Requesting static information (e.g., office hours, address)
        - Other inquiries
        Provide only the primary identified intent from the list above. Do not add anything extra..
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    retries = 3
    backoff_factor = 0.3
 
    for attempt in range(retries):
        try:
            
            intent = call_huggingface_endpoint(model_prompt_for_intent, api_url, hugging_face_api_token,256 ,False  ,0.1 ,0.9)
            
            intent = intent[len(model_prompt_for_intent):].strip().split('\n')[0]  # Extract only the first line of the output
            
            return intent
            # return intent
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Connection error occurred: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {retries} attempts: {e}")
                return "Error: Unable to identify intent due to a connection error."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "Error: An unexpected error occurred while identifying intent."

# funtion to find intent for practice/custom questions
def identify_intent_practice_question(user_query,data):

    print('identify_intent_practice_question')
    print("--",data,"-static data-")
    model_prompt_for_static_queries = (
    f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Data available for reference: {data}
        Instructions:
        1. Analyze the user's query to determine its intent.
        2. If the query requests information that is available in the provided data, respond with the relevant information from the data.
        3. If the query does not match any information available, respond with "Please provide valid information."
        4. If the query does not fit into the above categories, respond with "I'm sorry, I can't provide that information. Can you ask about something else related to our services or appointments?"
        5. If you don't understand the query, ask for clarification rather than returning the same text.
        6. if it is related to any glasses ques respond with "I'm sorry, I can't provide that information. Can you ask about something else related to our services or appointments?"
        please follow the above instructions carefully.
        Avoid formal language; aim for a friendly and human-like tone.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
        
    response = call_huggingface_endpoint(model_prompt_for_static_queries, api_url, hugging_face_api_token,150 ,True  ,0.6 ,0.9)
    response=response[len(model_prompt_for_static_queries):].strip()
    return response

# funtion for edit user info
def edit_msg(request):
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')
    user_response=''
    fields = ['FirstName', 'LastName', 'DateOfBirth', 'PhoneNumber', 'Email', 'PreferredDateOrTime']
    
    try:
        request.session[f'edit_msg{session_id}']
    except:
        request.session[f'edit_msg{session_id}']='True'
 
    if request.session[f'edit_msg{session_id}']=='True':
        # Extract the current context from the session
        current_context = request.session.get('context', '{}')   
        # Convert context from JSON string to dictionary if necessary
        
        # List of all possible fields
        fields = ['FirstName', 'LastName', 'DateOfBirth', 'PhoneNumber', 'Email', 'PreferredDateOrTime']
   
        # Ask which fields the user wants to edit
        prompt = "Which of the following fields would you like to edit? " + ", ".join(fields) 
        user_response = transform_input(prompt)
        return user_response
    else:

        edit_msg=request.session[f'edit_msg{session_id}']
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        context=request.session[f'context{session_id}']
        print("gjhgjhgjhgjhgjhgjhgjhvb ",edit_msg)
        
        response_content_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                "This is my current information: {context}\n"

                Instructions:
                - If the user specifies which information to change, provide the updated context with those changes applied only and return the full context donot delete anything.
                - If the user want to change information but does not specify any changes, respond with only word "no" doesnot ask  question. for example user ask i want to change
                - If the user does not want to change anything, respond with "yes"
                - if user want to say want to changes but mention field name and doensot give what to change then respond with only word "no" doesnot ask  question.
                please note that donot remove numerical values at end of context.
                please follow the above instructions carefully.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {edit_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
        response_content = call_huggingface_endpoint(response_content_prompt, api_url, hugging_face_api_token,256 ,False  ,0.09 ,0.9)
        response_content = response_content[len(response_content_prompt):].strip()
        print(response_content,"sdfhuh")
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        request.session[f'context{session_id}']=response_content
        del request.session[f'edit_msg{session_id}']
        del request.session[f'confirmation{session_id}']
        del request.session[f'fields{session_id}']
        
        return handle_user_query_postprocess(request,response_content)

# funtion to edit user context 
def confirmation_intent(request):
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')
    context = request.session[f'context{session_id}']
    user_input = request.session.get(f'confirmation{session_id}', '')
    request.session[f'context{session_id}']=context.replace(user_input,'')
    context = request.session[f'context{session_id}']
    response_content_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                "This is my current information: {context}\n"
                Instructions:
                - If the user specifies which information to change, provide the updated context with those changes applied only and return the full context donot delete anything.
                - If the user want to change information but does not specify any changes, respond with only word "no" doesnot ask  question. for example user ask i want to change
                - If the user does not want to change anything, respond with "yes"
                - if user want to say want to changes but mention field name and doensot give what to change then respond with only word "no" doesnot ask  question.
                please note that donot remove numerical values at end of context.
                please follow the above instructions carefully.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    response_content = call_huggingface_endpoint(response_content_prompt, api_url, hugging_face_api_token,256 ,False  ,0.9 ,0.9)
    response_content = response_content[len(response_content_prompt):].strip()
    if response_content == 'yes':
        request.session[f'confirmation{session_id}'] = 'yes'
    elif response_content == 'no':
        request.session[f'confirmation{session_id}'] = 'no'
    else:
        request.session[f'confirmation{session_id}'] = 'True'
        request.session[f'context{session_id}'] = response_content
        del request.session[f'fields{session_id}']
    
    return handle_user_query_postprocess(request, response_content)

# funtion to transform greeting response

def transform_input_greeting(user_input):

    # Construct a prompt to rephrase the user input
    modelPromptForAppointment = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are a helpful Eyecare assistant for MaximCaye Care. Start with a simple greeting and assist the user related to appointment and Do not anything from your end.<|eot_id|><|start_header_id|>user<|end_header_id|>

            {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Query the model
    response=call_huggingface_endpoint(modelPromptForAppointment, api_url, hugging_face_api_token,256 ,False  ,0.9 ,0.9)
    # response = query_llama3(modelPromptForAppointment)
    response = response[len(modelPromptForAppointment):].strip()

    return response

# function for transforming the responses
def format_appointment_date(date):
    current_date=datetime.now().strftime("%B %d, %Y")

    day=datetime.now().strftime('%A')
    model_prompt_for_appointment = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Instructions:
                -current date is : {current_date} 
                -day today is : {day}
            calculate the date according to users querry: {date}
            change the date in this format:"%m/%d/%Y" or  "month/day/year" 
            and return date in mm/dd/yyyy format only
            example: mm/dd/yyyy
            please provide only response
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {date}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    response = call_huggingface_endpoint(model_prompt_for_appointment, api_url, hugging_face_api_token,256 ,False  ,0.9 ,0.9)
    response_content = response[len(model_prompt_for_appointment):].strip()
    return response_content


def transform_input(input_text):
    # Define a list of prompts to transform the input text
    modelPromptTotransform = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        how would you ask this {input_text} as a question in a friendly and conversational tone related to appointment? 
        give only one option.
        user
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
     
    # Call the API using the latest method
    response = call_huggingface_endpoint(modelPromptTotransform, api_url, hugging_face_api_token,256 ,False  ,0.9 ,0.9)
    # response = query_llama3(modelPromptTotransform)
    # print(response,'transformed response-----')
    response = response[len(modelPromptTotransform):].strip()
    print(response,'transformed response-----')
    return response

# funtion to update user info
def update_info(extracted_info, additional_info):
    for key, value in additional_info.items():
        if (value and value.lower() != "none")or(value and value.lower() !=  'Not available'):
            extracted_info[key] = value

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
        print(response_json,"??????????????????????")
 
        if response_json.get('IsToken'):
            return response_json.get('Token')
        else:
            return f"Error message: {response_json.get('ErrorMessage')}"
    except requests.RequestException as e:
        return f"Authentication failed: {str(e)}"
    except json.JSONDecodeError:
        return "Failed to decode JSON response"




# Tool to book appointment
def book_appointment(request, auth_token, FirstName, LastName, DOB, PhoneNumber, Email, prefred_date_time):
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')
    try:
        request.session[f'book_appointment{session_id}']
    except:
        request.session[f'book_appointment{session_id}'] = 'True'
 
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
    result = ''
    valid_ids=''
    for idx, location in enumerate(locations):
        result += f"{idx + 1}: {location['Name']} (ID: {location['LocationId']})\n"
        valid_ids+= ' '+ (str(location['LocationId']).strip())
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')    
    if request.session[f'book_appointment{session_id}'] == 'True':

        result = f" Choose a location by entering the ID : {result}"
        return result
    if str(request.session[f'book_appointment{session_id}']) in valid_ids:
        location_id = request.session[f'book_appointment{session_id}']
    else:
        request.session[f'book_appointment{session_id}'] = 'True'

        return f"""Invalid location ID. 
                choose a valid location by entering the ID. {result} """    
    if location_id:
        print("Thanks for providing location")
   
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
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        print(request.session[f'provider_id{session_id}'], '-----------------------------')
    except:
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        request.session[f'provider_id{session_id}'] = 'True'
   
    print("Available providers:")
    result = ''
    valid_ids=''
    for idx, provider in enumerate(providers):
        result += f"{idx + 1}: {provider['Name']} (ID: {provider['ScheduleResourceId']})\n"
        valid_ids+= ' '+ (str(provider['ScheduleResourceId']).strip())
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')
    if request.session[f'provider_id{session_id}'] == 'True':
        result = f"Choose a provider by entering the ID: {result}"
        return result
    
    if str(request.session[f'provider_id{session_id}']) in valid_ids:
        provider_id = request.session[f'provider_id{session_id}']
    else:
        request.session[f'provider_id{session_id}'] = 'True'
        return f"""Invalid provider ID. 
                choose a valid provider by entering the ID. {result} """
 
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
        request.session[f'reason_id{session_id}']
    except:
        request.session[f'reason_id{session_id}'] = 'True'
    print("Available reasons:")
    result = ''
    valid_ids=''
    for idx, reason in enumerate(reasons):
        result += f"{idx + 1}: {reason['ReasonName']} (ID: {reason['ReasonId']})\n"
        valid_ids+= ' '+ (str(reason['ReasonId']).strip())
    if request.session[f'reason_id{session_id}'] == 'True':
        result = f"Choose a reason by entering the ID: {result}"
        return result
    if str(request.session[f'reason_id{session_id}']) in valid_ids:
        reason_id = request.session[f'reason_id{session_id}']
    else:
        request.session[f'reason_id{session_id}'] = 'True'
        return f"""Invalid reason ID. 
                choose a valid reason by entering the ID. {result} """
 
    # Step 4: Get the open slots for the selected location, provider, and reason
    print(prefred_date_time,'prefred_date_time -----------------')
    preferred = format_appointment_date(prefred_date_time)

    print(type(preferred),'=========',preferred)
    if 'Date is in the past' in preferred:
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        message=request.session[f'context{session_id}']
        prompt = (
        f"""You are given a text with a placeholder for a preferred date and time. Your task is to remove the placeholder `{prefred_date_time}` from the text while keeping the rest of the content exactly as it is. Here is the text with the placeholder included:
            "{message}"
            Please remove the placeholder `{prefred_date_time}` and return the updated text without changing anything else."""   
        )
        chat_completion = call_huggingface_endpoint(prompt, api_url, hugging_face_api_token,256 ,True  ,0.5 ,0.9)
        message = chat_completion
        request.session[f'context{session_id}']=message
        print(message,'====================+')
        return 'Please provide a valid date time for Appointment'
    
    print("Preferred date time", preferred) 
    from_date = preferred
    print("From date", from_date)
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
        request.session[f'slot_id{session_id}']
    except:
        request.session[f'slot_id{session_id}'] = 'True'
    print("Available open slots:")
    result = ''
    valid_ids=[]
    for idx, slot in enumerate(open_slots):
        result += f"{idx + 1}: {slot['ApptStartDateTime']} - {slot['ApptEndDateTime']} (ID: {slot['OpenSlotId']})\n"
        valid_ids.append(str(slot['OpenSlotId']).strip())
    if request.session[f'slot_id{session_id}'] == 'True':
        result = f"Choose an open slot by entering the ID: {result}"
        return result
    if str(request.session[f'slot_id{session_id}']).strip() in valid_ids:
        open_slot_id = request.session[f'slot_id{session_id}']
    else:
        request.session[f'slot_id{session_id}'] = 'True'
        return f"""Invalid slot ID. 
                choose a valid slot by entering the ID. {result} """
    
    # Step 5: Confirm details with the user   
     # otp and confirmation shifted before location id and providers id 
    # Step 8: Book the appointment     
    try:
        request.session['appointment_scheduled']
    except:
        request.session['appointment_scheduled'] = 'True'
    result = ''
    if request.session['appointment_scheduled'] == 'True':
        book_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/onlinescheduling"
        # Convert ApptDate to 'MM/DD/YYYY' format    
        appointment_date = format_appointment_date(from_date)
        
        print(appointment_date)
        book_appointment_payload = {
            "OpenSlotId": open_slot_id,
            "ApptDate": appointment_date,
            "ReasonId": reason_id,
            "FirstName": FirstName,
            "LastName": LastName,
            "PatientDob": DOB,
            "MobileNumber": PhoneNumber,
            "EmailId": Email}
        print(book_appointment_payload,'book_appointment_payload')
        try:
            book_appointment_response = requests.post(book_appointment_url, json=book_appointment_payload, headers=headers)
            book_appointment_response.raise_for_status()
            print(book_appointment_response.json(),'book_appointment_response')
            if book_appointment_response.json()=='Appointment scheduled successfully':
                result = f"Your Appointment is scheduled, Thanks for choosing eyecare location!, Is there anything i can help you with? "
                return result
                # return book_appointment_response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while booking the appointment: {e}")
            if book_appointment_response.status_code != 200:
                return f"Failed to book appointment. Status code: {book_appointment_response.status_code}"
        
        open_slot=''
        for idx, slot in enumerate(open_slots): 
            print(slot['OpenSlotId'],open_slot_id,'slot========')
            if  str(slot['OpenSlotId']).strip() == str(open_slot_id).strip():
                print(f"{slot['ApptStartDateTime']} - {slot['ApptEndDateTime']}")
                open_slot=f"{slot['ApptStartDateTime']} - {slot['ApptEndDateTime']}"

        book_appointment_payload = {
            "Time Slot ": open_slot,
            "Appt. Date": appointment_date,
            
            "FirstName": FirstName,
            "LastName": LastName,
            "PatientDob": DOB,
            "MobileNumber": PhoneNumber,
            "EmailId": Email}  
          
        result = f"""Your Appointment is scheduled, Thanks for choosing us , Here are the appointment details:
            
            Appt. Date: {appointment_date},
            Time Slot :{ open_slot},                  
            Name: {FirstName} {LastName} ,    
            MobileNumber: {PhoneNumber},
            EmailId: {Email}
        
            """
        print(result)
        # result= transform_input(result)
        result=result+'\n'+'Thanks, Have a great day! '
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        del request.session[f'context{session_id}']
        delete_session(request,session_id)
        return result    
    return "Thanks, Have a great day! "

def update_context(request,text):
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')
    context = request.session[f'context{session_id}']
    response_content_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                "This is my current information: {context}\n"
                Instructions:
                - If the user specifies which information to change, provide the updated context with those changes applied only and return the full context donot delete anything.
                
                please follow the above instructions carefully.
                <|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    response_content = call_huggingface_endpoint(response_content_prompt, api_url, hugging_face_api_token,256 ,False  ,0.09 ,0.9)
    response_content = response_content[len(response_content_prompt):].strip()
    delete_session(request,session_id)
    request.session[f'context{session_id}'] = response_content.replace('"','')
    return response_content
# Function to generate response using Hugging Face endpoint
def generate_response(user_query):

    prompt = (
    f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    instruction: You are a creative assistant for eye care services. You must ONLY provide information directly related to eye health, vision, and eye care services. If the user's query is not related to eye care, respond with EXACTLY this message: 'I apologize, but I can only answer questions related to eye care. If you have any eye-related questions, I'd be happy to help'
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
    response=call_huggingface_endpoint(prompt, api_url, hugging_face_api_token,256 ,False  ,0.9 ,0.9)
    response_content = response[len(prompt):].strip()
    return response_content
# Function to interactively handle the user query
def verification_check(FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time):
  

  a=f"FirstName: {FirstName}\nLastName: {LastName}\nDOB: {DOB}\nPhoneNumber: {PhoneNumber}\nEmail: {Email}\nprefred_date_time: {prefred_date_time}"


# funtion to validate email
def validate_email(email):
          # Regular expression pattern for a valid email address
          pattern = r'^[\w\.-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
          return re.match(pattern, email) is not None

# funtion to validate email         
def validate_phone(phone):
    # Regular expression pattern for a valid US phone number
    pattern = r'^\d{10}$|^\(\d{3}\) \d{3}-\d{4}$|^\(\d{3}\)-\d{3}-\d{4}$'
    return re.match(pattern, phone) is not None


def validate_date(request,preferred_date_time,DOB):
    current_date = datetime.now()
    preferred_date_time = datetime.strptime(preferred_date_time.strip(), '%m/%d/%Y')
    DOB = datetime.strptime(DOB.strip(), '%m/%d/%Y')
    print(current_date,preferred_date_time,DOB,type(preferred_date_time),(DOB))
    
    # Initialize validity flags
    preferred_valid = 'valid'
    dob_valid = 'valid'
    
    # Validate preferred_date_time
    if preferred_date_time < current_date:
        preferred_valid = 'not valid'
    
    # Validate DOB
    if DOB > current_date:
        dob_valid = 'not valid'
    
    # Return results based on the validity of both dates
    if preferred_valid == 'not valid' and dob_valid == 'not valid':
        text = "remove the date of birth and preferred date and time fro appointment from the context"
        print(update_context(request,text))
        return 'Please provide valid Appointment date and time and Date of birth '
    elif preferred_valid == 'not valid':
        text = " remove preferred date and time for appointment from the context"
        print(update_context(request,text))
        return 'Please provide valid Appointment date and time'
    elif dob_valid == 'not valid':
        text = "remove date of birth from the context"
        print(update_context(request,text))
        return 'Please provide valid Date of birth'
    else:
        return True
    
       

def handle_user_query_postprocess(request, user_query):
    data = json.loads(request.body.decode('utf-8'))
    session_id = data.get('session_id', '')

    try:
        # Check if intent is already stored in the session
        print(request.session.get(f'intent{session_id}'),"request.session.get(f'intent{session_id}")
        if not request.session.get(f'intent{session_id}'):
            intent = identify_intent(user_query)
            request.session[f'intent{session_id}'] = intent
            print('getting new intent')
        else:
            intent = request.session[f'intent{session_id}']
            print('getting old intent')
    except :
        # In case of an exception, identify intent and store it
        intent = identify_intent(request, user_query, session_id)
        print('getting old except intent')
        request.session[f'intent{session_id}'] = intent
    
    # Handle different intents
    if "greeting" in intent.lower():

        response = transform_input_greeting(user_query)
        delete_session(request, session_id)
        return response

    data = data.get('practice1_details', '')

    if "requesting static information" in intent.lower():
        print("Requesting static information----")
        static_response = identify_intent_practice_question(user_query, data)
        if static_response:
            delete_session(request, session_id)
            return static_response

    if any(keyword in intent.lower() for keyword in ["booking an appointment", "schedule appointment", "book"]):
        print('book appointments---------------')

        if not request.session.get(f'fields{session_id}'):
            extracted_info = fetch_info(user_query)
            print("regsjk0",extracted_info)
            fields = ['FirstName', 'LastName', 'DateOfBirth', 'PhoneNumber', 'Email', 'PreferredDateOrTime']
            missing_fields = [field for field in fields if not extracted_info.get(field) or extracted_info.get(field).lower() == "none"]

            print("missing fields", missing_fields, 'user_query', user_query)

            
            if 'Email' not in missing_fields:
                extracted_email = extracted_info.get('Email', '')
                if extracted_email and extracted_email not in ["none", '(empty)']:
                    if not validate_email(extracted_email):
                        prompt = f"Please provide a valid Email address. The email you provided is not valid."
                        user_response = transform_input(prompt)
                        message = request.session.get(f'context{session_id}', '')
                        message = message.replace(extracted_email, '')
                        request.session[f'context{session_id}'] = message
                        return user_response

            if 'PhoneNumber' not in missing_fields:
                extracted_phone = extracted_info.get('PhoneNumber', '')
                if extracted_phone and extracted_phone not in ["none", '(empty)']:
                    if not validate_phone(extracted_phone):
                        prompt = f"Please provide a valid Phone Number. The number you provided is not valid."
                        user_response = transform_input(prompt)
                        message = request.session.get(f'context{session_id}', '')
                        message = message.replace(extracted_phone, '')
                        request.session[f'context{session_id}'] = message
                        return user_response

            if len(missing_fields) == 1 and ('PhoneNumber' in missing_fields or 'Email' in missing_fields):
                missing_fields = []

        else:
            extracted_info = request.session[f'fields{session_id}']
            
            extracted_info = extracted_info.replace("'", '"').replace('(not provided)','').replace('(Not provided)','')
            extracted_info = json.loads(extracted_info)
            
            fields = ['FirstName', 'LastName', 'DateOfBirth', 'PhoneNumber', 'Email', 'PreferredDateOrTime']
            missing_fields = [field for field in fields if not extracted_info.get(field) or extracted_info.get(field).lower() == "none"]
            if 'Email' not in missing_fields:
                extracted_email = extracted_info.get('Email', '')
                if extracted_email and extracted_email not in ["none", '(empty)']:
                    if not validate_email(extracted_email):
                        prompt = f"Please provide a valid Email address. The email you provided is not valid."
                        user_response = transform_input(prompt)
                        message = request.session.get(f'context{session_id}', '')
                        message = message.replace(extracted_email, '')
                        request.session[f'context{session_id}'] = message
                        return user_response

            if 'PhoneNumber' not in missing_fields:
                extracted_phone = extracted_info.get('PhoneNumber', '')
                if extracted_phone and extracted_phone not in ["none", '(empty)']:
                    if not validate_phone(extracted_phone):
                        prompt = f"Please provide a valid Phone Number. The number you provided is not valid."
                        user_response = transform_input(prompt)
                        message = request.session.get(f'context{session_id}', '')
                        message = message.replace(extracted_phone, '')
                        request.session[f'context{session_id}'] = message
                        return user_response

            if len(missing_fields) == 1 and ('PhoneNumber' in missing_fields or 'Email' in missing_fields):
                missing_fields = []
        print(extracted_info,'extracted_info')
        if missing_fields:
            prompt = f" Please provide your {', '.join(missing_fields)}: "
            print('missing_fields+++')
            user_response = transform_input(prompt)
            return user_response
        else:
            request.session[f'fields{session_id}'] = str(extracted_info)
        
        while missing_fields:
            prompt = f"Please provide your {missing_fields}: " 
            user_response = transform_input(prompt)
            additional_info = fetch_info(user_response)    
            for key in extracted_info:
              
              if not extracted_info[key] and key in additional_info:
                  extracted_info[key] = additional_info[key]    
            if 'DateOfBirth' in extracted_info and extracted_info['DateOfBirth']:
                extracted_info['DateOfBirth'] = format_appointment_date(extracted_info['DateOfBirth'])    
            missing_fields = [field for field in fields if not extracted_info.get(field) or extracted_info.get(field).lower() == "none"]
            print(extracted_info)
        print((extracted_info),'extracted info :')        
        FirstName = extracted_info.get('FirstName')
        LastName = extracted_info.get('LastName')
        DOB = extracted_info.get('DateOfBirth')
        PhoneNumber = extracted_info.get('PhoneNumber')
        Email = extracted_info.get('Email')
        prefred_date_time = extracted_info.get('PreferredDateOrTime')    
        verification_check(FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time)    
        # Get authentication token
        auth_token = get_auth_token(vendor_id, vendor_password, account_id, account_password)
        if not auth_token:
            return "Failed to authenticate."
        
        if 'confirmation' in request.session:
            if request.session[f'confirmation{session_id}'].lower() == 'no':
                edit_response = edit_msg(request)
                print("led",edit_response)
                return edit_response    
        # Book the appointment
        print(prefred_date_time,'prefred_date_time -----------------')
        preferred = format_appointment_date(prefred_date_time)
        
        
        prefred_date_time=preferred
        DOB=format_appointment_date(DOB)
        print(DOB,'dob=======--+++')
        validation=validate_date(request,prefred_date_time,DOB)
        if validation != True:
            return validation

        print(type(preferred),'=========',preferred)
        if 'Date is in the past' in preferred:
            data = json.loads(request.body.decode('utf-8'))
            session_id = data.get('session_id', '')
            message=request.session[f'context{session_id}']
            prompt = (
            f"""You are given a text with a placeholder for a preferred date and time. Your task is to remove the placeholder `{prefred_date_time}` from the text while keeping the rest of the content exactly as it is. Here is the text with the placeholder included:
                "{message}"
                Please remove the placeholder `{prefred_date_time}` and return the updated text without changing anything else."""   
            )
            
            message = call_huggingface_endpoint(prompt, api_url, hugging_face_api_token,256 ,False  ,0.5 ,0.9)
            request.session[f'context{session_id}']=message
            print(message,'====================+')
            return 'Please provide a valid date time for Appointment'  
        
        
        #step for otp and confirmation----
        
        
        confirmation_message = (
            f"Here are the details of your appointment:\n"
            
            f"Date and Time: {preferred}\n"
            f"Name: {FirstName} {LastName}\n"
            f"DOB: {DOB}\n"
            f"Phone: {PhoneNumber}\n"
            f"Email: {Email}\n"
            f"Is this information correct? (yes/no)"
        )
        try:
            print(request.session[f'confirmation{session_id}'],'======================')
        except:
            request.session[f'confirmation{session_id}'] = 'True'
        
        if request.session[f'confirmation{session_id}'] == "True":
            return confirmation_message
        
        # Step 6: If user confirms, send OTP. Only after user confirms, proceed with OTP
        
        
        if request.session[f'confirmation{session_id}'] == 'yes':
            send_otp_url = "https://iochatbot.maximeyes.com/api/common/sendotp"
            # DOB=prefred_date_time_fun(DOB)
            # DOB=format_appointment_date(DOB)
            otp_payload = {
                "FirstName": FirstName,
                "LastName": LastName,
                "DOB": DOB,
                "PhoneNumber": PhoneNumber,
                "Email": Email
            }
            headers = {
                    'Content-Type': 'application/json',
                    'apiKey': f'bearer {auth_token}'}
            print(otp_payload,'otp_payload')
            try:
                otp_response = requests.post(send_otp_url, json=otp_payload, headers=headers)
                otp_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while sending OTP: {e}")
                return
            if otp_response.status_code != 200:
                return f"Failed to send OTP. Status code: {otp_response.status_code}"
    
            try:
                request.session[f'otp{session_id}']
            except:
                request.session[f'otp{session_id}'] = 'True'
            result = ''
            if request.session[f'otp{session_id}'] == 'True':
                result = f"Enter the OTP received: "
                return result
            otp = request.session[f'otp{session_id}']
    
            # Step 7: Validate OTP
            validate_otp_url = "https://iochatbot.maximeyes.com/api/common/checkotp"
    
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
            if not validation_result.get("Isvalidated"):
                request.session[f'otp{session_id}'] = 'True'
                return "Invalid OTP. Please try again."
            
        elif request.session[f'confirmation{session_id}'] == 'no':
            msg=edit_msg(request)
            return msg
        
        else :
            
            context = confirmation_intent(request)
            return context
    

        # step to get location id provides id 
        book_appt = book_appointment(request,auth_token, FirstName, LastName, DOB, PhoneNumber, Email,prefred_date_time)
        
        try:
            data=json.loads(request.body.decode('utf-8'))
            session_id = data.get('session_id', '')
            if 'Appointment scheduled successfully' in book_appt:
                delete_session(request,session_id)
                print('Appointment scheduled successfully---')
                pass
        except:
            pass    
        return book_appt
    
  #    f the intent is not related to booking, generate a response using the fine-tuned model
    else:
        response = generate_response(user_query)
        print(response,'response111--------')
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        delete_session(request,session_id)
        return response

def delete_session(request,session_id):
    # data = json.loads(request.body.decode('utf-8'))
    # session_id = data.get('session_id', '')
    

        try:
            del request.session[f'context{session_id}']
        except:
            print('Not able to delete context')
        try:
            del request.session[f'book_appointment{session_id}']
        except:
            print('Not able to delete book_appointment')
        try:
            del request.session[f'provider_id{session_id}']
        except:
            print('Not able to delete provider_id')
        try:
            del request.session[f'reason_id{session_id}']
        except:
            print('Not able to delete reason_id')
        try:
            del request.session[f'slot_id{session_id}']
        except:
            print('Not able to delete slot_id')
        try:
            del request.session[f'otp{session_id}']
        except:
            print('Not able to delete otp')
        try:
            del request.session[f'confirmation{session_id}']
        except:
            print('Not able to delete confirmation')
        try:
            del request.session[f'edit_msg{session_id}']
        except:
            print('Not able to edit_msg')
        try:
            del request.session[f'return_response{session_id}']
        except:
            print('Not able to delete return_response')
        try:
            del request.session[f'appointment_scheduled{session_id}']
        except:
            print('Not able to delete appointment_scheduled')
        try:
            del request.session[f'intent{session_id}']
        except:
            print('Not able to delete appointment_scheduled')
        try:
            del request.session[f'fields{session_id}']
        except:
            print('Not able to delete appointment_scheduled')

        
        return request

def home(request):
    # if 'session_id1' in request.session:
    #     session_id=request.session['session_id1']
    # else:

    request.session['session_id1'] = str(uuid.uuid4())
    session_id=request.session['session_id1'] 

    # request.session._session_key_prefix = 'view1'
    context = {
        'session_id':session_id,
        "debug": settings.DEBUG,
        "django_ver": get_version(),
        # "python_ver": os.environ["PYTHON_VERSION"],
    }
    if request.method=='GET':
        
        delete_session(request,session_id)
       
    return render(request, "pages/home.html", context)
def home2(request):
    # request.session._session_key_prefix = 'view2'
    # if 'session_id2' in request.session:
    #     session_id=request.session['session_id2']
    # else:

    request.session['session_id2'] = str(uuid.uuid4())
    session_id=request.session['session_id2'] 

    context = {
        'session_id':session_id,
        "debug": settings.DEBUG,
        "django_ver": get_version(),
        # "python_ver": os.environ["PYTHON_VERSION"],
    }
    if request.method=='GET':
        delete_session(request,session_id)
    return render(request, "pages/home2.html", context)
    
@csrf_exempt
def func(request):
    resp=''
    memory = ConversationBufferMemory()
    if request.method=='POST':
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        message = data.get('input', '')
        querry=message
        
        
        if  request.session.get(f'book_appointment{session_id}')=='True':
            try:
                request.session[f'book_appointment{session_id}']=message
            except:
                print('Not able select location')
        
        if  request.session.get(f'provider_id{session_id}')=='True':
            try:
                request.session[f'provider_id{session_id}']=message
            except:
                print('Not able select privider')
        if  request.session.get(f'reason_id{session_id}')=='True':
            try:
                request.session[f'reason_id{session_id}']=message
            except:
                print('Not able select reason')
        if  request.session.get(f'slot_id{session_id}')=='True':
            try:
                request.session[f'slot_id{session_id}']=message
            except:
                print('Not able select slot')
        if  request.session.get(f'confirmation{session_id}')=='True':
            try:
                request.session[f'confirmation{session_id}']=message
            except:
                print('Not able to confirm')
        if  request.session.get(f'otp{session_id}')=='True':
            try:
                request.session[f'otp{session_id}']=message
            except:
                print('Not able select otp')
        if  request.session.get(f'edit_msg{session_id}')=='True':
            try:
                request.session[f'edit_msg{session_id}']=message
            except:
                print('Not able edit_msg')
        if  request.session.get('edit_msg')=='True':
            try:
                request.session[f'edit_msg{session_id}']=message
            except:
                print('Not able edit_msg')
        
        resp=str(handle_user_query(request))
       
    # return render(request, "pages/home.html",{'resp':resp,'return_response':return_response})
    return JsonResponse({"response": resp})



@csrf_exempt
@require_POST
def handle_user_query(request):
    # memory = ConversationBufferMemory()
    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get('input', '')
        session_id = data.get('session_id', '')
    except:
        message=''
        session_id = 0
    # print(session_id,'session_id',message)
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method"}, status=405)
    try:
        
        request.session[f'context{session_id}']= request.session[f'context{session_id}'] +' '+ message
        print('editing',request.session[f'context{session_id}'],'-----',message)
    except:
        request.session[f'context{session_id}']=message
    print(session_id,'session_id',message,'message', request.session[f'context{session_id}'],f" request.session[f'context{session_id}']")
    try:
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get('session_id', '')
        input_message = request.session[f'context{session_id}']
        if not input_message:
            return JsonResponse({"error": "Missing 'message' in 'query' data"}, status=400)

        print(input_message)

        response = handle_user_query_postprocess(request,input_message)

        # memory.save_context({"User": input_message}, {"Assistant": response})


        if response=='none' or response==None:
            data = json.loads(request.body.decode('utf-8'))

            response=f"Please contact :{data.get('practice_email', '')} "
            return response
        

        return response
   
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except KeyError as e:
        print(f"Key error: {e}")
        return JsonResponse({"error": "Missing 'message' in 'query' data"}, status=400)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return JsonResponse({"error": "An internal error occurred"}, status=500)
