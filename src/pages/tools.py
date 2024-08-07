
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import re
import requests
import os
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (format_to_openai_tool_messages,)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.tools import Tool
from langchain.agents.output_parsers import (ReActJsonSingleInputOutputParser,)
from langchain.agents.format_scratchpad import format_log_to_str
from langchain import hub
from langchain.agents import AgentExecutor, load_tools, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (  ReActJsonSingleInputOutputParser,)
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain.core.messages import HumanMessage, AIMessage
# from . model_loader import *

###hugging_face_api_token = "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"
###HUGGINGFACEHUB_API_TOKEN = "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"

#https://python.langchain.com/v0.1/docs/integrations/chat/huggingface/
#https://python.langchain.com/v0.2/docs/how_to/#embedding-models
#https://medium.com/@minekayaa/multi-agent-systems-langgraph-63c1abb3e242

import os

from langchain_community.llms import HuggingFaceTextGenInference

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define your custom prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """Answer the following questions as best you can. You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action you will take to answer the question
            
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat 1 times)
                you should stop after one observation or one action and return  the information from the tool function
                i want Action: and Action Input:  also in return. 
                
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!
            
                Question: {input}
                Thought:{agent_scratchpad}"""
         ),
        
        ("user", "{input}"),
        
    ]
)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system", """Assistant is a large language model trained by OpenAI.
#               Assistant is designed to be able to assist with a wide range of tasks,
#                 from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
#                   Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
#                   It is able to process and understand large amounts of text, 
#                   and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
#                   Additionally, Assistant is able to generate its own text based on the input it receives, 
#                   allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
#                     Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable 
#                     insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a 
#                     conversation about a particular topic, Assistant is here to assist.
#                       TOOLS: ------ {tool_names}Assistant has access to the following tools: > get_location:  Useful for when you need to fetch location initailly if user wants to book appointment.
                          
#                     get_providers:  Useful for when you need to fetch providers based on location id given by the user location .
#                       To use a tool, please use the following format: 
#                     ``` Thought: Do I need to use a tool? Yes Action: the action to take,
#                       should be one of [{tools}] Action Input: the input to the action Observation:
#                         the result of the action ``` When you have a response to say to the Human, 
#                       or if you do not need to use a tool, you MUST use the format: ``` 
#             Thought: Do I need to use a tool? No AI: [your response here] ``` B
#             egin!  New input: {input} {agent_scratchpad}"""
#          ),
        
#         ("user", "{input}"),
        
#     ]
# )



def format_log_to_messages(intermediate_steps):
    messages = []
    for step in intermediate_steps:
        if step["type"] == "human":
            messages.append(HumanMessage(content=step["content"]))
        elif step["type"] == "ai":
            messages.append(AIMessage(content=step["content"]))
    return messages


ENDPOINT_URL = "https://zzjxmo3ypzlaxmsp.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv('hugging_face_api_token') 

llm = HuggingFaceTextGenInference(
    inference_server_url=ENDPOINT_URL,
    max_new_tokens=512,
    top_k=20,
    temperature=0.01,
    repetition_penalty=0.7,
    server_kwargs={
        "headers": {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
    },
)


from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

chat_model = ChatHuggingFace(llm=llm)


def get_auth_token() -> str:
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



@tool
def get_locations(token):
    """Get the list of locations for booking appointments"""
    # token=get_auth_token()
    print('token++++++++++++++++++++++++++++++=================================',get_auth_token())
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {get_auth_token()}'}

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
    for idx, location in enumerate(locations):
        print(f"{idx + 1}: {location['Name']} (ID: {location['LocationId']})")
    # location_id = input("Choose a location by entering the ID: ")
    # if location_id:
      # print('Thanks for providing location')
    return [{'Name': 'Hillsboro', 'LocationId': 1}, {'Name': 'Beaverton', 'LocationId': 3}]
    # return locations
@tool
def get_providers(location_id,token):
    """Get the list of providers for a specific location"""
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
    print('location_id-',location_id)
    get_providers_url = f"https://iochatbot.maximeyes.com/api/scheduledresource/GetScheduledResourcesChatBot?LocationId={location_id}"
    try:
        providers_response = requests.get(get_providers_url, headers=headers)
        providers_response.raise_for_status()
        providers = providers_response.json()
        print(providers,'providers---')
        provider_list = [{"Name": provider['Name'], "ProviderId": provider['ScheduleResourceId']} for provider in providers]
        print("Available providers:", provider_list)
        return provider_list
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching providers: {e}")
        # return
    if providers_response.status_code != 200:
        return f"Failed to get providers. Status code: {providers_response.status_code}"
    try:
        providers = providers_response.json()
    except ValueError:
        return "Failed to parse providers response as JSON."

    # print("Available providers:")
    # for idx, provider in enumerate(providers):
    #     print(f"{idx + 1}: {provider['Name']} (ID: {provider['ScheduleResourceId']})")

    # provider_id = input("Choose a provider by entering the ID: ")
    # return [{'Name': 'Provider1', 'ProviderId': 101}, {'Name': 'Provider2', 'ProviderId': 102}]
@tool
def get_appointment_reasons(token, location_id, provider_id):
    """Get the list of appointment reasons for a specific provider and location"""
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
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

    print("Available reasons:")
    for idx, reason in enumerate(reasons):
        print(f"{idx + 1}: {reason['ReasonName']} (ID: {reason['ReasonId']})")

    reason_id = input("Choose a reason by entering the ID: ")
    if reason_id:
      print('Thanks for providing reason')
    return [{'Reason': 'Consultation', 'ReasonId': 201}, {'Reason': 'Follow-up', 'ReasonId': 202}]

def prefred_date_time_fun(response):
    print("response12",response)
    def get_next_weekday(day_name, use_next=False):
    # Dictionary to convert day names to weekday numbers
        days_of_week = {
            'monday': 0, 'mon': 0, 'Monday': 0, 'Mon': 0,
            'tuesday': 1, 'tues': 1,'Tuesday': 1, 'Tues': 1,
            'wednesday': 2, 'wed': 2,'Wednesday': 2, 'Wed': 2,
            'thursday': 3, 'thurs': 3,'Thursday': 3, 'Thurs': 3,
            'friday': 4, 'fri': 4,'Friday':4, 'Fri':4,
            'saturday': 5, 'sat': 5,'Saturday': 5, 'Sat': 5,
            'sunday': 6, 'sun': 6,'Sunday': 6, 'Sun': 6
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

    def get_upcoming_weekday(day_name):
        # Dictionary to convert day names to weekday numbers
        days_of_week = {
            'monday': 0, 'mon': 0, 'Monday': 0, 'Mon': 0,
            'tuesday': 1, 'tues': 1,'Tuesday': 1, 'Tues': 1,
            'wednesday': 2, 'wed': 2,'Wednesday': 2, 'Wed': 2,
            'thursday': 3, 'thurs': 3,'Thursday': 3, 'Thurs': 3,
            'friday': 4, 'fri': 4,'Friday':4, 'Fri':4,
            'saturday': 5, 'sat': 5,'Saturday': 5, 'Sat': 5,
            'sunday': 6, 'sun': 6,'Sunday': 6, 'Sun': 6
        }
        # Get today's date and the current weekday
        today = datetime.now()
        current_weekday = today.weekday()

        # Convert the day name to a weekday number
        target_weekday = days_of_week[day_name.lower()]

        # Calculate the number of days until the upcoming target weekday
        days_until_target = (target_weekday - current_weekday + 7) % 7

        # If the day is today and has not passed, use today's date
        if days_until_target == 0:
            next_weekday = today
        else:
            next_weekday = today + timedelta(days=days_until_target)

        return next_weekday

    def get_relative_day(keyword):
        today = datetime.now()
        if keyword == "tomorrow":
            return today + timedelta(days=1)
        elif keyword == "day after tomorrow":
            return today + timedelta(days=2)
        return None

    def extract_date_from_response(response):
        keywords = ["next", "coming", "upcoming", "tomorrow", "day after tomorrow"]
        use_next = any(keyword in response.lower() for keyword in ["next", "coming"])
        use_upcoming = "upcoming" in response.lower()

        # Check for "tomorrow" and "day after tomorrow"
        relative_day = None
        for keyword in ["tomorrow", "day after tomorrow"]:
            if keyword in response.lower():
                relative_day = get_relative_day(keyword)
                response = re.sub(keyword, "", response, flags=re.IGNORECASE).strip()
                break

        # Extract the day name from the response
        day_name_match = re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tues|Wed|Thurs|Fri|Sat|Sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues|wed|thurs|fri|sat|sun)\b', response, re.IGNORECASE)
        if day_name_match:
            day_name = day_name_match.group(0)
        else:
            day_name = None

        if relative_day:
            if day_name:
                # If there's a specific day mentioned, calculate from the relative day
                next_day = get_next_weekday(day_name)
                if next_day <= relative_day:
                    next_day += timedelta(days=7)
                if next_day < datetime.now():
                    return "Date is in the past"
                return next_day.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                return relative_day.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            # Remove the keyword from the input if it exists
            for keyword in ["next", "coming", "upcoming"]:
                response = re.sub(keyword, "", response, flags=re.IGNORECASE).strip()

            if day_name:
                if use_upcoming:
                    next_day = get_upcoming_weekday(day_name)
                else:
                    next_day = get_next_weekday(day_name, use_next)
                if next_day < datetime.now():
                    return "Date is in the past"
                return next_day.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                return "No valid day found in the response"
    if "next" in response.lower() or "coming" in response.lower() or "upcoming" in response.lower() or "tomorrow" in response.lower() or "next day" in response.lower():

        return extract_date_from_response(response)

    else:
        response=response.replace(',','')
        patterns = [
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4})  (Morning|Afternoon|Evening|Night)\b', '%B %d %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4})   (Morning|Afternoon|Evening|Night)\b', '%B %d %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4}) (Morning|Afternoon|Evening|Night)\b', '%B %d %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4})  (Morning|Afternoon|Evening|Night)\b', '%B %d %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4})\b', '%B %d %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{1,2}) (\d{4})\b', '%B %d %Y'),
            (r'(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{4}) (Morning|Afternoon|Evening|Night)\b', '%d %B %Y'),
            (r'(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|May|june|july|august|september|october|november|december) (\d{4})\b', '%d %B %Y'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}) (\d{4})\b','%B %d %Y'), # Added this line
            (r'\b(\d{1,2}) (AM|PM)\b', None),
            (r'\b(Morning|Afternoon|Evening|Night)\b', None),
            (r'\b(0[1-9]|1[0-2])(\/|-)(0[1-9]|[12][0-9]|3[01])(\/|-)(19|20)\d{2}\b', '%m/%d/%Y'),
            (r'\b(0[1-9]|1[0-2])(\/|-)(0[1-9]|[12][0-9]|3[01])(\/|-)(19|20)\d{2}\b', '%m-%d-%Y'),
            (r'\b(0[1-9]|[12][0-9]|3[01])(\/|-)(0[1-9]|1[0-2])(\/|-)(19|20)\d{2}\b', '%d/%m/%Y'),
            (r'\b(0[1-9]|[12][0-9]|3[01])(\/|-)(0[1-9]|1[0-2])(\/|-)(19|20)\d{2}\b', '%d-%m-%Y'),
            (r'\b(\d{4})-(\d{2})-(\d{1,2})\b', '%Y-%m-%d'),
            (r'\b(\d{2})/(\d{1,2})/(\d{4})\b', '%m/%d/%Y'),
            (r'\b(\d{4})/(\d{2})/(\d{1,2})\b', '%Y/%m/%d'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{1,2}), (\d{4}) (Morning|Afternoon|Evening|Night)\b', '%b %d, %Y'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{1,2}) (\d{4}) (Morning|Afternoon|Evening|Night)\b', '%b %d %Y'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{1,2}), (\d{4})\b', '%b %d, %Y'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{1,2}) (\d{4})\b', '%b %d %Y'),
            (r'(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{4}) (Morning|Afternoon|Evening|Night)\b', '%d %b %Y'),
            (r'(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) (\d{4})\b', '%d %b %Y'),
    ]

        # Time mappings for periods of the day
        time_mappings = {
            "Morning": 9,
            "Afternoon": 15,
            "Evening": 18,
            "Night": 21
        }

        datetime_obj = None
        for pattern, date_format in patterns:
            match = re.search(pattern, response)

            if match:
                groups = match.groups()
                if date_format:

                    date_str = ' '.join(groups[:3])

                    datetime_obj = datetime.strptime(date_str, date_format)
                    if len(groups) == 4:  # If there's a time of day
                        period = groups[3]
                        hour = time_mappings.get(period, 12)
                        datetime_obj = datetime_obj.replace(hour=hour)
                        if datetime_obj < datetime.now():
                            return "Date is in the past"
                    break
                else:
                    if len(groups) == 2 and groups[1] in ["AM", "PM"]:
                        hour, am_pm = groups
                        hour = int(hour)
                        if am_pm == 'PM' and hour != 12:
                            hour += 12
                        elif am_pm == 'AM' and hour == 12:
                            hour = 0
                        if datetime_obj:
                            datetime_obj = datetime_obj.replace(hour=hour)
                        else:
                            datetime_obj = datetime.combine(datetime.now().date(), datetime.min.time()).replace(hour=hour)

                    elif len(groups) == 1 and groups[0] in time_mappings:
                        period = groups[0]
                        hour = time_mappings[period]
                        if datetime_obj:
                            datetime_obj = datetime_obj.replace(hour=hour)
                        else:
                            datetime_obj = datetime.combine(datetime.now().date(), datetime.min.time()).replace(hour=hour)
                break

        if not datetime_obj:
            raise ValueError("No valid date format found in the response")

        return datetime_obj.isoformat()
@tool
def get_open_slots(token, prefred_date_time,location_id, reason_id, provider_id):
    
    """get the listof open slots"""
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
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

    print("Available open slots:")
    for idx, slot in enumerate(open_slots):
        print(f"{idx + 1}: {slot['ApptStartDateTime']} - {slot['ApptEndDateTime']} (ID: {slot['OpenSlotId']})")

    open_slot_id = input("Choose an open slot by entering the ID: ")
    pass
@tool
def sndotp(token, slot_id, appointment_date, reason_id, FirstName, LastName, PhoneNumber, DOB, Email):
    """to send the otp for confirmation"""
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
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

    otp = input("Enter the OTP received: ")
    pass
@tool
def book_appointment(token, open_slot_id, appointment_date,from_date, reason_id, FirstName, LastName, DOB, PhoneNumber, Email):
    """Book an appointment with the specified details"""
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
    book_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/onlinescheduling"
    # Convert ApptDate to 'MM/DD/YYYY' format
     #  print()
    #appointment_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S").strftime("%m/%d/%Y")
    parsed_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")

    # Convert the datetime object to the desired format
    appointment_date = parsed_date.strftime("%m/%d/%Y")
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

    try:
        book_appointment_response = requests.post(book_appointment_url, json=book_appointment_payload, headers=headers)
        book_appointment_response.raise_for_status()
        return book_appointment_response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while booking the appointment: {e}")
        # return
    if book_appointment_response.status_code != 200:
        return f"Failed to book appointment. Status code: {book_appointment_response.status_code}"

    return {'Status': 'Success', 'AppointmentId': 301}


# tools = [
#     Tool(name="Get Locations", func=lambda: get_locations(get_auth_token()), description="Get the list of locations for booking appointments"),
#     Tool(name="Get Providers", func=lambda location_id: get_providers(location_id, get_auth_token()), description="Get the list of providers for a specific location"),
#     Tool(name="Get Appointment Reasons", func=lambda provider_id, location_id: get_appointment_reasons(provider_id, location_id, get_auth_token()), description="Get the list of appointment reasons for a specific provider and location"),
#     Tool(name="Book Appointment", func=lambda location_id, provider_id, appointment_reason_id: book_appointment(location_id, provider_id, appointment_reason_id, get_auth_token()), description=)
# ]

tools=[get_locations,get_providers,get_appointment_reasons,get_open_slots,sndotp,book_appointment,]

# prompt = hub.pull("hwchase17/react-json")
# prompt = prompt.partial(
#     tools=render_text_description(tools),
#     tool_names=", ".join([t.name for t in tools]),
# )

# define the agent
chat_history = [{"role": "user", "content": "What are the available locations?"},
        {"role": "assistant", "content": "Here are the locations..."}
    ]
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
#     }
#     | prompt
#     | chat_model_with_stop
#     | ReActSingleInputOutputParser()
# )

agent = create_react_agent(llm, tools, prompt)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

response=agent_executor.invoke({"input": "i want to book an appointment"})
# # response=agent_executor.invoke({
# #     "input": "i want to book an appointment",
# #     "agent_scratchpad": "",  # Initial empty scratchpad
# #     "tools": tools,
# #     "tool_names": ", ".join([t.name for t in tools])
# # })

# print(response,'response----')


state = {
    "step": "start",
    "location_selected": None,
    "provider_selected": None,
    "appointment_reason_selected": None,
    "intermediate_steps": [],
    "locations": [],
    "providers": []
}


def handle_user_input(user_input):
    global state
    
    if state["step"] == "start":
        if "book appointment" in user_input.lower():
            
            chat_history = [
                                {"role": "user", "content": "What are the available locations?"},
                                {"role": "assistant", "content": "Here are the locations..."}
                            ]
            result=agent_executor.invoke({"input": f"user_input"})
            print(result,'result-------')
            try:
                action, tool_input = parse_output(result)
                if action == "Get Locations":
                    locations = tools[0].func()
                    state["locations"] = locations
                    location_names = "\n".join([f"{loc['LocationId']}: {loc['Name']}" for loc in locations])
                    state["step"] = "location_selection"
                    return "Sure, let's book an appointment. Please provide the location ID from the following list:\n" + location_names
                else:
                    return "Unexpected action. Please try again."
            except ValueError:
                return "I'm sorry, I couldn't parse the response. Please try again."

        else:
            return "I can help you book an appointment. Please type 'book appointment' to get started."

    elif state["step"] == "location_selection":
        try:
            location_id = int(user_input)
            if any(loc['LocationId'] == location_id for loc in state["locations"]):
                state["location_selected"] = location_id
                result = agent.invoke({
                    "context": "",
                    "user_input": f"Get Providers {location_id}",
                    "agent_scratchpad": "",
                    "tools": tools,
                    "tool_names": ", ".join([t.name for t in tools]),
                    "intermediate_steps": state["intermediate_steps"]
                })
                output = result['output']
                try:
                    action, tool_input = parse_output(output)
                    if action == "Get Providers":
                        providers = tools[1].func(location_id)
                        if providers:
                            provider_list = "\n".join([f"{prov['ProviderId']}: {prov['Name']}" for prov in providers])
                            state["step"] = "provider_selection"
                            return "Please choose a provider by entering the ID from the following list:\n" + provider_list
                        else:
                            return "No providers found for this location. Please try another location."
                    else:
                        return "Unexpected action. Please try again."
                except ValueError:
                    return "I'm sorry, I couldn't parse the response. Please try again."

            else:
                return "Invalid location ID. Please enter a valid ID from the list provided."
        except ValueError:
            return "Invalid input. Please enter a numerical ID from the list provided."

    elif state["step"] == "provider_selection":
        try:
            provider_id = int(user_input)
            state["provider_selected"] = provider_id
            location_id = state["location_selected"]
            result = agent.invoke({
                "context": "",
                "user_input": f"Get Appointment Reasons {provider_id} {location_id}",
                "agent_scratchpad": "",
                "tools": tools,
                "tool_names": ", ".join([t.name for t in tools]),
                "intermediate_steps": state["intermediate_steps"]
            })
            output = result['output']
            try:
                action, tool_input = parse_output(output)
                if action == "Get Appointment Reasons":
                    appointment_reasons = tools[2].func(provider_id, location_id)
                    if appointment_reasons:
                        reason_list = "\n".join([f"{reason['ReasonId']}: {reason['Reason']}" for reason in appointment_reasons])
                        state["step"] = "appointment_reason_selection"
                        return "Please choose an appointment reason by entering the ID from the following list:\n" + reason_list
                    else:
                        return "No appointment reasons found for this provider. Please try another provider."
                else:
                    return "Unexpected action. Please try again."
            except ValueError:
                return "I'm sorry, I couldn't parse the response. Please try again."

        except ValueError:
            return "Invalid provider ID. Please enter a valid ID."

    elif state["step"] == "appointment_reason_selection":
        try:
            appointment_reason_id = int(user_input)
            state["appointment_reason_selected"] = appointment_reason_id
            location_id = state["location_selected"]
            provider_id = state["provider_selected"]
            result = agent.invoke({
                "context": "",
                "user_input": f"Book Appointment {location_id} {provider_id} {appointment_reason_id}",
                "agent_scratchpad": "",
                "tools": tools,
                "tool_names": ", ".join([t.name for t in tools]),
                "intermediate_steps": state["intermediate_steps"]
            })
            output = result['output']
            try:

                # action, tool_input = parse_output(output)
                if action == "Book Appointment":
                    booking_result = tools[3].func(location_id, provider_id, appointment_reason_id)
                    state["step"] = "start"
                    state["location_selected"] = None
                    state["provider_selected"] = None
                    state["appointment_reason_selected"] = None
                    if booking_result.get("Status") == "Success":
                        return f"Your appointment has been booked successfully. Appointment ID: {booking_result['AppointmentId']}"
                    else:
                        return "Failed to book the appointment. Please try again."
                else:
                    return "Unexpected action. Please try again."
            except ValueError:
                return "I'm sorry, I couldn't parse the response. Please try again."

        except ValueError:
            return "Invalid appointment reason ID. Please enter a valid ID."

    return "Unexpected state. Please start over."






def home(request):
    return render(request, "home.html")

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        user_input = data.get('input', '')
        # response=handle_user_input(user_input)
        response = agent_executor.invoke({"input": 'I want to book an appointment', "chat_history": chat_history}) # Manual Executor for Testing
        print(response,'-----response')
        # response=''
        try:
            # Parse the output using parse_output
            action, tool_input = parse_output(response)
        except:
            pass
        if response:
            bot_response = response
           
        else:
            bot_response = "I'm sorry, I couldn't process your request. Can you please try again?"
        return JsonResponse({"response": bot_response})
    return JsonResponse({"response": "Invalid request"}, status=400)




def parse_output(output):
    try:
        lines = str(output).strip().split("\n")
        action_line = next(line for line in lines if line.startswith("Action:"))
        tool_input_line = next(line for line in lines if line.startswith("Tool Input:"))
        action = action_line.split("Action:")[1].strip()
        tool_input = tool_input_line.split("Tool Input:")[1].strip()
        return action, tool_input
    except (StopIteration, IndexError) as e:
        raise ValueError("Could not parse output correctly.") from e
