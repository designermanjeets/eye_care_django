from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from datetime import datetime
import re
import requests
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor,  create_react_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
# from langchain.core.messages import HumanMessage, AIMessage
# from . model_loader import *
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

# Define your custom prompt template

prompt = PromptTemplate(
    template=("""
     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
      
      You are the health care assistent Agent read the user queries and return the response using the tools
      you do not know any thing  you will use tools for everything. \
      
      Answer the following questions as best you can. You have access to the following tools:
      return the answers as soon as you get the answer to the question
      
         
      Tools: {tools}
      Tools_names: {tool_names}
      Tools_description : {tool_description}
      Tools_args : {tool_args}
  
      Use the following format:
  
      Question: the input question you must answer.
  
      Thought: you should always think about what to do.
  
      Action: the action to take, should be one of [{tool_names}]
  
      Action Input: the input to the action.
  
      Observation: Observe the result of the action.
      output : Observation.
  
      ... (this Thought/Action/Action Input/Observation can repeat only 1 time.)
  
      Thought: I now know the final answer

      Final Answer: the final answer to the original input question based on observation
      terminate the chain
      Begin!

      Question: {input}

      Thought:{agent_scratchpad}
    
      stop the chain after the obsevation
      <|start_header_id|>user<|end_header_id|>
         question: {input}.
       <|eot_id|><|start_header_id|>assistant<|end_header_id|>
       
      <|eot_id|>
      <|end_of_text|>
          """
      )
     
    ,input_variables=["input","tools","tool_names","tool_description",'tool_args','agent_scratchpad'],
  )




llm = ChatGroq(
            model="llama3-70b-8192",
            groq_api_key='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'
        )

api_url = "https://dgltkszlxd0qaoge.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Authorization": "Bearer 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",  
    "Content-Type": "application/json"
}

def call_huggingface_endpoint(prompt, api_url,  max_new_tokens,  do_sample, temperature, top_p ,max_length=512,retries=1, backoff_factor=0.3):
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

@tool
def get_greeting_response(user_input):
    """Respond to the user's greeting."""
    modelPromptForAppointment = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are a helpful Eyecare assistant for MaximCaye Care. Start with a simple greeting and assist the user related to appointment and Do not anything from your end.<|eot_id|><|start_header_id|>user<|end_header_id|>

            {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    response=call_huggingface_endpoint(modelPromptForAppointment,api_url,256 ,False,0.9 ,0.9)
    # response = query_llama3(modelPromptForAppointment)
    response = response[len(modelPromptForAppointment):].strip()
    print("TDgx",response)
    return response
get_greeting_response.return_direct=True



def get_auth_token() -> str:
    """
    Get authentication token using vendor and account credentials.
    """
    print("Get authentication token")
    auth_url = "https://iochatbot.maximeyes.com/api/v2/account/authenticate"
    auth_payload = { "VendorId": "e59ec838-2fc5-4639-b761-78e3ec55176c", "VendorPassword": "password@123", "AccountId": "chatbot1", "AccountPassword": "sJ0Y0oniZb6eoBMETuxUNy0aHf6tD6z3wynipZEAxcg=" }
    headers = {'Content-Type': 'application/json'}
    try:
        auth_response = requests.post(auth_url, json=auth_payload, headers=headers)
        auth_response.raise_for_status()
        response_json = auth_response.json()

        if response_json.get('IsToken'):
            print(response_json)
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
    print('token++++++++++++++++++++++++++++++=================================')
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
    # return 'list of available location = ["Hillsboro", "Beaverton"]'
    # return [{'Name': 'Hillsboro', 'LocationId': 1}, {'Name': 'Beaverton', 'LocationId': 3}]
    return locations
get_locations.return_direct=True
@tool
def get_providers(location_id):
    """Get the list of providers for a specific location"""
    token=get_auth_token()
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
    print('location_id-',location_id)
    match = re.search(r'\{.*\}', location_id)
    
    json_string = match.group(0)
    location_data = json.loads(json_string)
    location_id = location_data.get("location_id")
    print("erfssssssss",location_id)
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
get_providers.return_direct=True

@tool
def get_appointment_reasons(location_id_provider_id):
    """Get the list of appointment reasons for a specific provider and location"""
    print("efjawkjk")

    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {get_auth_token()}'}

    data = json.loads(location_id_provider_id)
    print("ssdfff",data)

    provider_id = data.get("provider_id")
    location_id = data.get("location_id")

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

    # reason_id = input("Choose a reason by entering the ID: ")
    # if reason_id:
    #   print('Thanks for providing reason')
    # return [{'Reason': 'Consultation', 'ReasonId': 201}, {'Reason': 'Follow-up', 'ReasonId': 202}]
    return reasons
get_appointment_reasons.return_direct=True




@tool
def get_open_slots(id):
    """Get the list of open slots based on the given reason ID and other parameters."""
    token=get_auth_token()
    print("rfesf",id)
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
    # preferred = prefred_date_time_fun(prefred_date_time)
    # print("prefred date time",preferred)
    data = json.loads(id)  # Assuming user_input is a JSON string
    preferred_date_time = data.get("preferred_date_time")
    location_id = data.get("location_id")
    reason_id = data.get("reason_id")
    provider_id = data.get("provider_id")
    # from_date = "2024-07-20T15:30:00"
    # from_date = preferred
    # print("from_date",from_date)

    get_open_slots_url = f"https://iochatbot.maximeyes.com/api/appointment/openslotforchatbot?fromDate={preferred_date_time}&isOpenSlotsOnly=true"
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

    return open_slots
get_open_slots.return_direct=True


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
def book_appointment(data):
    """Book an appointment with the specified details"""
    token=get_auth_token()
    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {token}'}
    data = json.loads(data)
    print("ddfnzn",data)
    open_slot_id = data.get("open_slot_id")
    from_date = data.get("preferred_date_time")
    reason_id = data.get("appointment_reason_id")
    FirstName = data.get("first_name")
    LastName = data.get("last_name")
    DOB = data.get("DOB")
    PhoneNumber = data.get("PhoneNumber")
    Email = data.get("Email")
    book_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/onlinescheduling"
    # Convert ApptDate to 'MM/DD/YYYY' format
     #  print()
    # appointment_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S").strftime("%m/%d/%Y")
    # parsed_date = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")

    # # Convert the datetime object to the desired format
    # appointment_date = parsed_date.strftime("%m/%d/%Y")
    # print(appointment_date)
    book_appointment_payload = {
        "OpenSlotId": str(open_slot_id),
        "ApptDate": str(from_date),
        "ReasonId": str(reason_id),
        "FirstName": str(FirstName),
        "LastName": str(LastName),
        "PatientDob": str(DOB),
        "MobileNumber": str(PhoneNumber),
        "EmailId": str(Email)}
    print("dnakjik",book_appointment_payload)

    try:
        book_appointment_response = requests.post(book_appointment_url, json=book_appointment_payload, headers=headers)
        book_appointment_response.raise_for_status()
        print("dwwsfqf",book_appointment_response.json())
        return book_appointment_response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while booking the appointment: {e}")
        # return
    if book_appointment_response.status_code != 200:
        return f"Failed to book appointment. Status code: {book_appointment_response.status_code}"

    return book_appointment_response.json()

book_appointment.return_direct=True

tools=[get_locations,get_providers,get_appointment_reasons,get_open_slots,sndotp,book_appointment,get_greeting_response]
memory = ConversationBufferMemory(memory_key="chat_history")

# define the agent
agent = create_react_agent(llm, tools, prompt,stop_sequence=["Final Answer","Observation"])
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,verbose=True, handle_parsing_errors=True,max_iterations=6)

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
            print('book appointment')
            chat_history = [
                                {"role": "user", "content": "What are the available locations?"},
                                {"role": "assistant", "content": "Here are the locations..."}
                            ]
            tools=[get_locations,get_providers,get_appointment_reasons,get_open_slots,sndotp,book_appointment]
            Tools_names=", ".join([t.name for t in tools])
            tool_description=", ".join([t.description for t in tools])
            tool_args=", ".join([str(t.args) for t in tools])
            result=agent_executor.invoke({"input": user_input,'tools':tools,"tool_names":Tools_names,"tool_description":tool_description,'tool_args':tool_args,'agent_scratchpad':""})
            print(result,'result-------')
        
            locations = result['output']

            # Printing the extracted locations
            # for location in locations:
            #     print(f"Location Name: {location['Name']}, Location ID: {location['LocationId']}")

            # # Convert to JSON
            # json_output = json.dumps({"output": locations}, indent=4)
            # # Transition to the next step
            state["step"] = "location_selection"
            state["locations"] = locations
            print(locations)
            print(state["step"])
            return locations
        else :
            print('get_greeting_response')
            # Generate a response for the greeting using the LLM
            result = agent_executor.invoke({
                "input": user_input,
                "tools": [get_greeting_response],
                "tool_names": "get_greeting_response",
                "tool_description": get_greeting_response.description,
                "tool_args": json.dumps({"greeting": user_input}),
                "agent_scratchpad": ""
            })
            greeting_response = result.get('output', "I'm not sure how to respond to that greeting.")
            return greeting_response
    elif state["step"] == "location_selection":
        try:
            print("location_selection")
            location_data = json.loads(user_input)  
            location_id = int(location_data)
            if any(loc['LocationId'] == location_id for loc in state["locations"]):
                state["location_selected"] = location_id
                result = agent_executor.invoke({
                    "input": f"Get Providers for location {location_id}",
                    "tools": [get_providers],
                    "tool_names": "get_providers",
                    "tool_description": get_providers.description,
                    "tool_args": json.dumps({"location_id": location_id}),
                    "agent_scratchpad": ""
                })
                providers = result['output']
                print("fhdb",providers)

                # Printing the extracted locations
                # for provider in providers:
                #     print(f"provider Name: {provider['Name']}, provider ID: {provider['ProviderId']}")
                    
                json_output = json.dumps({"output": providers}, indent=4)
                print("RSG",json_output)
                state["step"] = "provider_selection"  # Move to the next step
        
                state["providers"] = providers
                print(json_output,"fewjwij")
                return providers
            else:
                return "Invalid providers ID. Please enter a prociders ID from the list provided."
        
        except ValueError:
            return "Invalid input. 1111Please enter a numerical ID from the list provided."


    elif state["step"] == "provider_selection":
        try:
            provider_id = int(user_input)
            print("dfgxg",provider_id)

            if any(provider['ProviderId'] == provider_id for provider in state["providers"]):
                state["provider_selected"] = provider_id
                location_id = state["location_selected"]
                print("dtrgdt",location_id)
                result = agent_executor.invoke({
                    "input": f"Get Appointment Reasons when selected location:{location_id} and  selected provider is :{provider_id} ",
                    "tools": [get_appointment_reasons],
                    "tool_names": "get_appointment_reasons ",
                    "tool_description": get_appointment_reasons.description,
                    "tool_args": json.dumps({"provider_id": provider_id, "location_id": location_id}),
                    #  "tool_args":get_appointment_reasons.args,
                    "agent_scratchpad": ""
                })
                appointment_reasons = result['output']
                print("fhdb", appointment_reasons)

                # Print the extracted appointment reasons
                # reason_list = "\n".join(f"Reason ID: {reason['ReasonId']}, Reason: {reason['Reason']}" for reason in appointment_reasons)
                state["step"] = "appointment_reason_selection"
                state["appointment_reasons"] = appointment_reasons
                return appointment_reasons

            else:
                return "Invalid provider ID. Please enter a valid ID from the list provided."

        except ValueError:
            return "Invalid input. Please enter a numerical ID from the list provided."

    elif state["step"] == "appointment_reason_selection":
        try:
            appointment_reason_id = int(user_input)
            state["appointment_reason_selected"] = appointment_reason_id
            location_id = state["location_selected"]
            provider_id = state["provider_selected"]
            preferred_date_time = '2024-08-13T12:00:00'
            result = agent_executor.invoke({
                "input": f"Get open slots for location {location_id}, provider {provider_id}, reason {appointment_reason_id}",
                "tools": [get_open_slots],
                "tool_names": "get_open_slots",
                "tool_description": get_open_slots.description,
                "tool_args": json.dumps({"preferred_date_time": preferred_date_time, "location_id": location_id, "reason_id": appointment_reason_id, "provider_id": provider_id}),
                "agent_scratchpad": ""
            })
            open_slots = result['output']
            if open_slots:
                state["step"] = "slot_selection"
                state["open_slots"] = open_slots
                return open_slots
            else:
                return "No open slots available. Please try again later."
    
        except ValueError:
            return "Invalid appointment reason ID. Please enter a valid ID."

    elif state["step"] == "slot_selection":
        # try:
            open_slot_id = int(user_input)
            if any(slot['OpenSlotId'] == open_slot_id for slot in state["open_slots"]):
                state["open_slot_selected"] = open_slot_id
                location_id = state["location_selected"]
                provider_id = state["provider_selected"]
                preferred_date_time = '05/10/2024'
                first_name='preeti'
                last_name='sidana'
                DOB='12/09/1998'
                PhoneNumber='9874563210'
                Email='dmk@gmail.com'


                appointment_reason_id = state["appointment_reason_selected"]
                
                # Use the tool to book the appointment
                booking_result = agent_executor.invoke({
                    "input": f"Book Appointment with location {location_id}, provider {provider_id}, reason {appointment_reason_id}, and slot {open_slot_id} ,first_name {first_name},last_name {last_name},DOB {DOB},preferred_date_time {preferred_date_time},PhoneNumber {PhoneNumber},Email{Email}",
                    "tools": [book_appointment],
                    "tool_names": "book_appointment",
                    "tool_description": book_appointment.description,
                    "tool_args": json.dumps({"location_id": location_id, "provider_id": provider_id, "appointment_reason_id": appointment_reason_id, "open_slot_id": open_slot_id}),
                    "agent_scratchpad": ""
                })
                
                booking_response = booking_result['output']
                # state["step"] = "start"
                # state["location_selected"] = None
                # state["provider_selected"] = None
                # state["appointment_reason_selected"] = None
                # state["open_slot_selected"] = None
                
                # if booking_response.get("Status") == "Success":
                    # return f"Your appointment has been booked successfully. Appointment ID: {booking_response['AppointmentId']}"
        #         else:
        #             return "Failed to book the appointment. Please try again."
        #     else:
        #         return "Invalid slot ID. Please enter a valid ID from the list provided."
        # except ValueError:
        #     return "Invalid input. Please enter a numerical ID from the list provided."
            return booking_response
    
    else :
        print('get_greeting_response')
        # Generate a response for the greeting using the LLM
        result = agent_executor.invoke({
            "input": user_input,
            "tools": [get_greeting_response],
            "tool_names": "get_greeting_response",
            "tool_description": get_greeting_response.description,
            "tool_args": json.dumps({"greeting": user_input}),
            "agent_scratchpad": ""
        })
        greeting_response = result.get('output', "I'm not sure how to respond to that greeting.")
        return greeting_response
        # return "Unexpected state. Please start over."


def home(request):
    return render(request, "home.html")

@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        user_input = data.get('input', '')
        print(user_input,'user_input')
        response=handle_user_input(user_input)

        # response = agent_executor.invoke({"input": 'I want to book an appointment', "chat_history": chat_history}) # Manual Executor for Testing
        print(response,'-----response')
        # response=''
    #     try:
    #         # Parse the output using parse_output
    #         action, tool_input = parse_output(response)
    #     except:
    #         pass
    #     if response:
    #         bot_response = response
           
    #     else:
    #         bot_response = "I'm sorry, I couldn't process your request. Can you please try again?"
    #     return JsonResponse({"response": bot_response})
    # return JsonResponse({"response": "Invalid request"}, status=400)
        response=str(response)
        return JsonResponse({"response": response})
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
