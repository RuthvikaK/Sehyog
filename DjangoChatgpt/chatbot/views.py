from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
from django.utils import timezone
from together import Together
import paramiko
import time
import shlex
import re

def clean_output(output):
    lines = output.splitlines()
    cleaned_lines = []

    for line in lines:
        if line.strip().lower() == "cat output.txt":
            continue
        line = re.sub(r'^\[INST\]\s*', '', line)
        line = re.sub(r'^\[PROMPT\]\s*', '', line)
        line = re.sub(r'^\[SOL\]\s*', '', line)
        line = re.sub(r'^\s*Answer:\s*', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^\[ANS\]\s*', '', line)
        line = re.sub(r'# Context', '', line)
        line = re.sub(r'\(hackathon\) \[.*\]\$\s*$', '', line)
        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines).strip()

def query(message, model):
    hostname = "longleaf.unc.edu"
    port = 22
    username = "rkosuri"
    password = "Shreeram2!" 
    job_check_command = "squeue --me | grep interact"
    start_job_command = "salloc -t 1:00:00 -p volta-gpu --mem=10g -N 1 -n 1 --qos gpu_access --gpus=1  --job-name=interactive_gpu"

    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, port=port, username=username, password=password)
        print(f"Connected to {hostname}")
        stdin, stdout, stderr = ssh_client.exec_command(job_check_command)
        job_status = stdout.read().decode().strip()

        if not job_status:
            print("No interactive GPU session found. Starting a new one...")
            ssh_client.exec_command(start_job_command)

        node_name = None
        while not node_name:
            stdin, stdout, stderr = ssh_client.exec_command("squeue --me --format='%N' | tail -n 1")
            node_name = stdout.read().decode().strip()
            if not node_name:
                print("Waiting for node assignment...")
                time.sleep(5)
                
        print(f"Allocated Node: {node_name}")
        ssh_shell = ssh_client.invoke_shell()
        ssh_shell.send(f"ssh {node_name}\n")
        wait_for_prompt(ssh_shell)
        print(f"Connected to GPU node: {node_name}")
        
        safe_message = shlex.quote(message)
        print("🔎 Prompt from user:", message)
        print("📦 Safe message sent to GPU node:", safe_message)
        execution_commands = [
            "cd /work/users/r/k/rkosuri/last",
            "module load python/3.9.6",
            "source ~/.bashrc",
            "source qwenenv/bin/activate"
        ]
        if model == "llama":
            execution_commands.append(f"python llama.py {safe_message}")
        elif model == "qwen":
            execution_commands.append(f"python qwen.py {safe_message}")
        else:
            execution_commands.append(f"python planner.py {safe_message}")
        
        execution_commands.extend(["cat output.txt"])
        python_output = ""
        for cmd in execution_commands:
            print(f"running cmd: {cmd}")
            ssh_shell.send(cmd + "\n")
            #
            cmd_output = wait_for_prompt(ssh_shell)
            print(f"cmd output: {cmd_output}")
            if cmd.strip() == "cat output.txt":
                cleaned_response = clean_output(cmd_output)
                print(f"cleaned response: {cleaned_response}")
                python_output = cleaned_response
                print(f"python output: {python_output}")
            #

            # cmd_output = cmd.strip()
            # cleaned_response = clean_output(cmd_output)
            # python_output = cmd_output
        
        # output = read_output(ssh_shell)
        # output = " The capital of France is Paris, which has been its capital since the Middle Ages. There were many significant events that took place in France during the 1960s. One major event was the student riots in May 1968. These riots began as a protest against the education system but quickly turned into a nationwide movement against capitalism and consumerism. Another important event was the Algerian War of Independence, which lasted from 1954-1962. This war resulted in the independence of Algeria from France. Additionally, the 1960"
        # print("Execution Output:\n", output)
        print("Execution Output:\n", python_output)

        ssh_shell.close()
        ssh_client.close()
        print("Connection closed.")

        return python_output if python_output else None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def wait_for_prompt(ssh_shell, timeout=600):
    buffer = ""
    start_time = time.time()
    while True:
        if ssh_shell.recv_ready():
            chunk = ssh_shell.recv(1024).decode(errors='ignore')
            buffer += chunk

        if any(buffer.endswith(prompt) for prompt in ('$ ', '# ', ':~$ ', '> ')):
            break

        if time.time() - start_time > timeout:
            print("Timeout waiting for command to finish.")
            break
    return buffer

def read_output(ssh_shell):
    buffer = ""
    while ssh_shell.recv_ready():
        chunk = ssh_shell.recv(1024).decode()
        buffer += chunk
        print(chunk, end="")
    return buffer

def ask_model(message, model):
    if model == 'Meta Llama':
        return query(message, "llama")
    elif model == 'Qwen':
        return query(message, "qwen")
    else: 
        return query(message, "OurModel")


def ask_openai(message, model):
    client = Together(api_key="630e0bb3b0990263b9ab779ca9e80376388b627d571c1c922e72b366dbf918a8")

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True,
    )
    response_text = ''
    for chunk in stream:
        response_text += chunk.choices[0].delta.content

    return response_text

def chatbot(request):
    chats = Chat.objects.filter(user=request.user.id)

    if request.method == 'POST':
        message = request.POST.get('message')
        model = request.POST.get('model')
        if not model:
            return JsonResponse({'error': 'Please select a model'}, status=400)

        response = ask_model(message, model)

        if response is None: #
            return JsonResponse({'error': 'Model failed to respond. Please try again.'}, status=500)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    
    return render(request, 'chatbot.html', {'chats': chats})

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Passwords do not match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')