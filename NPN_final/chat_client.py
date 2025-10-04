import requests

url = "http://127.0.0.1:5000/chat"
user_id = "test1"

print("Chat with LoanBot! (type 'quit' to exit)\n")

while True:
    msg = input("You: ")
    if msg.lower() in ["quit", "exit"]:
        break
    res = requests.post(url, json={"user_id": user_id, "message": msg})
    print("Bot:", res.json()["reply"])
