import requests


url = "http://127.0.0.1:5000/api"
file_path = "/home/gws/projects/case-event/lung_ai/db/data/101_1b1_Al_sc_Meditron.wav"

def test_api():
    with open(file_path, mode="rb") as file:
       files = {"file": file}
       response = requests.post(url, files=files)
       print(response)
       print(response.json())

if __name__ == "__main__":
    test_api()