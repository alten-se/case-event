import requests, os

from lungai.paths import DATA_PATH

url = "http://127.0.0.1:5000/api"
file_name ="101_1b1_Al_sc_Meditron.wav"
file_path = os.path.join(DATA_PATH, file_name)

def test_api():
    with open(file_path, mode="rb") as file:
       files = {"file": file}
       response = requests.post(url, files=files)
       print(response)
       print(response.json())

if __name__ == "__main__":
    test_api()