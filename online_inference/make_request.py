import json
import requests

import pandas as pd

DATA_PATH = "data/inference_test/data.csv"


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH).drop("target", axis=1)
    data["id"] = range(len(data))
    request_data = data.to_dict(orient="records")
    print("First sample of request data:")
    print(request_data[0])
    response = requests.post(
        "http://0.0.0.0:8000/predict",
        json.dumps(request_data)
    )
    print(f"Response status code: {response.status_code}")
    print("First sample of response data:")
    print(response.json()[0])
