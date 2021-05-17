import json
import requests

import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv("data.csv").drop("target", axis=1)
    data["id"] = range(len(data))
    request_data = data.to_dict(orient="records")
    print("Request data:")
    print(request_data)
    response = requests.post(
        "http://0.0.0.0:8000/predict",
        json.dumps(request_data)
    )
    print(f"Status code: {response.status_code}")
    print(f"Response data:")
    print(response.json())
