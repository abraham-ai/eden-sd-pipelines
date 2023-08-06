import requests
import time

# args = https://github.com/abraham-ai/eden-api/blob/main/mongo-init.js

EDEN_API_URL = "https://api.eden.art"

header = {
    "x-api-key": "d20d21f66625e7209161967a410971498658be1f2dc04432",
    "x-api-secret": "a63f9957d90a1ad6b08e66b98907e45d09bd92376d4a448d"
}


def run_task(generatorName, config):
    request = {
        "generatorName": generatorName,
        "config": config
    }

    response = requests.post(
        f'{EDEN_API_URL}/tasks/create', 
        json=request, 
        headers=header
    )

    if response.status_code != 200:
        print(response.text)
        return None
    
    result = response.json()
    taskId = result['taskId']

    while True:
        response = requests.get(
            f'{EDEN_API_URL}/tasks/:taskId', 
            json={"taskIds": [taskId]},
            headers=header
        )

        if response.status_code != 200:
            print(response.text)
            return None

        result = response.json()

        print(result)

        try:
            task = result['task'][0]
            status = task['status']

            if status == 'completed':
                return task
            elif status == 'failed':
                print("FAILED!")
                return None
        except:
            pass

        time.sleep(1)



config = {
    "text_input": "masterful artwork or the passing of time, infinity, glass, and the universe",
}

result = run_task("create", config)

output_url = result['output'][-1]
print(output_url)
  