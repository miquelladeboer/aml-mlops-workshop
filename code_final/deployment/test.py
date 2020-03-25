input_data = {
        "text": "it's late on the evening",
        "target": 1
    }

try:
    r = aci_service.run(input_data)
    result = r
    print(result)
except KeyError as e:
    print(str(e))