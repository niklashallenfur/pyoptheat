import json
from datetime import datetime

from heat_optimizer_port import OptimizationService, OptimizationParameters

with open('output/nodeplan.json', 'r') as file:
    json_text = file.read()

service = OptimizationService()

data = json.loads(json_text)
params = OptimizationParameters.model_validate(data['params'])


def add_timezone_to_showers(params: OptimizationParameters):
    if params.showers:
        for shower in params.showers:
            if shower.start.tzinfo is None:
                shower.start = shower.start.replace(tzinfo=params.time.tzinfo)
            if shower.end.tzinfo is None:
                shower.end = shower.end.replace(tzinfo=params.time.tzinfo)


add_timezone_to_showers(params)

plan = service.optimize(params)


class PrettyFloat(float):
    def __repr__(self):
        return 0 if self == 0 else '%.15g' % self


def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        val = round(obj, 15)
        return int(round(val, 0)) if round(val, 0) == val else val
    elif isinstance(obj, dict):
        return dict((k, custom_serializer(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(custom_serializer, obj))
    return obj


# Convert the plan to a JSON string
plan_json = json.dumps(custom_serializer(plan), indent=4)

# Print the JSON string
with open('output/pyplan.json', 'w') as file:
    file.write(plan_json)