from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import pulp
import json
import simplejson
from dataclasses import asdict, is_dataclass

from dataclass_wizard import fromdict, asdict


@dataclass
class IndoorSpec:
    target_temp: float
    current_temp: Optional[float] = None
    passive_heating_degrees: float = 0.0

@dataclass
class Forecast:
    time: datetime
    temp: float

@dataclass
class RadiatorSpec:
    power_per_temp_delta: float  # W per degree Celsius
    c_flow: float

@dataclass
class AccSpec:
    liters: float
    current_temp: float
    min_temp: float
    max_temp: float
    temp_loss_per_hour_degrees: float
    radiator_flow_temp_above_acc_avg_temp: float = 0.0

@dataclass
class HotWaterSpec:
    min_temp: float
    average_power: float  # W

@dataclass
class PumpPowerSpec:
    consumed_power_watt: float
    heating_power_watt: float
    below_degrees: Optional[float] = None  # if None, then no limit

@dataclass
class PumpSpec:
    power: List[PumpPowerSpec]
    max_temp: float
    start_delay_minutes: int = 0

@dataclass
class Shower:
    start: datetime
    end: datetime
    temp: float
    energy: float  # W

@dataclass
class PriceSpec:
    today: List[float]  # Prices per hour
    tomorrow: Optional[List[float]] = None

@dataclass
class OptimizationParameters:
    time: datetime
    split_hours_into: int
    plan_hours: int
    indoor: IndoorSpec
    outdoor: List[Forecast]
    radiator: RadiatorSpec
    acc_spec: AccSpec
    hot_water: HotWaterSpec
    pump_spec: PumpSpec
    prices: PriceSpec
    showers: Optional[List[Shower]] = None

class PriceExtractor:
    def __init__(self, price_spec: PriceSpec, start_at: datetime):
        self.start_at = start_at
        self.prices = price_spec.today + (price_spec.tomorrow or []) + price_spec.today

    def get_price_at(self, time: datetime) -> float:
        hour_since_start = int((time - self.start_at).total_seconds() / 3600)
        return self.prices[hour_since_start % len(self.prices)]

class OutdoorTempExtractor:
    def __init__(self, forecasts: List[Forecast]):
        self.forecasts = forecasts

    def get_temp_at(self, time: datetime) -> float:
        temp = self.forecasts[0].temp if self.forecasts else 0.0
        for f in self.forecasts:
            if f.time <= time:
                temp = f.temp
            else:
                break
        return temp

class PumpEffectExtractor:
    def __init__(self, pump_spec: PumpSpec, acc_volume: float):
        self.pump_spec = pump_spec
        self.acc_volume = acc_volume
        self.heating_per_hour_intervals = [
            {
                'deg': self.get_heating_per_hour(i),
                'max_temp': ps.below_degrees if ps.below_degrees is not None else 10000
            }
            for i, ps in enumerate(self.pump_spec.power)
        ]

    def get_heating_per_hour(self, i: int) -> float:
        return self.pump_spec.power[i].heating_power_watt / (self.acc_volume * 1.16)

class ShowerExtractor:
    def __init__(self, showers: Optional[List[Shower]] = None):
        self.showers = showers or []

    def get_shower(self, period):
        for shower in self.showers:
            if not (shower.start >= period['end'] or shower.end <= period['start']):
                return {'temp': shower.temp, 'power': shower.energy}
        return {'temp': 0.0, 'power': 0.0}

class OptimizationService:
    def optimize(self, params: OptimizationParameters):
        time = params.time
        split_hours_into = params.split_hours_into
        plan_hours = params.plan_hours
        indoor = params.indoor
        outdoor = params.outdoor
        radiator = params.radiator
        acc_spec = params.acc_spec
        hot_water = params.hot_water
        pump_spec = params.pump_spec
        showers = params.showers

        number_of_periods = plan_hours * split_hours_into

        start_of_current_hour = datetime(time.year, time.month, time.day, time.hour, tzinfo=time.tzinfo)
        start_of_current_day = datetime(time.year, time.month, time.day, tzinfo=time.tzinfo)

        current_part_of_hour = (time - start_of_current_hour).total_seconds() / 3600.0
        current_split_hour = int(current_part_of_hour * split_hours_into)
        current_split_hour_start = start_of_current_hour + timedelta(
            minutes=current_split_hour * (60 / split_hours_into)
        )

        showerX = ShowerExtractor(showers)
        prices = PriceExtractor(params.prices, start_of_current_day)
        outdoorTemps = OutdoorTempExtractor(outdoor)
        pumpEffect = PumpEffectExtractor(pump_spec, acc_spec.liters)

        acc_heat_capacity = acc_spec.liters * 1.16

        periods = []
        for i in range(number_of_periods):
            if i == 0:
                start = time
            else:
                start = current_split_hour_start + timedelta(
                    minutes=i * (60 / split_hours_into)
                )
            end = current_split_hour_start + timedelta(
                minutes=(i + 1) * (60 / split_hours_into)
            )
            duration = (end - start).total_seconds() / 3600.0

            period = {
                'start': start,
                'end': end,
                'duration': duration
            }
            periods.append(period)

        for period in periods:
            period['outdoor_temp'] = outdoorTemps.get_temp_at(period['start'])

        for period in periods:
            current_temp = indoor.current_temp if indoor.current_temp is not None else indoor.target_temp
            rad_flow_temp = max(
                indoor.target_temp +
                indoor.target_temp +
                (indoor.target_temp - current_temp) +
                (indoor.target_temp - period['outdoor_temp'] - indoor.passive_heating_degrees) * radiator.c_flow,
                indoor.target_temp
            )
            period['rad_flow_temp'] = rad_flow_temp

        for period in periods:
            shower_info = showerX.get_shower(period)
            period['acc_min_temp'] = max(
                period['rad_flow_temp'] - (acc_spec.radiator_flow_temp_above_acc_avg_temp or 0.0),
                hot_water.min_temp,
                acc_spec.min_temp,
                shower_info['temp']
            )

        for period in periods:
            period['price'] = prices.get_price_at(period['start'])

        for period in periods:
            consumption = {}
            radiator_temp_change = (
                (period['rad_flow_temp'] - indoor.target_temp) * radiator.power_per_temp_delta * period['duration']
            ) / acc_heat_capacity
            hot_water_temp_loss = (
                (hot_water.average_power / acc_heat_capacity) * period['duration']
            )
            shower_info = showerX.get_shower(period)
            shower_temp_loss = (
                (period['duration'] * shower_info['power']) / acc_heat_capacity
            )
            consumption['radiator'] = radiator_temp_change
            consumption['hot_water'] = hot_water_temp_loss
            consumption['shower'] = shower_temp_loss
            consumption['total'] = radiator_temp_change + hot_water_temp_loss + shower_temp_loss
            period['consumption'] = consumption

        # Create the optimization model
        model = pulp.LpProblem("Optimize", pulp.LpMinimize)

        # Variables
        acc_temp = []
        for i, period in enumerate(periods):
            if i == 0:
                lb = acc_spec.current_temp
                ub = acc_spec.current_temp
            else:
                lb = period['acc_min_temp']
                ub = min(acc_spec.max_temp, pump_spec.max_temp)
            var = pulp.LpVariable(
                f"acc_temp_{i}",
                lowBound=lb,
                upBound=ub,
                cat='Continuous'
            )
            acc_temp.append(var)

        acc_surrounding_diff = []
        for i, period in enumerate(periods):
            var = pulp.LpVariable(
                f"acc_surrounding_diff_{i}",
                lowBound=None,
                upBound=None,
                cat='Continuous'
            )
            acc_surrounding_diff.append(var)


        # constraint temeraturdiff mot omgivningen
        for i, period in enumerate(periods):
            lhs = acc_temp[i] - acc_surrounding_diff[i]
            rhs = indoor.current_temp if indoor.current_temp is not None else indoor.target_temp
            model += (lhs == rhs), f"calc_acc_surrounding_diff_{i}"

        objective = 0
        # pump 0-1 power in respective temperature interval
        period_pump = []
        for i, period in enumerate(periods):
            period_pump_vars = []
            for powerSpec in pump_spec.power:
                var = pulp.LpVariable(
                    f"pump_on_{i}_below_{powerSpec.below_degrees or 'max'}",
                    lowBound=0,
                    upBound=1,
                    cat='Continuous'
                )
                # Objective function
                costPerFullPower = period['price'] * powerSpec.consumed_power_watt * period['duration'] / 1000
                print('costPerFullPower', costPerFullPower)
                objective += var * costPerFullPower

                period_pump_vars.append(var)
            period_pump.append(period_pump_vars)

        # Constraint Ackumulatortankens temperaturändring
        for i in range(len(periods) - 1):
            lhs = acc_temp[i] - acc_temp[i+1] - acc_spec.temp_loss_per_hour_degrees * periods[i]['duration'] * acc_surrounding_diff[i]
            heating_terms = 0
            for j, hph in enumerate(pumpEffect.heating_per_hour_intervals):
                heating_terms += period_pump[i][j] * hph['deg'] * periods[i]['duration']
            lhs += heating_terms
            rhs = periods[i]['consumption']['total']
            model += (lhs == rhs), f"acc_temp_change_{i}"

        # Constraint sum of pump power <= 1
        for i, temp_intervals in enumerate(period_pump):
            model += (pulp.lpSum(temp_intervals) <= 1), f"pump_power_sum_{i}"

        # constraints pump power in respective temperature interval
        acc_below_limit = []
        for i, period in enumerate(periods):
            period_acc_below_limit = []
            for hph_i, main_hph in enumerate(pumpEffect.heating_per_hour_intervals[:-1]):
                below = pulp.LpVariable(
                    f"acc_below_{i}_{main_hph['max_temp']}",
                    cat='Binary'
                )
                objective += -0.00001 * below # "force" to 1
                period_acc_below_limit.append(below)

                # either the effect of period_pump[period_i,hph_i] is below the temp limit
                M1 = 30
                heating_terms = 0
                for j in range(hph_i + 1):
                    hph = pumpEffect.heating_per_hour_intervals[j]
                    heating_terms += period_pump[i][j] * hph['deg'] * period['duration']
                temp_after_heating = acc_temp[i] + heating_terms - acc_spec.temp_loss_per_hour_degrees * period['duration'] * acc_surrounding_diff[i]
                model += (
                    temp_after_heating <= main_hph['max_temp'] + period['consumption']['total'] + M1 * (1 - below)
                ), f"pump_temp_limit_{i}_{hph_i}"
                model += (period_pump[i][hph_i] <= below), f"pump_below_{i}_{hph_i}"
            acc_below_limit.append(period_acc_below_limit)

        model += objective
        # Solve the model with increased allowed iterations
        model.solve(pulp.PULP_CBC_CMD())

        status = pulp.LpStatus[model.status]
        if status not in ['Optimal', 'Feasible']:
            # If not optimal or feasible, try again with relaxed constraints
            for var in model.variables():
                if var.cat == 'Continuous':
                    continue
                if var.cat == 'Integer':
                    var.cat = 'Continuous'
                    continue
                if var.cat == 'Boolean':
                    var.cat = 'Continuous'
            model.solve(pulp.PULP_CBC_CMD())

        def print_model_equations(m):
            # Print the objective function
            print("Objective Function:")
            objective = m.objective
            print(f"Minimize: {objective}")

            # # Print the constraints
            # print("\nConstraints:")
            # for name, constraint in m.constraints.items():
            #     print(f"{name}: {constraint}")

        # Assuming `model` is your PuLP model
        print_model_equations(model)
        print("hellooo!")

        # Retrieve the results
        plan = []
        for i, period in enumerate(periods):
            pump_consumption = sum(
                pulp.value(period_pump[i][j]) * pump_spec.power[j].consumed_power_watt
                for j in range(len(pump_spec.power))
            )
            production = sum(
                pulp.value(period_pump[i][j]) * pump_spec.power[j].heating_power_watt
                for j in range(len(pump_spec.power))
            )
            on_fraction = sum(pulp.value(pump) for pump in period_pump[i])
            pump = {
                'on_fraction': on_fraction,
                'consumption': pump_consumption,
                'cost': (pump_consumption * period['duration'] * period['price']) / 1000,
                'production': production
            }
            consumption_radiator = (
                (period['consumption']['radiator'] * acc_heat_capacity) / period['duration']
            )
            consumption_hot_water = (
                (period['consumption']['hot_water'] * acc_heat_capacity) / period['duration']
            )
            consumption_shower = (
                (period['consumption']['shower'] * acc_heat_capacity) / period['duration']
            )
            consumption_loss = (
                acc_spec.temp_loss_per_hour_degrees *
                pulp.value(acc_surrounding_diff[i]) *
                acc_heat_capacity
            )
            consumption = {
                'radiator': consumption_radiator,
                'shower': consumption_shower,
                'hot_water': consumption_hot_water,
                'loss': consumption_loss,
                'total': consumption_radiator + consumption_hot_water + consumption_shower + consumption_loss
            }
            resultPeriod = {
                'start': period['start'],
                'end': period['end'],
                'duration': period['duration'],
                'price': period['price'],
                'outdoor_temp': period['outdoor_temp'],
                'rad_flow_temp': period['rad_flow_temp'],
                'acc_temp': pulp.value(acc_temp[i]),
                'consumption': consumption,
                'pump': pump
            }
            plan.append(resultPeriod)

        return {
            'result': status,
            'ok': status in ['Optimal', 'Feasible'],
            'plan': plan,
            'cost': pulp.value(model.objective),
            'params': asdict(params)
        }



with open('output/nodeplan.json', 'r') as file:
    json_text = file.read()



service = OptimizationService()


data = json.loads(json_text)
params = fromdict(OptimizationParameters, data['params'])

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