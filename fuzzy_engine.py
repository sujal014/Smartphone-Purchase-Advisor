"""
Fuzzy Logic Engine for Smartphone Suitability Scoring
-------------------------------------------------------
This is a genuine Mamdani-style Fuzzy Inference System (FIS) built with
scikit-fuzzy. It takes 4 crisp inputs per phone and produces a crisp
"suitability" score (0-100) via:

  1. Fuzzification   -> membership functions convert crisp inputs into
                         degrees of membership in linguistic sets
                         (low / medium / high)
  2. Rule evaluation  -> a rule base of IF-THEN fuzzy rules fires with
                         a strength determined by the antecedent
                         memberships (AND = min, OR = max)
  3. Aggregation      -> fired rule outputs are combined into one
                         fuzzy output set
  4. Defuzzification  -> centroid method converts the fuzzy output set
                         back into a single crisp suitability score

Inputs:
    price_fit   (0-100): how well the phone's price matches the user's budget
    camera      (0-10):  phone's camera quality, weighted by user's stated
                          camera importance
    performance (0-10):  phone's performance/gaming quality, weighted by
                          user's stated performance importance
    battery     (0-10):  phone's battery quality, weighted by user's stated
                          battery importance

Output:
    suitability (0-100): overall fit score for ranking phones
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system() -> ctrl.ControlSystem:
    """Constructs the Mamdani fuzzy control system (antecedents, consequent,
    membership functions, and rule base) once. Reused for every phone."""

    # ---- Antecedents (inputs) ----
    price_fit = ctrl.Antecedent(np.arange(0, 101, 1), "price_fit")
    camera = ctrl.Antecedent(np.arange(0, 11, 1), "camera")
    performance = ctrl.Antecedent(np.arange(0, 11, 1), "performance")
    battery = ctrl.Antecedent(np.arange(0, 11, 1), "battery")

    # ---- Consequent (output) ----
    suitability = ctrl.Consequent(np.arange(0, 101, 1), "suitability")

    # ---- Membership functions (fuzzification) ----
    price_fit["low"] = fuzz.trimf(price_fit.universe, [0, 0, 50])
    price_fit["medium"] = fuzz.trimf(price_fit.universe, [20, 50, 80])
    price_fit["high"] = fuzz.trimf(price_fit.universe, [50, 100, 100])

    for var in (camera, performance, battery):
        var["low"] = fuzz.trimf(var.universe, [0, 0, 5])
        var["medium"] = fuzz.trimf(var.universe, [2, 5, 8])
        var["high"] = fuzz.trimf(var.universe, [5, 10, 10])

    suitability["low"] = fuzz.trimf(suitability.universe, [0, 0, 50])
    suitability["medium"] = fuzz.trimf(suitability.universe, [20, 50, 80])
    suitability["high"] = fuzz.trimf(suitability.universe, [50, 100, 100])

    # ---- Rule base ----
    rules = [
        ctrl.Rule(price_fit["low"], suitability["low"]),
        ctrl.Rule(price_fit["low"] & camera["low"], suitability["low"]),
        ctrl.Rule(
            camera["low"] & performance["low"] & battery["low"], suitability["low"]
        ),
        ctrl.Rule(
            price_fit["high"] & camera["high"] & performance["high"],
            suitability["high"],
        ),
        ctrl.Rule(price_fit["high"] & battery["high"], suitability["high"]),
        ctrl.Rule(
            price_fit["medium"] & (camera["high"] | performance["high"]),
            suitability["high"],
        ),
        ctrl.Rule(
            performance["high"] & battery["high"] & price_fit["medium"],
            suitability["high"],
        ),
        ctrl.Rule(
            price_fit["medium"] & camera["medium"] & performance["medium"],
            suitability["medium"],
        ),
        ctrl.Rule(
            camera["medium"] & performance["medium"] & battery["medium"]
            & price_fit["medium"],
            suitability["medium"],
        ),
        ctrl.Rule(
            price_fit["high"] & camera["medium"] & performance["medium"]
            & battery["medium"],
            suitability["medium"],
        ),
    ]

    return ctrl.ControlSystem(rules)


def compute_price_fit(price: float, budget: float) -> float:
    """Crisp pre-processing: how close a phone's price is to the user's
    budget, expressed on a 0-100 scale. A phone right at (or under) budget
    scores highest; going over budget is penalized more steeply."""
    if price <= budget:
        # Small reward for being close to budget (uses more of the budget
        # generally means better specs), but never below 70 if in-budget.
        ratio = price / budget if budget > 0 else 1
        return float(np.clip(70 + 30 * ratio, 0, 100))
    else:
        over_pct = (price - budget) / budget
        return float(np.clip(100 - over_pct * 200, 0, 100))


def score_phone(
    system: ctrl.ControlSystem,
    price: float,
    budget: float,
    camera_spec: float,
    performance_spec: float,
    battery_spec: float,
    camera_importance: float,
    performance_importance: float,
    battery_importance: float,
) -> dict:
    """Runs one phone through the fuzzy inference system and returns the
    crisp inputs used plus the defuzzified suitability score."""

    price_fit_val = compute_price_fit(price, budget)

    # Weight each spec by the user's stated importance (1-10 -> 0.1-1.0),
    # so a great camera matters less to someone who said it doesn't matter.
    camera_val = camera_spec * (camera_importance / 10)
    performance_val = performance_spec * (performance_importance / 10)
    battery_val = battery_spec * (battery_importance / 10)

    sim = ctrl.ControlSystemSimulation(system)
    sim.input["price_fit"] = price_fit_val
    sim.input["camera"] = float(np.clip(camera_val, 0, 10))
    sim.input["performance"] = float(np.clip(performance_val, 0, 10))
    sim.input["battery"] = float(np.clip(battery_val, 0, 10))
    sim.compute()

    return {
        "price_fit": round(price_fit_val, 1),
        "camera_input": round(camera_val, 1),
        "performance_input": round(performance_val, 1),
        "battery_input": round(battery_val, 1),
        "suitability": round(float(sim.output["suitability"]), 1),
    }
