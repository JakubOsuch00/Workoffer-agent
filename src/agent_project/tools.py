import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

Level = Literal["intern", "junior", "mid", "senior", "lead"]


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetProfileArgs(StrictArgs):
    pass


class SearchOffersArgs(StrictArgs):
    query: str = Field(
        min_length=1,
        max_length=100,
        description="Keywords matched against title, skills, and description.",
    )
    location: str | None = Field(
        default=None, max_length=50, description="City name or 'Remote'."
    )
    level: Level | None = Field(default=None, description="Seniority level.")
    limit: int = Field(default=10, ge=1, le=30, description="Maximum results.")


class GetOfferArgs(StrictArgs):
    offer_id: str = Field(pattern=r"^J\d{2}$", description="Offer id, e.g. 'J01'.")


class CheckMatchArgs(StrictArgs):
    offer_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Offer ids to compare with the candidate profile.",
    )


class FinalAnswerArgs(StrictArgs):
    recommended_offer_ids: list[str] = Field(
        max_length=5,
        description="Offers that fully match and were not applied to yet.",
    )
    skills_to_learn: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Missing skills that block near-miss offers.",
    )
    summary: str = Field(
        min_length=1, max_length=2000, description="Short answer for the candidate."
    )


def load_offers() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "offers.json").read_text(encoding="utf-8"))


def load_profile() -> dict[str, Any]:
    return json.loads((DATA_DIR / "profile.json").read_text(encoding="utf-8"))


def load_applied_ids() -> set[str]:
    with (DATA_DIR / "applications.csv").open(encoding="utf-8", newline="") as f:
        return {row["offer_id"] for row in csv.DictReader(f)}


def _summary(offer: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "company", "location", "level")
    return {key: offer[key] for key in keys}


def get_profile(args: GetProfileArgs) -> dict[str, Any]:
    return {"ok": True, "profile": load_profile()}


def search_offers(args: SearchOffersArgs) -> dict[str, Any]:
    words = args.query.lower().split()
    scored = []
    for offer in load_offers():
        if args.location and offer["location"].lower() != args.location.lower():
            continue
        if args.level and offer["level"] != args.level:
            continue
        text = " ".join(
            [offer["title"], offer["description"], *offer["required_skills"]]
        ).lower()
        score = sum(word in text for word in words)
        if score:
            scored.append((-score, offer["id"], offer))
    scored.sort(key=lambda item: item[:2])
    results = [_summary(offer) for _, _, offer in scored[: args.limit]]
    return {"ok": True, "total_found": len(scored), "offers": results}


def get_offer(args: GetOfferArgs) -> dict[str, Any]:
    for offer in load_offers():
        if offer["id"] == args.offer_id:
            return {"ok": True, "offer": offer}
    return {"ok": False, "error": "not_found", "offer_id": args.offer_id}


def match_offer(
    offer: dict[str, Any], profile: dict[str, Any], applied: set[str]
) -> dict[str, Any]:
    skills = {skill.lower() for skill in profile["skills"]}
    required = offer["required_skills"]
    matched = [skill for skill in required if skill.lower() in skills]
    missing = [skill for skill in required if skill.lower() not in skills]

    salary = offer["salary_min_pln"]
    if salary is None:
        salary_ok, salary_note = False, "salary not published"
    else:
        salary_ok = salary >= profile["min_salary_pln"]
        salary_note = f"minimum {salary} PLN vs required {profile['min_salary_pln']}"

    checks = {
        "level_ok": offer["level"] in profile["accepted_levels"],
        "location_ok": offer["location"] in profile["preferred_locations"],
        "salary_ok": salary_ok,
        "skills_ok": not missing,
    }
    return {
        "offer_id": offer["id"],
        "title": offer["title"],
        "company": offer["company"],
        "is_full_match": all(checks.values()),
        "already_applied": offer["id"] in applied,
        "checks": checks,
        "salary_note": salary_note,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def check_match(args: CheckMatchArgs) -> dict[str, Any]:
    offers = {offer["id"]: offer for offer in load_offers()}
    profile, applied = load_profile(), load_applied_ids()
    results = []
    for offer_id in dict.fromkeys(args.offer_ids):
        if offer_id in offers:
            results.append(match_offer(offers[offer_id], profile, applied))
        else:
            results.append({"offer_id": offer_id, "error": "not_found"})
    return {"ok": True, "results": results}


ToolFunc = Callable[[Any], dict[str, Any]]

TOOLS: dict[str, tuple[type[StrictArgs], ToolFunc, str]] = {
    "get_profile": (
        GetProfileArgs,
        get_profile,
        "Read the candidate profile: skills, accepted levels, preferred "
        "locations, and minimum salary.",
    ),
    "search_offers": (
        SearchOffersArgs,
        search_offers,
        "Search job offers by keywords, optionally filtered by location and "
        "level. Returns short summaries only.",
    ),
    "get_offer": (
        GetOfferArgs,
        get_offer,
        "Read all details of one job offer by id.",
    ),
    "check_match": (
        CheckMatchArgs,
        check_match,
        "Compare up to 10 offers with the candidate profile. Returns level, "
        "location, salary, and skill checks, missing skills, and whether the "
        "candidate already applied.",
    ),
}

FINAL_ANSWER_DESCRIPTION = (
    "Finish the task. Recommend only offers that check_match reported as "
    "is_full_match=true and already_applied=false in this run."
)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    # Optional[X] -> X, the model handles plain types better than anyOf
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        variants = prop.pop("anyOf", None)
        if variants:
            prop.update(next(v for v in variants if v.get("type") != "null"))
    return schema


def tool_schemas() -> list[dict[str, Any]]:
    entries = [(name, model, desc) for name, (model, _, desc) in TOOLS.items()]
    entries.append(("final_answer", FinalAnswerArgs, FINAL_ANSWER_DESCRIPTION))
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": _clean_schema(model.model_json_schema()),
            },
        }
        for name, model, description in entries
    ]


def parse_arguments(
    model: type[StrictArgs], raw_arguments: str | dict[str, Any] | None
) -> tuple[StrictArgs | None, dict[str, Any] | None]:
    try:
        data = raw_arguments
        if isinstance(raw_arguments, str):
            data = json.loads(raw_arguments) if raw_arguments.strip() else {}
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("arguments must be a JSON object")
        return model.model_validate(data), None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, {"ok": False, "error": "invalid_json", "message": str(exc)}
    except ValidationError as exc:
        details = [
            {"field": ".".join(map(str, e["loc"])) or "(root)", "message": e["msg"]}
            for e in exc.errors()
        ]
        return None, {"ok": False, "error": "invalid_arguments", "details": details}


def execute_tool(name: str, raw_arguments: str | dict[str, Any] | None) -> dict:
    if name not in TOOLS:
        return {
            "ok": False,
            "error": "unknown_tool",
            "message": f"Unknown tool '{name}'. Allowed: {', '.join(TOOLS)}.",
        }
    model, func, _ = TOOLS[name]
    args, error = parse_arguments(model, raw_arguments)
    if error:
        return error
    return func(args)
