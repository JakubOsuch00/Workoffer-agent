from agent_project.tools import (
    TOOLS,
    execute_tool,
    load_applied_ids,
    load_offers,
    load_profile,
    match_offer,
    tool_schemas,
)


def test_data_has_thirty_offers() -> None:
    ids = [offer["id"] for offer in load_offers()]
    assert ids == [f"J{i:02d}" for i in range(1, 31)]


def test_expected_matches_for_main_example() -> None:
    profile, applied = load_profile(), load_applied_ids()
    full = {
        offer["id"]
        for offer in load_offers()
        if match_offer(offer, profile, applied)["is_full_match"]
    }
    assert full == {"J01", "J05", "J11", "J19", "J24", "J29"}
    assert full - applied == {"J01", "J19", "J24"}


def test_get_profile() -> None:
    result = execute_tool("get_profile", "{}")
    assert result["ok"] and "Python" in result["profile"]["skills"]


def test_search_offers_filters_location_and_level() -> None:
    result = execute_tool(
        "search_offers", {"query": "python", "location": "warsaw", "level": "junior"}
    )
    ids = [offer["id"] for offer in result["offers"]]
    assert result["ok"]
    assert "J01" in ids and "J02" not in ids and "J04" not in ids


def test_search_offers_returns_summaries_only() -> None:
    offer = execute_tool("search_offers", {"query": "data"})["offers"][0]
    assert set(offer) == {"id", "title", "company", "location", "level"}


def test_get_offer_not_found() -> None:
    assert execute_tool("get_offer", {"offer_id": "J99"}) == {
        "ok": False,
        "error": "not_found",
        "offer_id": "J99",
    }


def test_check_match_reports_missing_skill_and_salary() -> None:
    result = execute_tool("check_match", {"offer_ids": ["J01", "J03", "J07", "J05"]})
    j01, j03, j07, j05 = result["results"]
    assert j01["is_full_match"] and not j01["already_applied"]
    assert j03["missing_skills"] == ["Django"] and not j03["is_full_match"]
    assert j07["salary_note"] == "salary not published" and not j07["is_full_match"]
    assert j05["is_full_match"] and j05["already_applied"]


def test_check_match_unknown_id_is_reported() -> None:
    result = execute_tool("check_match", {"offer_ids": ["J01", "X1"]})
    assert result["results"][1] == {"offer_id": "X1", "error": "not_found"}


def test_unknown_tool() -> None:
    result = execute_tool("delete_offer", {"offer_id": "J01"})
    assert result["error"] == "unknown_tool"


def test_extra_field_is_rejected() -> None:
    result = execute_tool("get_offer", {"offer_id": "J01", "force": True})
    assert result["error"] == "invalid_arguments"


def test_wrong_type_is_not_coerced() -> None:
    result = execute_tool("search_offers", {"query": "python", "limit": "5"})
    assert result["error"] == "invalid_arguments"
    assert result["details"][0]["field"] == "limit"


def test_invalid_json() -> None:
    assert execute_tool("get_offer", "{not json")["error"] == "invalid_json"


def test_schemas_match_tools() -> None:
    schemas = tool_schemas()
    names = [schema["function"]["name"] for schema in schemas]
    assert names == [*TOOLS, "final_answer"]
    for schema in schemas:
        assert "anyOf" not in str(schema)
        assert schema["function"]["description"]
