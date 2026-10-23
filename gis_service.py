"""
gis_service.py - RescueBot 2.0 (Member 2: GIS & Hazards)

WHAT THIS FILE DOES
--------------------
This is the ONE file the rest of the team (and the backend integration
owner) should actually import. It ties geo_data.py, hazard_engine.py
and risk_analysis.py together into:

  1. GISService
     A plain class with four methods:
       - get_area_info()                      -> for GET /api/geo/area
       - get_hazard_snapshot(scenario_id, ..) -> for POST /api/hazards/simulate
       - get_flood_risk(hours, scenario_id)   -> one risk tool
       - get_affected_areas(scenario_id, hrs) -> the other risk tool

  2. FloodServiceAdapter
     The EXACT class named in Member 1's MEMBER1_INTEGRATION_HANDOFF.md:

         adapters = BackendAdapters(
             flood=FloodServiceAdapter(...),   # Member 2
             ...
         )

     It implements ai_tools.FloodBackendAdapter's two methods with the
     exact parameter names Member 1's code calls by keyword
     (get_flood_risk(hours, scenario_id) and
     get_affected_areas(scenario_id, hours)), and otherwise just hands
     the call to GISService. Construct it with no arguments:

         FloodServiceAdapter()

HOW TO WIRE THIS IN (for whoever owns the backend composition root,
per the team report that is usually Member 3, since they mount the API)
--------------------------------------------------------------------
    from ai_agent import AIAssistant
    from ai_tools import BackendAdapters, create_tool_registry_from_env
    from gemini_service import GeminiService
    from gis_service import FloodServiceAdapter

    adapters = BackendAdapters(
        flood=FloodServiceAdapter(),      # <- this file
        rescue=RescueServiceAdapter(...), # <- Member 3's file
    )
    registry = create_tool_registry_from_env(adapters=adapters)
    assistant = AIAssistant(gemini_service=GeminiService(), tool_registry=registry)

If anything goes wrong (bad scenario_id, bad hours), the methods below
RAISE a plain Python exception instead of returning an error dict.
That is deliberate - Member 1's ToolRegistry.execute() already expects
exactly this, and turns the exception into a safe "SERVICE_UNAVAILABLE"
result for the AI assistant automatically. Nothing else needs to catch
these exceptions.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geo_data
import hazard_engine
import risk_analysis


class GISService:
    """The GIS/hazard facade. One instance can be shared by the whole
    backend process; it holds no mutable state of its own (every
    scenario's data is recomputed from its fixed seed each call, so the
    answers are always reproducible)."""

    def get_area_info(self) -> dict[str, Any]:
        """For GET /api/geo/area."""
        return geo_data.get_area_info()

    def get_hazard_snapshot(
        self,
        scenario_id: str,
        time_steps: tuple[int, ...] = hazard_engine.DEFAULT_TIME_STEPS,
    ) -> dict[str, Any]:
        """For POST /api/hazards/simulate. Returns the 0/2/5/10-hour
        (by default) grid-aligned snapshots Members 3 and 4 need, with
        coordinates, blocked cells, and assumptions included."""
        return hazard_engine.get_snapshots(scenario_id, time_steps)

    def get_flood_risk(self, hours: int, scenario_id: str) -> dict[str, Any]:
        return risk_analysis.get_flood_risk(hours, scenario_id)

    def get_affected_areas(self, scenario_id: str, hours: int) -> dict[str, Any]:
        return risk_analysis.get_affected_areas(scenario_id, hours)


class FloodServiceAdapter:
    """The adapter Member 1's ai_tools.py / MEMBER1_INTEGRATION_HANDOFF.md
    expects, implementing FloodBackendAdapter's two methods with the
    exact required parameter names and order:

        get_flood_risk(self, hours: int, scenario_id: str)
        get_affected_areas(self, scenario_id: str, hours: int)

    Construct with no arguments (a default GISService is created for
    you), or pass your own `service=` if you want to swap in a real
    data source later without touching ai_tools.py or ai_agent.py.
    """

    def __init__(self, service: GISService | None = None) -> None:
        self._service = service or GISService()

    def get_flood_risk(self, hours: int, scenario_id: str) -> Mapping[str, Any]:
        return self._service.get_flood_risk(hours, scenario_id)

    def get_affected_areas(
        self, scenario_id: str, hours: int
    ) -> Mapping[str, Any]:
        return self._service.get_affected_areas(scenario_id, hours)


# A ready-to-use singleton, for callers that just want "the" adapter
# without constructing their own. The backend composition root can
# still make its own instance instead if it prefers.
default_flood_adapter = FloodServiceAdapter()
