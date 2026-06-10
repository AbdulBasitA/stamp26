"""Phase 2: caption JSON schema, prompts, and sampling params (plan.md section 3.1, D1/D10).

v2 (2026-06-10): enum-artifact fix. Strict guided decoding forces prefix-sharing enum values
(e.g. model intends "sunny", grammar only allows "snow" after token "s") and the model then
rationalizes the forced value in later free text. Fixes: (1) caption narration is generated
FIRST, constrained fields after; (2) enums cover common synonyms; (3) values diverge early
in their token prefixes. v1 captions archived in artifacts/captions/enumbug/.
"""

EVENT_TYPES = ["control_loss", "road_departure", "animal", "pedestrian", "pedalcyclist",
               "lane_change", "opposite_direction", "rear_end", "crossing_paths",
               "other", "none", "unknown"]  # NHTSA 2019 9 pre-crash groups + other/none/unknown

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        # free narration FIRST: the model commits to enums only after describing what it saw
        "caption": {"type": "string"},
        "scene_type": {"type": "string", "enum": ["urban", "suburban", "residential", "highway",
                                                  "rural", "industrial", "parking_lot", "other", "unknown"]},
        "weather": {"type": "string", "enum": ["sunny", "clear", "cloudy", "overcast", "rain",
                                               "snow", "fog", "unknown"]},
        "lighting": {"type": "string", "enum": ["daytime", "dark_night", "twilight", "unknown"]},
        "road_type": {"type": "string", "enum": ["intersection", "straight_road", "street", "curve",
                                                 "roundabout", "ramp", "bridge_or_tunnel",
                                                 "parking_area", "unknown"]},
        "road_surface": {"type": "string", "enum": ["dry", "wet", "snowy_icy", "unknown"]},
        "traffic_density": {"type": "string", "enum": ["none", "light", "moderate", "heavy", "unknown"]},
        "agents": {"type": "array", "items": {"type": "string", "enum": [
            "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "animal",
            "emergency_vehicle", "construction", "train_tram", "other"]}},
        "ego_maneuver": {"type": "string", "enum": ["proceeding_straight", "turning_left", "turning_right",
                                                    "lane_change", "braking", "stopped", "merging",
                                                    "reversing", "unknown"]},
        "event_observed": {"type": "boolean"},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe", "unknown"]},
        "event_description": {"type": "string"},
        "hazard_description": {"type": "string"},
        "uncertainty_notes": {"type": "string"},
    },
    "required": ["caption", "scene_type", "weather", "lighting", "road_type", "road_surface",
                 "traffic_density", "agents", "ego_maneuver", "event_observed", "event_type",
                 "severity", "event_description", "hazard_description", "uncertainty_notes"],
    "additionalProperties": False,
}

_SHARED_RULES = """Rules:
- FIRST write `caption`: 2-3 sentences narrating the drive from what you actually see.
- THEN fill the structured fields, consistent with your own caption. If a field cannot be
  determined from the video, use "unknown" — never guess.
- event_observed is true ONLY if a collision or clear near-collision involving any road user occurs.
- event_type categories: control_loss (skid/loss of control), road_departure, animal, pedestrian,
  pedalcyclist, lane_change (incl. cut-ins/sideswipe), opposite_direction, rear_end,
  crossing_paths (intersection conflicts incl. turning across traffic), other, none.
- event_description: 1-2 sentences on what happened and who was involved ("" if no event).
- hazard_description: the main risk factor in the scene, even without an event ("" if none).
- uncertainty_notes: anything you are unsure about ("" if nothing)."""

CLIP_PROMPT = ("You are annotating ego-centric dashcam footage for road-safety research.\n"
               "Watch the ENTIRE clip carefully, then produce the JSON annotation.\n\n" + _SHARED_RULES)

WINDOW_PROMPT = ("You are annotating a SHORT EXCERPT of ego-centric dashcam footage, centered on a\n"
                 "suspected safety-critical moment, for road-safety research. Focus on the dynamics:\n"
                 "who moves where, who brakes/swerves, the time-order of what happens. Watch to the\n"
                 "very end, then produce the JSON annotation.\n\n" + _SHARED_RULES)

SAMPLING = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=900)
EXTRA_TOPK = {"top_k": 20}
