"""Phase 2: caption JSON schema, prompts, and sampling params (plan.md section 3.1, D1/D10)."""

EVENT_TYPES = ["control_loss", "road_departure", "animal", "pedestrian", "pedalcyclist",
               "lane_change", "opposite_direction", "rear_end", "crossing_paths",
               "other", "none", "unknown"]  # NHTSA 2019 9 pre-crash groups + other/none/unknown

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_type": {"type": "string", "enum": ["urban", "suburban", "highway", "rural", "industrial", "parking", "other", "unknown"]},
        "weather": {"type": "string", "enum": ["clear", "cloudy", "rain", "snow", "fog", "unknown"]},
        "lighting": {"type": "string", "enum": ["day", "night", "twilight", "unknown"]},
        "road_type": {"type": "string", "enum": ["intersection", "straight_road", "curve", "roundabout", "ramp", "bridge_tunnel", "parking_area", "unknown"]},
        "road_surface": {"type": "string", "enum": ["dry", "wet", "snowy_icy", "unknown"]},
        "traffic_density": {"type": "string", "enum": ["none", "light", "moderate", "heavy", "unknown"]},
        "agents": {"type": "array", "items": {"type": "string", "enum": [
            "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "animal",
            "emergency_vehicle", "construction", "train_tram", "other"]}},
        "ego_maneuver": {"type": "string", "enum": ["proceeding_straight", "turning_left", "turning_right",
                                                    "lane_change", "braking", "stopped", "merging", "reversing", "unknown"]},
        "event_observed": {"type": "boolean"},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "event_description": {"type": "string"},
        "hazard_description": {"type": "string"},
        "severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe", "unknown"]},
        "caption": {"type": "string"},
        "uncertainty_notes": {"type": "string"},
    },
    "required": ["scene_type", "weather", "lighting", "road_type", "road_surface", "traffic_density",
                 "agents", "ego_maneuver", "event_observed", "event_type", "event_description",
                 "hazard_description", "severity", "caption", "uncertainty_notes"],
    "additionalProperties": False,
}

CLIP_PROMPT = """You are annotating ego-centric dashcam footage for road-safety research.
Watch the ENTIRE clip carefully, then fill the JSON annotation.

Rules:
- Report only what is VISIBLE. If a field cannot be determined from the video, use "unknown".
- event_observed is true ONLY if a collision or clear near-collision involving any road user occurs.
- event_type categories: control_loss (skid/loss of control), road_departure, animal, pedestrian,
  pedalcyclist, lane_change (incl. cut-ins/sideswipe), opposite_direction, rear_end,
  crossing_paths (intersection conflicts incl. turning across traffic), other, none.
- event_description: 1-2 sentences on what happened and who was involved ("" if no event).
- hazard_description: the main risk factor in the scene, even without an event ("" if none).
- caption: 2-3 sentences describing the drive: setting, traffic, ego behavior, and any notable moment.
- uncertainty_notes: anything you are unsure about ("" if nothing)."""

WINDOW_PROMPT = """You are annotating a SHORT EXCERPT of ego-centric dashcam footage, centered on a
suspected safety-critical moment, for road-safety research.

Rules:
- Report only what is VISIBLE in this excerpt. If a field cannot be determined, use "unknown".
- Focus on the dynamics: who moves where, who brakes/swerves, time-order of what happens.
- event_observed is true ONLY if a collision or clear near-collision occurs IN THIS EXCERPT.
- event_type categories: control_loss, road_departure, animal, pedestrian, pedalcyclist,
  lane_change (incl. cut-ins/sideswipe), opposite_direction, rear_end,
  crossing_paths (intersection conflicts incl. turning across traffic), other, none.
- event_description: 1-2 sentences, concrete and specific ("" if no event).
- caption: 2-3 sentences narrating this excerpt moment by moment.
- uncertainty_notes: anything ambiguous ("" if nothing)."""

SAMPLING = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=900)
EXTRA_TOPK = {"top_k": 20}
