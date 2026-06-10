"""Phase 5: NHTSA-aligned Jinja prompt templates (D15) + ClusterLayer subclass that injects
per-cluster differential metadata histograms into prompts (D12, RQ#1)."""
import jinja2
import numpy as np

from toponymy.cluster_layer import ClusterLayerText
from toponymy.templates import GET_TOPIC_NAME_REGEX, PROMPT_TEMPLATES

CATEGORIES = ["Control loss", "Road departure", "Animal", "Pedestrian", "Pedalcyclist",
              "Lane change", "Opposite direction", "Rear-end", "Crossing paths", "No conflict"]

NHTSA_BLOCK = """Use the NHTSA pre-crash scenario typology (DOT HS 812 745). The category MUST be exactly one of:
- Control loss (vehicle failure, loss of control with/without prior maneuver)
- Road departure (drifting off the road edge, departure while backing)
- Animal (animal in roadway)
- Pedestrian (pedestrian crossing or in roadway)
- Pedalcyclist (cyclist conflicts)
- Lane change (cut-ins, sideswipes, turning/drifting/parking in same direction, backing into vehicle)
- Opposite direction (head-on or wrong-way conflicts)
- Rear-end (lead vehicle stopped, decelerating, slower, accelerating; following vehicle maneuvering)
- Crossing paths (intersection conflicts: straight crossing paths, left/right turns across or into path, signal/sign violations)
- No conflict (ordinary driving with no dominant crash pattern; describe the driving setting instead)"""

FORMAT_RULES = """The topic_name MUST follow the exact format "<Category>: <specific scenario qualifier>".
The qualifier names the scenario and its salient conditions (road type, lighting, weather, traffic, maneuver).
Good examples: "Rear-end: lead vehicle braking in dense urban traffic at night", "No conflict: free-flowing highway cruising in clear daytime", "Crossing paths: left turn across oncoming traffic at signalized intersection".
Choose the category from the metadata profile and what the sample descriptions and keywords say ACTUALLY happens in this group — if collisions/near-misses are not a dominant theme, use "No conflict"."""

_SYSTEM = jinja2.Template("""
You are a road-safety analyst classifying groups of {{document_type}} from {{corpus_description}}.
Your task: analyze information about one group and assign it a {{summary_kind}} scenario name.
""" + NHTSA_BLOCK + "\n" + FORMAT_RULES + """
The response must be JSON formatted as {"topic_name":<NAME>, "topic_specificity":<SCORE>} where
SCORE is a float in [0,1] for how specific and well-defined the name is given the information.
{% if is_very_specific_summary %}
Make the qualifier specific enough to distinguish this group from other similar groups.
{% elif is_general_summary %}
Make the qualifier broad enough to cover the whole diverse group at a glance.
{% endif %}
{% if has_major_subtopics %}
Primarily use the major and minor subtopics; the name must reflect the core of ALL major subtopics.
{% endif %}
Ensure your entire response is only the JSON object, with no other text before or after it.
""")

_USER = jinja2.Template("""
Here is the information about the group of {{document_type}}:
{% if cluster_keywords %}
- Keywords for this group include: {{", ".join(cluster_keywords)}}
{% endif %}
{%- if cluster_subtopics["major"] %}
- Major subtopics: {% for s in cluster_subtopics["major"] %}
    * {{s}}{% endfor %}
{%- endif %}
{%- if cluster_subtopics["minor"] %}
- Minor subtopics: {% for s in cluster_subtopics["minor"] %}
    * {{s}}{% endfor %}
{%- endif %}
{%- if cluster_subtopics["misc"] %}
- Miscellaneous subtopics: {% for s in cluster_subtopics["misc"] %}
    * {{s}}{% endfor %}
{%- endif %}
{%- if cluster_sentences %}
- Sample {{document_type}} from this group:
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}

Based on this information, provide a {{summary_kind}} scenario name for this group.
Recall the format rules: "<Category>: <qualifier>" with Category from the NHTSA list, and
output JSON only: {"topic_name":<NAME>, "topic_specificity":<SCORE>}.
""")

_COMBINED = jinja2.Template("""
You are a road-safety analyst classifying groups of {{document_type}} from {{corpus_description}}.
""" + NHTSA_BLOCK + "\n" + FORMAT_RULES + """
{% if cluster_keywords %}
- Keywords for this group include: {{", ".join(cluster_keywords)}}
{% endif %}
{%- if cluster_subtopics["major"] %}
- Major subtopics: {% for s in cluster_subtopics["major"] %}
    * {{s}}{% endfor %}
{%- endif %}
{%- if cluster_sentences %}
- Sample {{document_type}} from this group:
{%- for sentence in cluster_sentences %}
{{exemplar_start_delimiter}}{{sentence}}{{exemplar_end_delimiter}}
{%- endfor %}
{%- endif %}

Provide a {{summary_kind}} scenario name. Output JSON only:
{"topic_name":<NAME>, "topic_specificity":<SCORE>}
""")


def build_templates():
    """Full custom dict: driving layer templates + stock disambiguation with the LLM's
    renames DISCARDED (issue #101 workaround — keeps taxonomy-format names intact, D15)."""
    disamb = dict(PROMPT_TEMPLATES["disambiguate_topics"])
    disamb["extract_topic_names"] = lambda json_response, old_names, raw: list(old_names)
    return {
        "layer": {
            "system": _SYSTEM,
            "user": _USER,
            "combined": _COMBINED,
            "extract_topic_name": lambda j: str(j["topic_name"]).strip(),
            "get_topic_name_regex": GET_TOPIC_NAME_REGEX,
        },
        "disambiguate_topics": disamb,
    }


class DrivingClusterLayer(ClusterLayerText):
    """ClusterLayerText + per-cluster differential metadata histogram appended to each prompt.
    Call set_layer_metadata(df) (rows aligned with the objects passed to Toponymy.fit) first."""
    METADATA = None

    def make_prompts(self, *args, **kwargs):
        super().make_prompts(*args, **kwargs)
        meta = DrivingClusterLayer.METADATA
        if meta is None:
            return self.prompts
        for c in range(len(self.prompts)):
            block = self._histogram_block(meta, c)
            if isinstance(self.prompts[c], dict):
                self.prompts[c]["user"] = self.prompts[c]["user"] + block
            else:
                self.prompts[c] = self.prompts[c] + block
        return self.prompts

    def _histogram_block(self, meta, c):
        m = self.cluster_labels == c
        sub, n = meta[m], int(m.sum())
        lines = [f"\n- Human-annotation metadata profile of this group (n={n} clips) vs the whole corpus:"]
        lines.append(f"    * collision/near-collision label: {sub['label'].mean():.0%} of this group "
                     f"(corpus average {meta['label'].mean():.0%})")
        for col, label in [("scene", "scene"), ("light_conditions", "lighting"), ("weather", "weather")]:
            d = (sub[col].value_counts(normalize=True)
                 - meta[col].value_counts(normalize=True)).dropna().sort_values(ascending=False)
            tops = [f"{k} {sub[col].value_counts(normalize=True)[k]:.0%} ({v:+.0%} vs corpus)"
                    for k, v in d.head(2).items() if abs(v) > 0.08]
            if tops:
                lines.append(f"    * {label}: " + ", ".join(tops))
        return "\n".join(lines) + "\n"


def set_layer_metadata(df):
    DrivingClusterLayer.METADATA = df.reset_index(drop=True)
