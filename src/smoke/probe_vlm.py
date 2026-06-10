"""Phase 0 smoke: probe a captioner server — text request + real-video request with guided JSON."""
import json
import sys
import time

from openai import OpenAI

port = sys.argv[1]
client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
model = client.models.list().data[0].id

t0 = time.time()
r = client.chat.completions.create(
    model=model, max_tokens=32, temperature=0.7,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
)
print(f"text probe OK ({time.time()-t0:.1f}s): {r.choices[0].message.content!r}")

schema = {
    "type": "object",
    "properties": {
        "scene_type": {"type": "string", "enum": ["urban", "highway", "suburban", "rural", "other", "unknown"]},
        "weather": {"type": "string", "enum": ["clear", "cloudy", "rain", "snow", "unknown"]},
        "lighting": {"type": "string", "enum": ["day", "night", "twilight", "unknown"]},
        "event_observed": {"type": "boolean"},
        "caption": {"type": "string"},
    },
    "required": ["scene_type", "weather", "lighting", "event_observed", "caption"],
    "additionalProperties": False,
}
t0 = time.time()
r = client.chat.completions.create(
    model=model, max_tokens=512,
    temperature=0.7, top_p=0.8, presence_penalty=1.5,
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "file:///home/b3ali/projects/stamp26/data/nexar/train/positive/00822.mp4"}},
        {"type": "text", "text": "Describe this dashcam clip. Report scene type, weather, lighting, whether a collision or near-collision occurs, and a 2-sentence caption."},
    ]}],
    response_format={"type": "json_schema", "json_schema": {"name": "caption", "strict": True, "schema": schema}},
    extra_body={"mm_processor_kwargs": {"fps": 1}, "chat_template_kwargs": {"enable_thinking": False}},
)
out = json.loads(r.choices[0].message.content)
print(f"video+guided-JSON probe OK ({time.time()-t0:.1f}s): {json.dumps(out)}")
print("VLM_PROBE_PASS")
