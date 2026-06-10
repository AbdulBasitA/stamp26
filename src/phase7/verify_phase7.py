"""Phase 7 verify: static air-gap + completeness checks on the atlas HTMLs.
(The final human gate — open in a disconnected browser — is on the user.)"""
import re
from pathlib import Path

ROOT = Path("/home/b3ali/projects/stamp26")
gates = []


def gate(name, ok, detail=""):
    gates.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


for track in ["captions", "siglip2"]:
    f = ROOT / f"artifacts/atlas/atlas_{track}.html"
    print(f"== {f.name} ==")
    if not f.exists():
        gate("exists", False)
        continue
    html = f.read_text()
    mb = f.stat().st_size / 1e6
    gate("size sane (5-80 MB)", 5 <= mb <= 80, f"{mb:.1f} MB")
    import base64 as b64
    import gzip
    n_thumbs = html.count("data:image/jpeg;base64,")
    if n_thumbs < 1400:  # inline_data=True gzips point data — decompress blobs and count
        for blob in re.findall(r'"([A-Za-z0-9+/=]{200000,})"', html):
            try:
                n_thumbs += gzip.decompress(b64.b64decode(blob)).count(b"data:image/jpeg;base64,")
            except Exception:
                pass
    gate("thumbnails embedded", n_thumbs >= 1400, f"{n_thumbs}")
    gate("hover template present", "border-radius:4px" in html)
    gate("no live external scripts", not re.search(r'<script[^>]+src="https?://', html))
    gate("no live external links", not re.search(r'<link[^>]+href="https?://', html))
    gate("fonts inlined", html.count("data:font") >= 5, f"{html.count('data:font')} @font-face URIs")
    gate("topic tree enabled", "topic-tree" in html.lower() or "topicTree" in html)

print("PHASE7_VERIFY_PASS" if all(gates) and gates else f"PHASE7_VERIFY_FAIL ({sum(not g for g in gates)})")
print("HUMAN GATE: open artifacts/atlas/atlas_*.html in a browser with networking OFF;")
print("check hover thumbnails, search, topic tree, and every colormap layer.")
