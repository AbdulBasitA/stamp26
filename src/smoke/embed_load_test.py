"""Phase 0 smoke: load all three embedders and push real Nexar data through them."""
import warnings

warnings.filterwarnings("ignore")
import torch

DEV = "cuda:0"
CLIP = "data/nexar/train/positive/00822.mp4"

from torchcodec.decoders import VideoDecoder

dec = VideoDecoder(CLIP)
n = dec.metadata.num_frames

# V-JEPA2: 16 uniformly spaced frames, mean-pooled patch tokens
from transformers import AutoModel, AutoVideoProcessor

vj_id = "facebook/vjepa2-vitl-fpc64-256"
vj_proc = AutoVideoProcessor.from_pretrained(vj_id)
vj = AutoModel.from_pretrained(vj_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
idx = torch.linspace(0, n - 1, 16).long()
frames = dec.get_frames_at(indices=idx.tolist()).data  # (16, 3, H, W) uint8
inputs = vj_proc(frames, return_tensors="pt").to(DEV)
with torch.inference_mode():
    out = vj.get_vision_features(**inputs)
emb_vj = out.mean(dim=1)
print(f"V-JEPA2 OK: clip embedding {tuple(emb_vj.shape)} from {tuple(out.shape)} patch tokens")
del vj
torch.cuda.empty_cache()

# SigLIP2: one real frame through image features
from transformers import AutoModel as AM2, AutoProcessor

sg_id = "google/siglip2-so400m-patch16-384"
sg_proc = AutoProcessor.from_pretrained(sg_id)
sg = AM2.from_pretrained(sg_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
from PIL import Image

img = Image.fromarray(frames[8].permute(1, 2, 0).numpy())
with torch.inference_mode():
    feats = sg.get_image_features(**sg_proc(images=img, return_tensors="pt").to(DEV))
print(f"SigLIP2 OK: frame embedding {tuple(feats.shape)}")
del sg
torch.cuda.empty_cache()

# Qwen3-Embedding-0.6B via sentence-transformers (Toponymy's text_embedding_model)
from sentence_transformers import SentenceTransformer

st = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=DEV)
embs = st.encode(["a rainy intersection with a pedestrian crossing", "highway driving at night"],
                 normalize_embeddings=True)
print(f"Qwen3-Embedding-0.6B OK: {embs.shape}")
print("EMBED_SMOKE_PASS")
