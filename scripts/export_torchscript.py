import os, gc, torch, subprocess
from spandrel import ModelLoader

def ram(label=""):
    print(f"=== RAM {label} ===")
    subprocess.run(["free", "-h"])

tile = int(os.environ["TILE_SIZE"])
print(f"Tile: {tile}x{tile}")

ram("before load")
arch = ModelLoader().load_from_file("model.pth")
model = arch.model.eval()
ram("after load")

dummy = torch.zeros(1, 3, tile, tile)

print("Warm-up...")
with torch.no_grad():
    out = model(dummy)
print(f"Output shape: {out.shape}")
assert out.shape == (1, 3, tile * 2, tile * 2), f"Wrong shape: {out.shape}"
print("Shape OK")
del out; gc.collect()
ram("after warm-up")

print("JIT trace...")
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)
del model; gc.collect()
ram("after trace")

traced.save("model.pt")
del traced; gc.collect()
print(f"model.pt: {os.path.getsize('model.pt')/1e6:.1f} MB")
ram("after save")
