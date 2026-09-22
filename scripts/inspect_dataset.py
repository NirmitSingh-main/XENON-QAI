import tarfile
from pathlib import Path

path = Path("data/00-images.tar.xz")

compressed = path.stat().st_size

print("Reading archive contents...")
print("This does NOT extract anything.")

with tarfile.open(path, "r:xz") as tar:
    total = sum(
        member.size
        for member in tar
        if member.isfile()
    )

print()
print(f"Compressed size : {compressed / 1_000_000_000:.2f} GB")
print(f"Extracted size  : {total / 1_000_000_000:.2f} GB")
print(f"Extracted size  : {total / 1024**3:.2f} GiB")