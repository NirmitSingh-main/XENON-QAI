from pathlib import Path
import h5py

ROOT = Path("data/extracted")

files = list(ROOT.rglob("*.hdf5"))

print(f"Found {len(files)} HDF5 files")

for path in files:
    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    try:
        with h5py.File(path, "r") as f:

            print("\nTop-level objects:")

            for name, obj in f.items():
                if isinstance(obj, h5py.Group):
                    print(f"GROUP: {name}")
                    print(f"  Number of objects: {len(obj)}")

                elif isinstance(obj, h5py.Dataset):
                    print(f"DATASET: {name}")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")

            print("\nSample contents of each group:")

            for group_name, group in f.items():

                if not isinstance(group, h5py.Group):
                    continue

                print(f"\n[{group_name}]")

                names = list(group.keys())

                for name in names[:5]:
                    obj = group[name]

                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name}")
                        print(f"    Shape: {obj.shape}")
                        print(f"    Dtype: {obj.dtype}")

                        try:
                            print(f"    First values: {obj[:5]}")
                        except:
                            pass

                    elif isinstance(obj, h5py.Group):
                        print(f"  {name} (GROUP)")

                if len(names) > 5:
                    print(f"  ... and {len(names) - 5} more objects")

    except Exception as e:
        print("ERROR:", e)