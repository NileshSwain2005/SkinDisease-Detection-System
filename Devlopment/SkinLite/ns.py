from pathlib import Path
import zipfile
import os
import sys


MODEL = Path("best_densenet121.keras").resolve()


print("=" * 70)
print("             SKINLITE MODEL DIAGNOSTIC")
print("=" * 70)

print(f"\n📂 Current directory:")
print(Path.cwd())

print(f"\n🧠 Model:")
print(MODEL)


# =========================================================
# 1. CHECK EXISTENCE
# =========================================================

if not MODEL.exists():
    print("\n❌ ERROR: Model file does not exist.")
    sys.exit(1)

print("\n✅ Model exists.")


# =========================================================
# 2. CHECK WHETHER IT IS A FILE
# =========================================================

if MODEL.is_dir():
    print("\n❌ ERROR: best_densenet121.keras is a DIRECTORY.")
    print("It must be a normal file.")
    sys.exit(1)

print("✅ It is a normal file.")


# =========================================================
# 3. GET FILE INFORMATION
# =========================================================

try:
    info = MODEL.stat()

    size_mb = info.st_size / (1024 * 1024)

    print(f"📦 Size: {size_mb:.2f} MB")
    print(f"🔐 Permissions: {oct(info.st_mode)}")

except PermissionError:
    print("\n❌ WINDOWS PERMISSION ERROR")
    print("Python cannot read the model file.")
    sys.exit(1)


# =========================================================
# 4. CHECK READ ACCESS
# =========================================================

if not os.access(MODEL, os.R_OK):

    print("\n❌ Python does not have read permission.")
    sys.exit(1)

print("✅ Python has read permission.")


# =========================================================
# 5. TRY OPENING THE FILE
# =========================================================

print("\n🔓 Attempting to open model...")

try:

    with open(MODEL, "rb") as f:

        header = f.read(8)

    print("✅ File can be opened.")

    print(f"📄 Header bytes: {header!r}")

except PermissionError:

    print("\n❌ PermissionError while opening file.")

    print("""
Possible causes:

1. Another program is locking the file.
2. Streamlit/Python is using the model.
3. Antivirus is scanning the file.
4. Windows permissions are blocking access.
5. The file is located in a protected/synchronized folder.
""")

    sys.exit(1)


# =========================================================
# 6. CHECK KERAS ZIP STRUCTURE
# =========================================================

print("\n🔍 Checking Keras archive...")

try:

    with zipfile.ZipFile(MODEL, "r") as archive:

        print("\n📂 Files inside model:")

        for name in archive.namelist():

            print(f"   ├── {name}")


        print("\n🔐 Running CRC-32 verification...")

        corrupted = archive.testzip()


        if corrupted:

            print("\n❌ MODEL ARCHIVE IS CORRUPTED")

            print(
                f"❌ Corrupted file: {corrupted}"
            )

            print("""
The .keras container itself can be opened,
but one of its internal files is corrupted.

This means the model must be regenerated
or replaced with a valid copy.
""")

        else:

            print("\n✅ CRC-32 verification passed.")
            print("✅ .keras archive is healthy.")


except PermissionError:

    print("\n❌ PermissionError while reading .keras archive.")

    sys.exit(1)


except zipfile.BadZipFile:

    print("\n❌ Invalid .keras archive.")

    print(
        "best_densenet121.keras is not a valid Keras archive."
    )

    sys.exit(1)


except Exception as error:

    print("\n❌ Unexpected error:")
    print(type(error).__name__)
    print(error)

    sys.exit(1)


print("\n" + "=" * 70)
print("                    DIAGNOSTIC COMPLETE")
print("=" * 70)