import bpy
import csv
import os

filepath = os.path.expanduser("sphere_positions.csv")

with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "x", "y", "z"])

    for obj in bpy.data.objects:
        if obj.name.startswith("Sphere"):
            loc = obj.location
            scale = 25.0
            writer.writerow([obj.name, loc.x * scale, loc.y * scale, loc.z * scale])

print(f"Exported to {filepath}")
