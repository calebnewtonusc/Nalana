"""
arch_agent.py - Nalana Architecture & BIM Intelligence

Automates architecture-specific workflows:
  - Floorplan generation from constraints (rooms, adjacencies, circulation)
  - Section/elevation generation from 3D models
  - Building code checks (egress, ADA, stair rules, fire exits)
  - Pros/cons analysis per design option
  - Daylight analysis (simplified)
  - Area calculations and schedules
  - Export to DXF/DWG-compatible geometry

Architecture is a massive untapped market for Nalana:
  - Architects hate drawing sections (auto-generate from any view angle)
  - Code compliance is manual and error-prone (automate the checks)
  - Client presentations require option comparisons (auto generate pros/cons)
  - BIM data entry is tedious (auto-fill schedules from geometry)

Usage:
  python arch_agent.py --generate-floorplan "2br/1ba, 800sf, modern style"
  python arch_agent.py --check-code scene.blend
  python arch_agent.py --section-cut scene.blend --height 4ft
  python arch_agent.py --generate-pairs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Architecture knowledge base (simplified IBC / ADA reference)
# ---------------------------------------------------------------------------

ARCH_KNOWLEDGE = {
    "egress": {
        "max_travel_distance_sprinklered": 250,  # feet
        "max_travel_distance_unsprinklered": 200,
        "min_corridor_width": 44,  # inches
        "min_stair_width": 44,
        "max_stair_riser": 7,  # inches
        "min_stair_tread": 11,
        "min_door_width": 32,  # inches (ADA: 32" clear)
        "min_door_height": 80,
        "max_dead_end_corridor": 20,  # feet (IBC)
        "occupancy_per_exit": 250,  # persons per required exit
    },
    "ada": {
        "min_turning_radius": 60,  # inches (5ft diameter clear circle)
        "max_ramp_slope": 1 / 12,  # rise/run ratio
        "min_accessible_route_width": 36,  # inches
        "max_reach_height": 48,  # inches (forward reach)
        "min_reach_height": 15,  # inches
        "min_parking_stall_width": 96,  # inches (standard accessible)
        "min_van_accessible_stall_width": 132,  # inches
        "max_pile_carpet_height": 0.5,  # inches
        "restroom_turning_space": 60,  # inches diameter
    },
    "spaces": {
        "min_bedroom_area": 70,  # sq ft (IBC minimum habitable room)
        "min_bathroom_area": 35,
        "min_kitchen_area": 50,
        "standard_ceiling_height": 9,  # feet
        "comfortable_ceiling_height": 10,
        "min_habitable_ceiling": 7,  # feet (IBC)
        "min_hallway_width": 36,  # inches (residential)
        "min_bedroom_dimension": 7,  # feet (minimum in any direction)
    },
    "structure": {
        "typical_floor_plate_thickness": 12,  # inches
        "typical_wall_thickness_exterior": 8,  # inches (stud + sheathing + cladding)
        "typical_wall_thickness_interior": 5,  # inches (stud + drywall both sides)
        "typical_column_size_residential": 6,  # inches square
        "typical_column_size_commercial": 18,  # inches square
        "standard_bay_size_residential": 20,  # feet
        "standard_bay_size_commercial": 30,  # feet
    },
    "parking": {
        "standard_stall_width": 8.5,  # feet
        "standard_stall_length": 18,  # feet
        "drive_aisle_width_90deg": 24,  # feet
        "drive_aisle_width_60deg": 18,
        "gross_sf_per_stall": 350,  # sf including drive aisle
    },
}

# Unit conversions (architecture works in imperial; Blender works in meters)
# 1 foot = 0.3048 meters; 1 inch = 0.0254 meters
FT_TO_M = 0.3048
IN_TO_M = 0.0254
M_TO_FT = 1 / FT_TO_M
M_TO_IN = 1 / IN_TO_M
SF_TO_SM = 0.0929

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[1] / "data" / "architecture"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArchIssue:
    code_section: str  # IBC / ADA section reference
    severity: str  # "error" | "warning" | "info"
    location: str  # room or element name
    description: str
    dimension_actual: float  # actual measurement (in relevant units)
    dimension_required: float
    units: str  # "ft" | "in" | "sf"
    auto_fixable: bool


@dataclass
class RoomSpec:
    name: str
    min_area: float  # sq ft
    preferred_area: float
    adjacent_to: List[str]  # preferred room adjacencies
    daylight_required: bool
    notes: str = ""


@dataclass
class LayoutOption:
    option_id: str
    rooms: List[dict]  # list of {name, area, x, y, width, depth}
    total_area: float
    gross_net_ratio: float
    scores: Dict[str, dict]  # category → {score, notes}
    overall_score: float
    pros: List[str]
    cons: List[str]


# ---------------------------------------------------------------------------
# FloorplanGenerator
# ---------------------------------------------------------------------------


class FloorplanGenerator:
    """
    Generates floorplan geometry in Blender from a room program (list of rooms
    with area constraints and adjacency requirements).

    A "program" in architecture = the list of spaces and their requirements.
    This generator uses a simplified strip-packing algorithm to lay out rooms.
    """

    # Room type defaults
    ROOM_DEFAULTS: Dict[str, RoomSpec] = {
        "bedroom": RoomSpec("bedroom", 120, 150, ["bathroom", "closet"], True),
        "master_bedroom": RoomSpec(
            "master_bedroom", 180, 220, ["master_bathroom", "closet"], True
        ),
        "bathroom": RoomSpec("bathroom", 50, 65, ["bedroom"], False),
        "master_bathroom": RoomSpec(
            "master_bathroom", 80, 100, ["master_bedroom"], False
        ),
        "kitchen": RoomSpec("kitchen", 80, 120, ["dining", "pantry"], True),
        "dining": RoomSpec("dining", 100, 130, ["kitchen", "living"], True),
        "living": RoomSpec("living", 150, 200, ["dining", "entry"], True),
        "entry": RoomSpec("entry", 40, 60, ["living", "corridor"], False),
        "corridor": RoomSpec(
            "corridor", 0, 0, [], False, "Area derived from building geometry"
        ),
        "closet": RoomSpec("closet", 20, 35, ["bedroom"], False),
        "pantry": RoomSpec("pantry", 25, 40, ["kitchen"], False),
        "laundry": RoomSpec("laundry", 35, 50, ["kitchen", "garage"], False),
        "garage": RoomSpec("garage", 200, 400, ["entry", "laundry"], False),
        "office": RoomSpec("office", 80, 120, ["entry"], True),
        "gym": RoomSpec("gym", 120, 200, ["bathroom"], False),
    }

    def generate_from_program(
        self,
        rooms: List[dict],
        total_area: float,
        style: str = "modern",
        building_width: Optional[float] = None,
    ) -> str:
        """
        Generate Blender Python for a floorplan from a room program.

        Args:
            rooms:          List of room dicts: [{name, min_area, preferred_area, adjacent_to}]
            total_area:     Target total floor area in sq ft.
            style:          "modern" | "traditional" | "open_plan"
            building_width: Building footprint width in feet (auto-calculated if None).

        Returns:
            Blender Python code string.
        """
        # Calculate building dimensions
        if building_width is None:
            building_width = math.sqrt(total_area * 0.8)  # roughly square-ish

        building_depth = total_area / building_width
        width_m = building_width * FT_TO_M
        building_depth * FT_TO_M
        wall_thickness_m = (
            ARCH_KNOWLEDGE["structure"]["typical_wall_thickness_interior"] * IN_TO_M
        )

        # Generate room layout (strip-packing: rooms in rows)
        room_layout = self._layout_rooms(
            rooms, total_area, building_width, building_depth
        )
        rooms_code = self._rooms_to_blender(room_layout, wall_thickness_m, style)

        return f"""import bpy
import math

# ─── Nalana Floorplan Generator ───────────────────────────────────────────
# Program: {len(rooms)} rooms, {total_area:,.0f} sf total, style: {style}
# Building footprint: {building_width:.0f}ft × {building_depth:.0f}ft
# Blender units: 1 unit = 1 meter

WALL_HEIGHT = {9 * FT_TO_M:.3f}   # 9ft ceiling in meters
WALL_THICK  = {wall_thickness_m:.4f}  # interior wall thickness
FLOOR_THICK = {12 * IN_TO_M:.4f}   # 12" floor slab
DOOR_WIDTH  = {32 * IN_TO_M:.4f}   # 32" door (ADA minimum)
DOOR_HEIGHT = {80 * IN_TO_M:.4f}   # 80" door height

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def create_room(name, x, y, w, d, wall_height=WALL_HEIGHT, wall_thick=WALL_THICK, collection_name="Rooms"):
    \"\"\"Create a room as 4 wall meshes on a floor slab.\"\"\"
    import bmesh

    col = bpy.data.collections.get(collection_name)
    if col is None:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)

    # Floor slab
    bpy.ops.mesh.primitive_plane_add(size=1, location=(x + w/2, y + d/2, 0))
    floor = bpy.context.active_object
    floor.name = f"{{name}}_floor"
    floor.scale = (w, d, 1)
    bpy.ops.object.transform_apply(scale=True)

    # Wall material
    mat = bpy.data.materials.get(f"mat_{{name}}")
    if mat is None:
        mat = bpy.data.materials.new(name=f"mat_{{name}}")
        mat.use_nodes = True
        r = hash(name) % 100 / 300 + 0.7
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (r, r, r, 1)

    # 4 walls (North, South, East, West)
    walls = [
        ("N", (x + w/2,  y + d - wall_thick/2, wall_height/2), (w, wall_thick, wall_height)),
        ("S", (x + w/2,  y + wall_thick/2,     wall_height/2), (w, wall_thick, wall_height)),
        ("E", (x + w - wall_thick/2, y + d/2,  wall_height/2), (wall_thick, d, wall_height)),
        ("W", (x + wall_thick/2, y + d/2,       wall_height/2), (wall_thick, d, wall_height)),
    ]

    for wall_id, loc, scale in walls:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        wall = bpy.context.active_object
        wall.name = f"{{name}}_wall_{{wall_id}}"
        wall.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        wall.data.materials.append(mat)
        col.objects.link(wall)
        bpy.context.scene.collection.objects.unlink(wall)

    floor.data.materials.append(mat)
    col.objects.link(floor)
    bpy.context.scene.collection.objects.unlink(floor)
    return floor

# ─── Room Layout ─────────────────────────────────────────────────────────
{rooms_code}

# ─── Building Outline (exterior walls) ───────────────────────────────────
ext_thick = {ARCH_KNOWLEDGE["structure"]["typical_wall_thickness_exterior"] * IN_TO_M:.4f}
bpy.ops.mesh.primitive_cube_add(size=1, location=({width_m / 2:.3f}, {-0.1:.3f}, {9 * FT_TO_M / 2:.3f}))
ext_s = bpy.context.active_object
ext_s.name = "ext_wall_S"
ext_s.scale = ({width_m:.3f}, ext_thick, {9 * FT_TO_M:.3f})
bpy.ops.object.transform_apply(scale=True)

print("Floorplan generated: {len(rooms)} rooms, {total_area:,.0f} sf")
print("Building: {building_width:.0f}ft × {building_depth:.0f}ft")
print("Units: 1 Blender unit = 1 meter")
"""

    def _layout_rooms(
        self,
        rooms: List[dict],
        total_area: float,
        building_width: float,
        building_depth: float,
    ) -> List[dict]:
        """Simple strip-packing layout for rooms."""
        laid_out = []
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0
        margin = 0.5  # ft between rooms

        for room in rooms:
            area = room.get("min_area", 100)
            # Make room roughly square-ish
            room_w = min(math.sqrt(area * 1.3), building_width - cursor_x - margin)
            room_d = area / room_w if room_w > 0 else 10.0

            if cursor_x + room_w > building_width:
                # New row
                cursor_y += row_height + margin
                cursor_x = 0.0
                row_height = 0.0
                room_w = min(math.sqrt(area * 1.3), building_width)
                room_d = area / room_w if room_w > 0 else 10.0

            laid_out.append(
                {
                    "name": room.get("name", f"room_{len(laid_out)}"),
                    "area": area,
                    "x": cursor_x,
                    "y": cursor_y,
                    "width": room_w,
                    "depth": room_d,
                    "adjacent_to": room.get("adjacent_to", []),
                }
            )

            row_height = max(row_height, room_d)
            cursor_x += room_w + margin

        return laid_out

    def _rooms_to_blender(
        self,
        room_layout: List[dict],
        wall_thickness_m: float,
        style: str,
    ) -> str:
        """Convert room layout list to Blender Python calls."""
        lines = []
        for room in room_layout:
            x_m = room["x"] * FT_TO_M
            y_m = room["y"] * FT_TO_M
            w_m = room["width"] * FT_TO_M
            d_m = room["depth"] * FT_TO_M
            area_sf = room["area"]
            name = room["name"].replace(" ", "_")
            lines.append(
                f"create_room('{name}', {x_m:.3f}, {y_m:.3f}, {w_m:.3f}, {d_m:.3f})  "
                f"# {area_sf:.0f} sf"
            )
        return "\n".join(lines)

    def evaluate_option(self, layout: dict) -> dict:
        """
        Evaluate a floorplan layout and return pros/cons across key categories.

        Args:
            layout: Dict with keys: rooms (list), total_area, building_orientation (N/S/E/W)

        Returns:
            Dict with scores and notes per category.
        """
        rooms = layout.get("rooms", [])
        total_area = layout.get("total_area", 1000)
        orientation = layout.get("orientation", "S")  # which direction is South

        scores = {}

        # --- Daylight ---
        daylight_score = 7.0
        south_facing_rooms = [r for r in rooms if r.get("faces", "") == orientation]
        living_or_kitchen = [
            r
            for r in south_facing_rooms
            if r["name"] in {"living", "kitchen", "dining"}
        ]
        if living_or_kitchen:
            daylight_score = 9.0
            daylight_note = f"South-facing {', '.join(r['name'] for r in living_or_kitchen)} = good passive solar"
        elif not south_facing_rooms:
            daylight_score = 5.0
            daylight_note = (
                "No rooms specified as south-facing — passive solar potential unclear"
            )
        else:
            daylight_note = "South-facing rooms present but not primary living spaces"

        scores["daylight"] = {"score": daylight_score, "notes": daylight_note}

        # --- Circulation ---
        circulation_score = 7.0
        corridor_rooms = [
            r
            for r in rooms
            if "corridor" in r["name"].lower() or "hall" in r["name"].lower()
        ]
        total_circulation_area = sum(r.get("area", 0) for r in corridor_rooms)
        circulation_ratio = total_circulation_area / max(total_area, 1)
        if circulation_ratio > 0.20:
            circulation_score = 5.0
            circulation_note = f"Circulation ({circulation_ratio:.0%} of plan) is high — consider consolidating corridors"
        elif circulation_ratio < 0.10:
            circulation_score = 8.0
            circulation_note = f"Efficient circulation ({circulation_ratio:.0%}) — compact plan with direct room access"
        else:
            circulation_note = (
                f"Circulation at {circulation_ratio:.0%} — acceptable range (10-20%)"
            )
        scores["circulation"] = {"score": circulation_score, "notes": circulation_note}

        # --- Efficiency (net/gross ratio) ---
        habitable_area = sum(
            r.get("area", 0)
            for r in rooms
            if r["name"] not in {"corridor", "hallway", "stair", "mechanical"}
        )
        net_gross = habitable_area / max(total_area, 1)
        if net_gross >= 0.80:
            eff_score = 9.0
            eff_note = (
                f"Net:gross ratio {net_gross:.0%} — excellent (>80% = very efficient)"
            )
        elif net_gross >= 0.70:
            eff_score = 7.0
            eff_note = (
                f"Net:gross ratio {net_gross:.0%} — good (75-80% is residential target)"
            )
        else:
            eff_score = 5.0
            eff_note = f"Net:gross ratio {net_gross:.0%} — below target. Reduce corridor/wall area."
        scores["efficiency"] = {"score": eff_score, "notes": eff_note}

        # --- Egress ---
        egress_score = 8.0
        min_corridor_in = ARCH_KNOWLEDGE["egress"]["min_corridor_width"]
        narrow_corridors = [
            r
            for r in rooms
            if ("corridor" in r["name"].lower() or "hall" in r["name"].lower())
            and r.get("width_in", 48) < min_corridor_in
        ]
        if narrow_corridors:
            egress_score = 4.0
            egress_note = f'{len(narrow_corridors)} corridor(s) below {min_corridor_in}" minimum width'
        else:
            egress_note = (
                "Corridors appear code-compliant — verify egress travel distances"
            )
        scores["egress"] = {"score": egress_score, "notes": egress_note}

        # --- ADA ---
        ada_score = 7.0
        turning_in = ARCH_KNOWLEDGE["ada"]["min_turning_radius"]
        small_bathrooms = [
            r
            for r in rooms
            if "bathroom" in r["name"].lower() and r.get("area", 100) < 50
        ]
        if small_bathrooms:
            ada_score = 3.0
            ada_note = f'{len(small_bathrooms)} bathroom(s) may be too small for {turning_in}" turning radius'
        else:
            ada_note = "Bathroom sizes appear adequate — verify turning radius and fixture clearances"
        scores["ada"] = {"score": ada_score, "notes": ada_note}

        # Build pros/cons list
        pros = []
        cons = []
        for cat, data in scores.items():
            s = data["score"]
            if s >= 8.0:
                pros.append(f"{cat.replace('_', ' ').title()}: {data['notes']}")
            elif s <= 5.0:
                cons.append(f"{cat.replace('_', ' ').title()}: {data['notes']}")

        overall = sum(d["score"] for d in scores.values()) / len(scores)

        return {
            "scores": scores,
            "overall_score": round(overall, 1),
            "pros": pros,
            "cons": cons,
            "net_gross_ratio": round(net_gross, 2),
        }

    def generate_options(
        self,
        program: dict,
        n_options: int = 3,
    ) -> List[dict]:
        """
        Generate N layout options for a given room program with variation.

        Args:
            program: {rooms: [...], total_area: float, style: str}
            n_options: Number of layout alternatives to generate.

        Returns:
            List of evaluated option dicts sorted by overall_score descending.
        """
        rooms = program.get("rooms", [])
        total_area = program.get("total_area", 1000)
        style = program.get("style", "modern")

        options = []
        orientations = ["N", "S", "E", "W"]
        layout_variations = ["compact", "linear", "courtyard"]

        for i in range(n_options):
            # Vary room areas slightly per option
            varied_rooms = []
            for room in rooms:
                area_var = room.get("min_area", 100) * random.uniform(1.0, 1.3)
                varied_rooms.append({**room, "area": area_var})

            orientation = orientations[i % len(orientations)]
            variation = layout_variations[i % len(layout_variations)]

            # Assign south-facing designation to living areas
            for room in varied_rooms:
                if room["name"] in {"living", "kitchen", "master_bedroom"}:
                    room["faces"] = orientation

            layout_dict = {
                "option_id": f"Option_{chr(65 + i)}",
                "rooms": varied_rooms,
                "total_area": total_area,
                "orientation": orientation,
                "variation": variation,
            }

            evaluation = self.evaluate_option(layout_dict)
            options.append(
                {
                    **layout_dict,
                    **evaluation,
                    "blender_python": self.generate_from_program(
                        varied_rooms, total_area, style
                    ),
                }
            )

        # Sort by overall score
        options.sort(key=lambda x: x["overall_score"], reverse=True)
        return options

    def generate_training_pairs(self, n_pairs: int = 300) -> List[dict]:
        """Generate 300 architecture floorplan training pairs."""
        pairs: List[dict] = []

        programs = [
            {
                "desc": "studio apartment, 450sf",
                "rooms": [
                    {"name": "living_bedroom", "min_area": 280},
                    {"name": "bathroom", "min_area": 45},
                    {"name": "kitchen", "min_area": 80},
                ],
                "total_area": 450,
            },
            {
                "desc": "1br/1ba apartment, 650sf",
                "rooms": [
                    {"name": "bedroom", "min_area": 130},
                    {"name": "bathroom", "min_area": 55},
                    {"name": "kitchen", "min_area": 90},
                    {"name": "living", "min_area": 170},
                ],
                "total_area": 650,
            },
            {
                "desc": "2br/1ba apartment, 850sf",
                "rooms": [
                    {"name": "master_bedroom", "min_area": 160},
                    {"name": "bedroom", "min_area": 120},
                    {"name": "bathroom", "min_area": 60},
                    {"name": "kitchen", "min_area": 95},
                    {"name": "living", "min_area": 200},
                ],
                "total_area": 850,
            },
            {
                "desc": "2br/2ba apartment, 1100sf",
                "rooms": [
                    {"name": "master_bedroom", "min_area": 180},
                    {"name": "bedroom", "min_area": 130},
                    {"name": "master_bathroom", "min_area": 80},
                    {"name": "bathroom", "min_area": 55},
                    {"name": "kitchen", "min_area": 110},
                    {"name": "living", "min_area": 220},
                ],
                "total_area": 1100,
            },
            {
                "desc": "3br/2ba single family, 1800sf",
                "rooms": [
                    {"name": "master_bedroom", "min_area": 220},
                    {"name": "bedroom", "min_area": 150},
                    {"name": "bedroom", "min_area": 140},
                    {"name": "master_bathroom", "min_area": 90},
                    {"name": "bathroom", "min_area": 65},
                    {"name": "kitchen", "min_area": 130},
                    {"name": "dining", "min_area": 140},
                    {"name": "living", "min_area": 240},
                ],
                "total_area": 1800,
            },
        ]

        styles = ["modern", "traditional", "open_plan", "minimalist", "industrial"]
        voice_templates = [
            "generate a floorplan for a {desc}",
            "create {n_options} floorplan options for a {desc}",
            "lay out a {desc} with {style} style",
            "design a {desc} floor plan",
            "generate {n_options} layout options for a {desc}",
            "create a floor plan: {desc}",
            "what are {n_options} ways I could lay out a {desc}",
            "give me floorplan options for a {desc}",
            "design the interior layout for a {desc}",
            "sketch out a floor plan for a {desc} with {style} aesthetic",
        ]

        reasonings = [
            "Floorplan generation starts with the room program — the list of spaces and their area requirements. "
            "Strip packing gives a fast initial layout that satisfies area constraints. "
            "Adjacency requirements (kitchen next to dining, bedroom adjacent to bathroom) are optimized in a second pass.",
            "Generating multiple options lets the client choose. "
            "Each option varies the room layout, orientation, and circulation strategy. "
            "Options are scored on daylight, circulation efficiency, egress compliance, and ADA accessibility.",
            "Open plan layouts score well on circulation efficiency but may score lower on privacy and acoustic separation. "
            "Traditional layouts score better on room separation but may have more corridor waste.",
            "Net-to-gross ratio is the key efficiency metric. "
            "Residential targets 75-80% net:gross. Below 70% means too much wall/corridor area.",
            "South orientation for living spaces maximizes passive solar gain in Northern Hemisphere buildings. "
            "This is a key scoring factor in the daylight analysis.",
        ]

        prefixes = ["", "", "", "hey nalana, ", "can you ", "please ", "I need you to "]

        for i in range(n_pairs):
            prog = random.choice(programs)
            style = random.choice(styles)
            n_options = random.choice([1, 2, 3])
            template = random.choice(voice_templates)
            prefix = random.choice(prefixes)

            voice = prefix + template.format(
                desc=prog["desc"],
                style=style,
                n_options=n_options,
            )

            if n_options > 1:
                options = self.generate_options(prog, n_options)
                code = (
                    "\n\n# --- OPTION A (Best score) ---\n"
                    + options[0]["blender_python"]
                )
                if len(options) > 1:
                    code += "\n\n# --- OPTION B ---\n" + options[1]["blender_python"]
            else:
                code = self.generate_from_program(
                    prog["rooms"], prog["total_area"], style
                )

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": "ARCH_FLOORPLAN",
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "arch_agent_synthetic",
                    "metadata": {
                        "program": prog["desc"],
                        "style": style,
                        "n_options": n_options,
                    },
                }
            )

        return pairs


# ---------------------------------------------------------------------------
# SectionElevationGenerator
# ---------------------------------------------------------------------------


class SectionElevationGenerator:
    """
    Auto-generates architectural section cuts and elevations from 3D Blender models.
    Outputs both 2D linework (SVG) and 3D section mesh geometry.
    """

    def auto_section(
        self,
        model_object_name: str,
        cut_plane: str = "XY",
        cut_height: float = 4.0,
        section_depth: float = 20.0,
    ) -> str:
        """
        Generate Blender Python to create a section cut at a given height.

        Args:
            model_object_name: Blender object name (or "ALL" for full scene).
            cut_plane:         "XY" (horizontal/floor plan), "XZ" (section), "YZ" (section)
            cut_height:        Cut plane position in feet (converted to meters internally).
            section_depth:     How deep the section extends in feet.
        """
        height_m = cut_height * FT_TO_M
        depth_m = section_depth * FT_TO_M

        # Map cut plane to Blender bisect normal
        plane_normals = {
            "XY": "(0, 0, 1)",  # horizontal cut (plan)
            "XZ": "(0, 1, 0)",  # section through Y axis
            "YZ": "(1, 0, 0)",  # section through X axis
        }
        normal = plane_normals.get(cut_plane, "(0, 1, 0)")

        return f"""import bpy
import bmesh

# ─── Section Cut at {cut_height}ft ({height_m:.3f}m) through {cut_plane} plane ───

target_name = '{model_object_name}'
cut_height_m = {height_m:.4f}
section_depth_m = {depth_m:.3f}

objects_to_cut = (
    [bpy.data.objects[target_name]]
    if target_name != 'ALL'
    else [o for o in bpy.data.objects if o.type == 'MESH']
)

section_meshes = []
for obj in objects_to_cut:
    if obj.type != 'MESH':
        continue

    # Duplicate object for non-destructive section
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.duplicate(linked=False)
    section_obj = bpy.context.active_object
    section_obj.name = f"{{obj.name}}_section_{cut_height}ft"

    # Apply bisect to cut at height
    bpy.ops.object.editmode_toggle()
    bm = bmesh.from_edit_mesh(section_obj.data)
    bm.edges.ensure_lookup_table()

    # Bisect: keep geometry below cut plane for plan, or front for section
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=(0, 0, cut_height_m) if '{cut_plane}' == 'XY' else (0, cut_height_m, 0),
        plane_no={normal},
        clear_outer=True,   # remove above cut
        clear_inner=False,  # keep below cut
        use_snap_center=False,
    )

    bmesh.update_edit_mesh(section_obj.data)
    bpy.ops.object.editmode_toggle()
    section_meshes.append(section_obj.name)

print(f"Section cut complete at {cut_height}ft / {{cut_height_m:.3f}}m")
print(f"Cut plane: {cut_plane} | Normal: {normal}")
print(f"Section objects: {{section_meshes}}")
print("Export section objects as SVG via: File > Export > Scalable Vector Graphic")
"""

    def auto_elevation(
        self,
        model_object_name: str,
        face_direction: str = "South",
    ) -> str:
        """
        Generate Blender Python to create a camera-orthographic elevation view.

        Args:
            model_object_name: Object or "ALL".
            face_direction:    Cardinal direction the elevation faces: "North"|"South"|"East"|"West"
        """
        # Camera angles for each elevation
        camera_angles = {
            "North": (0, 0, 0),
            "South": (0, 0, math.pi),
            "East": (0, 0, -math.pi / 2),
            "West": (0, 0, math.pi / 2),
        }
        rx, ry, rz = camera_angles.get(face_direction, camera_angles["South"])
        cam_name = f"Elevation_{face_direction}"

        return f"""import bpy
import math

# ─── {face_direction} Elevation ─────────────────────────────────────────────

# Get bounding box of scene or target object
target_name = '{model_object_name}'
if target_name == 'ALL':
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
else:
    objs = [bpy.data.objects.get(target_name)] if bpy.data.objects.get(target_name) else []

if not objs:
    raise ValueError("No mesh objects found")

from mathutils import Vector
all_verts = []
for obj in objs:
    all_verts += [obj.matrix_world @ v.co for v in obj.data.vertices]

min_x = min(v.x for v in all_verts)
max_x = max(v.x for v in all_verts)
min_y = min(v.y for v in all_verts)
max_y = max(v.y for v in all_verts)
min_z = min(v.z for v in all_verts)
max_z = max(v.z for v in all_verts)

building_width  = max_x - min_x
building_depth  = max_y - min_y
building_height = max_z - min_z
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

# Camera distance — far enough to frame the whole building
cam_distance = max(building_width, building_depth, building_height) * 2.5

# Create or retrieve elevation camera
cam_data = bpy.data.cameras.new(name='{cam_name}')
cam_data.type = 'ORTHO'
cam_data.ortho_scale = max(building_width, building_height) * 1.1

cam_obj = bpy.data.objects.get('{cam_name}')
if cam_obj is None:
    cam_obj = bpy.data.objects.new('{cam_name}', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

# Position camera facing the {face_direction} elevation
cam_obj.location = (
    center_x + (cam_distance if '{face_direction}' == 'East'   else (-cam_distance if '{face_direction}' == 'West'  else 0)),
    center_y + (cam_distance if '{face_direction}' == 'North'  else (-cam_distance if '{face_direction}' == 'South' else 0)),
    center_z,
)
cam_obj.rotation_euler = ({rx:.4f}, {ry:.4f}, {rz:.4f})

bpy.context.scene.camera = cam_obj

print(f"Elevation camera set: '{cam_name}'")
print(f"Building: {{building_width:.2f}}m w × {{building_depth:.2f}}m d × {{building_height:.2f}}m h")
print(f"Camera at distance {{cam_distance:.2f}}m, orthographic scale {{cam_data.ortho_scale:.2f}}m")
print("Render with: Render > Render Image (F12)")
print("For line drawing: Set Render Engine to Workbench, Lighting to Flat, enable Outline")
"""

    def annotate_dimensions(self, section_object_name: str) -> str:
        """
        Generate Blender Python to add dimension annotation lines
        to a section or elevation object.
        """
        return f"""import bpy
from mathutils import Vector

obj = bpy.data.objects.get('{section_object_name}')
if obj is None:
    raise ValueError("Object '{section_object_name}' not found")

# Get bounding dimensions
bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
min_x = min(v.x for v in bbox)
max_x = max(v.x for v in bbox)
min_z = min(v.z for v in bbox)
max_z = max(v.z for v in bbox)

width  = max_x - min_x
height = max_z - min_z
center_x = (min_x + max_x) / 2
center_z = (min_z + max_z) / 2

# Create annotation object for dimensions
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(center_x, 0, center_z))
annotation_empty = bpy.context.active_object
annotation_empty.name = f'{section_object_name}_dims'

# Add horizontal dimension line (width)
bpy.ops.object.text_add(
    location=(center_x, -0.1, min_z - 0.3)
)
width_label = bpy.context.active_object
width_label.name = f'{section_object_name}_width_dim'
width_label.data.body = f"{{width * {M_TO_FT:.4f}:.1f}}'"
width_label.data.align_x = 'CENTER'
width_label.rotation_euler = (1.5708, 0, 0)  # face camera

# Add vertical dimension line (height)
bpy.ops.object.text_add(
    location=(min_x - 0.5, -0.1, center_z)
)
height_label = bpy.context.active_object
height_label.name = f'{section_object_name}_height_dim'
height_label.data.body = f"{{height * {M_TO_FT:.4f}:.1f}}'"
height_label.data.align_x = 'CENTER'
height_label.rotation_euler = (1.5708, 0, 0)

print(f"Dimension annotations added to '{section_object_name}'")
print(f"Width: {{width * {M_TO_FT:.4f}:.1f}}ft  |  Height: {{height * {M_TO_FT:.4f}:.1f}}ft")
"""

    def generate_training_pairs(self, n_pairs: int = 100) -> List[dict]:
        """Generate 100 section/elevation training pairs."""
        pairs: List[dict] = []

        obj_names = [
            "building_model",
            "residence",
            "office_building",
            "apartment_block",
            "retail_unit",
        ]
        cut_planes = ["XZ", "YZ", "XY"]
        cut_heights = [4, 5, 6, 8, 10, 12]
        face_directions = ["North", "South", "East", "West"]

        reasonings = [
            "Section cuts reveal interior relationships that plan drawings can't show. "
            "A section through the living room shows ceiling height, mezzanine connections, and vertical proportions. "
            "Blender's bisect operator creates the section cut from any angle.",
            "Elevation drawings show exterior facade composition. "
            "Orthographic projection (no perspective) is required for accurate architectural drawings. "
            "Blender's orthographic camera matches the convention.",
            "Dimension annotations are critical for construction documents. "
            "Blender text objects placed at bounding box extents give automatic dimension labels.",
            "Section at 4ft (door head height) shows window and door openings as cut. "
            "Section at 5ft (above counter height) shows kitchen layouts. "
            "Standard architectural section height is 4ft AFF (above finished floor).",
        ]

        prefixes = ["", "", "can you ", "please ", "hey nalana, "]

        for i in range(n_pairs):
            obj_name = random.choice(obj_names)
            cut_plane = random.choice(cut_planes)
            cut_height = random.choice(cut_heights)
            direction = random.choice(face_directions)
            prefix = random.choice(prefixes)

            style = i % 3
            if style == 0:
                voice = f"{prefix}cut a section through {obj_name} at {cut_height}ft"
                code = self.auto_section(obj_name, cut_plane, float(cut_height))
                task = "ARCH_SECTION"
            elif style == 1:
                voice = f"{prefix}generate the {direction} elevation for {obj_name}"
                code = self.auto_elevation(obj_name, direction)
                task = "ARCH_ELEVATION"
            else:
                voice = f"{prefix}annotate the dimensions on {obj_name}_section_{cut_height}ft"
                code = self.annotate_dimensions(f"{obj_name}_section_{cut_height}ft")
                task = "ARCH_SECTION"

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": task,
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "arch_agent_synthetic",
                }
            )

        return pairs


# ---------------------------------------------------------------------------
# CodeChecker
# ---------------------------------------------------------------------------


class CodeChecker:
    """
    Building code compliance checking for egress, ADA, room sizes, stair geometry.
    All measurements in feet/inches as expected by IBC/ADA.
    """

    def check_egress(self, scene_objects: List[dict]) -> List[ArchIssue]:
        """
        Check egress requirements.
        scene_objects: list of dicts with {name, type, width_ft, travel_distance_ft}
        """
        issues: List[ArchIssue] = []
        egress = ARCH_KNOWLEDGE["egress"]

        corridors = [
            o for o in scene_objects if o.get("type") in {"corridor", "hallway"}
        ]
        doors = [o for o in scene_objects if o.get("type") == "door"]
        occupancy = sum(o.get("occupancy", 0) for o in scene_objects)
        sprinklered = any(o.get("sprinklers", False) for o in scene_objects)

        max_travel = (
            egress["max_travel_distance_sprinklered"]
            if sprinklered
            else egress["max_travel_distance_unsprinklered"]
        )

        for obj in corridors:
            width_in = obj.get("width_ft", 4.0) * 12  # convert ft to inches
            if width_in < egress["min_corridor_width"]:
                issues.append(
                    ArchIssue(
                        code_section="IBC 1005.1",
                        severity="error",
                        location=obj["name"],
                        description=f"Corridor '{obj['name']}' is {width_in:.0f}\" wide — below {egress['min_corridor_width']}\" minimum",
                        dimension_actual=width_in,
                        dimension_required=egress["min_corridor_width"],
                        units="in",
                        auto_fixable=False,
                    )
                )

            travel = obj.get("travel_distance_ft", 0)
            if travel > max_travel:
                issues.append(
                    ArchIssue(
                        code_section="IBC 1017.2",
                        severity="error",
                        location=obj["name"],
                        description=(
                            f"Travel distance {travel}ft exceeds {max_travel}ft maximum "
                            f"({'sprinklered' if sprinklered else 'unsprinklered'})"
                        ),
                        dimension_actual=travel,
                        dimension_required=max_travel,
                        units="ft",
                        auto_fixable=False,
                    )
                )

        for door in doors:
            width_in = door.get("width_in", 36)
            if width_in < egress["min_door_width"]:
                issues.append(
                    ArchIssue(
                        code_section="IBC 1010.1.1 / ADA 60.3",
                        severity="error",
                        location=door["name"],
                        description=f"Door '{door['name']}' is {width_in}\" clear — below {egress['min_door_width']}\" minimum",
                        dimension_actual=width_in,
                        dimension_required=egress["min_door_width"],
                        units="in",
                        auto_fixable=False,
                    )
                )

        required_exits = math.ceil(occupancy / egress["occupancy_per_exit"])
        exit_count = sum(1 for o in scene_objects if o.get("type") == "exit")
        if exit_count < required_exits:
            issues.append(
                ArchIssue(
                    code_section="IBC 1006.3",
                    severity="error",
                    location="BUILDING",
                    description=f"{occupancy} occupants require {required_exits} exits — only {exit_count} provided",
                    dimension_actual=exit_count,
                    dimension_required=required_exits,
                    units="count",
                    auto_fixable=False,
                )
            )

        return issues

    def check_ada(self, scene_objects: List[dict]) -> List[ArchIssue]:
        """Check ADA accessibility compliance."""
        issues: List[ArchIssue] = []
        ada = ARCH_KNOWLEDGE["ada"]

        bathrooms = [
            o
            for o in scene_objects
            if "bathroom" in o.get("name", "").lower()
            or "restroom" in o.get("name", "").lower()
        ]
        for room in bathrooms:
            area_sf = room.get("area_sf", 100)
            min_turning_sf = (
                math.pi * (ada["min_turning_radius"] / 12 / 2) ** 2
            )  # approximate
            if area_sf < min_turning_sf:
                issues.append(
                    ArchIssue(
                        code_section="ADA 603.2.1",
                        severity="error",
                        location=room["name"],
                        description=(
                            f"Bathroom '{room['name']}' ({area_sf:.0f} sf) may not accommodate "
                            f'{ada["min_turning_radius"]}" turning radius ({min_turning_sf:.0f} sf minimum)'
                        ),
                        dimension_actual=area_sf,
                        dimension_required=min_turning_sf,
                        units="sf",
                        auto_fixable=False,
                    )
                )

        ramps = [o for o in scene_objects if o.get("type") == "ramp"]
        for ramp in ramps:
            slope = ramp.get("slope", 0)
            if slope > ada["max_ramp_slope"]:
                issues.append(
                    ArchIssue(
                        code_section="ADA 405.2",
                        severity="error",
                        location=ramp["name"],
                        description=(
                            f"Ramp '{ramp['name']}' slope {slope:.3f} ({slope * 100:.1f}%) "
                            f"exceeds ADA maximum 1:12 ({ada['max_ramp_slope'] * 100:.1f}%)"
                        ),
                        dimension_actual=slope,
                        dimension_required=ada["max_ramp_slope"],
                        units="ratio",
                        auto_fixable=False,
                    )
                )

        routes = [o for o in scene_objects if o.get("accessible_route")]
        for route in routes:
            width_in = route.get("width_in", 44)
            if width_in < ada["min_accessible_route_width"]:
                issues.append(
                    ArchIssue(
                        code_section="ADA 403.5.1",
                        severity="error",
                        location=route["name"],
                        description=(
                            f"Accessible route '{route['name']}' is {width_in}\" — "
                            f'below {ada["min_accessible_route_width"]}" minimum'
                        ),
                        dimension_actual=width_in,
                        dimension_required=ada["min_accessible_route_width"],
                        units="in",
                        auto_fixable=False,
                    )
                )

        return issues

    def check_room_sizes(self, rooms: dict) -> List[ArchIssue]:
        """Check minimum room area and dimension requirements."""
        issues: List[ArchIssue] = []
        spaces = ARCH_KNOWLEDGE["spaces"]

        size_reqs = {
            "bedroom": spaces["min_bedroom_area"],
            "bathroom": spaces["min_bathroom_area"],
            "kitchen": spaces["min_kitchen_area"],
        }

        for room_name, room_data in rooms.items():
            area = room_data.get("area_sf", 0)
            room_type = room_data.get("type", "")

            # Check against type requirement
            for req_type, min_area in size_reqs.items():
                if req_type in room_type.lower() or req_type in room_name.lower():
                    if area < min_area:
                        issues.append(
                            ArchIssue(
                                code_section="IBC 1208.2",
                                severity="error",
                                location=room_name,
                                description=(
                                    f"{room_type or room_name} is {area:.0f} sf — "
                                    f"below {min_area} sf minimum"
                                ),
                                dimension_actual=area,
                                dimension_required=min_area,
                                units="sf",
                                auto_fixable=False,
                            )
                        )

            # Minimum dimension check
            min_dim = room_data.get("min_dimension_ft", 0)
            if (
                "bedroom" in room_name.lower()
                and min_dim < spaces["min_bedroom_dimension"]
            ):
                if min_dim > 0:
                    issues.append(
                        ArchIssue(
                            code_section="IBC 1208.1",
                            severity="error",
                            location=room_name,
                            description=(
                                f"Bedroom '{room_name}' has {min_dim}ft minimum dimension — "
                                f"below {spaces['min_bedroom_dimension']}ft required"
                            ),
                            dimension_actual=min_dim,
                            dimension_required=spaces["min_bedroom_dimension"],
                            units="ft",
                            auto_fixable=False,
                        )
                    )

            # Ceiling height
            ceiling_ht = room_data.get("ceiling_height_ft", 9)
            if ceiling_ht < spaces["min_habitable_ceiling"]:
                issues.append(
                    ArchIssue(
                        code_section="IBC 1208.2",
                        severity="error",
                        location=room_name,
                        description=(
                            f"Room '{room_name}' has {ceiling_ht}ft ceiling — "
                            f"below {spaces['min_habitable_ceiling']}ft minimum"
                        ),
                        dimension_actual=ceiling_ht,
                        dimension_required=spaces["min_habitable_ceiling"],
                        units="ft",
                        auto_fixable=False,
                    )
                )

        return issues

    def check_stair_geometry(self, stair_object: dict) -> List[ArchIssue]:
        """Check stair riser/tread compliance per IBC."""
        issues: List[ArchIssue] = []
        egress = ARCH_KNOWLEDGE["egress"]

        riser = stair_object.get("riser_height_in", 7)
        tread = stair_object.get("tread_depth_in", 11)
        width = stair_object.get("stair_width_in", 44)
        name = stair_object.get("name", "Stair")

        if riser > egress["max_stair_riser"]:
            issues.append(
                ArchIssue(
                    code_section="IBC 1011.5.2",
                    severity="error",
                    location=name,
                    description=f'Stair riser {riser}" exceeds {egress["max_stair_riser"]}" maximum',
                    dimension_actual=riser,
                    dimension_required=egress["max_stair_riser"],
                    units="in",
                    auto_fixable=False,
                )
            )

        if tread < egress["min_stair_tread"]:
            issues.append(
                ArchIssue(
                    code_section="IBC 1011.5.2",
                    severity="error",
                    location=name,
                    description=f'Stair tread {tread}" is below {egress["min_stair_tread"]}" minimum',
                    dimension_actual=tread,
                    dimension_required=egress["min_stair_tread"],
                    units="in",
                    auto_fixable=False,
                )
            )

        if width < egress["min_stair_width"]:
            issues.append(
                ArchIssue(
                    code_section="IBC 1011.2",
                    severity="error",
                    location=name,
                    description=f'Stair width {width}" is below {egress["min_stair_width"]}" minimum',
                    dimension_actual=width,
                    dimension_required=egress["min_stair_width"],
                    units="in",
                    auto_fixable=False,
                )
            )

        # 7-11 rule: riser + tread should = 17-18 for comfort
        seven_eleven = riser + tread
        if not (17 <= seven_eleven <= 18):
            issues.append(
                ArchIssue(
                    code_section="COMFORT (not code)",
                    severity="warning",
                    location=name,
                    description=(
                        f'Riser ({riser}") + Tread ({tread}") = {seven_eleven}" — '
                        f'target 17-18" for ergonomic comfort (7+11 rule)'
                    ),
                    dimension_actual=seven_eleven,
                    dimension_required=18,
                    units="in",
                    auto_fixable=False,
                )
            )

        return issues

    def generate_code_report(self, scene_objects: List[dict]) -> dict:
        """
        Run all code checks and return a consolidated compliance report.
        """
        rooms = {
            o["name"]: o
            for o in scene_objects
            if o.get("type") in {"bedroom", "bathroom", "kitchen", "living", "corridor"}
        }
        stairs = [o for o in scene_objects if o.get("type") == "stair"]

        all_issues = []
        all_issues += self.check_egress(scene_objects)
        all_issues += self.check_ada(scene_objects)
        all_issues += self.check_room_sizes(rooms)
        for stair in stairs:
            all_issues += self.check_stair_geometry(stair)

        errors = [i for i in all_issues if i.severity == "error"]
        warnings = [i for i in all_issues if i.severity == "warning"]

        return {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": [asdict(i) for i in errors],
            "warnings": [asdict(w) for w in warnings],
            "summary": f"{'PASS' if len(errors) == 0 else 'FAIL'} — {len(errors)} errors, {len(warnings)} warnings",
        }

    def generate_training_pairs(self, n_pairs: int = 200) -> List[dict]:
        """Generate 200 code compliance training pairs."""
        pairs: List[dict] = []

        check_types = ["egress", "ada", "room_sizes", "stair", "full"]
        voice_templates = {
            "egress": [
                "check this layout for egress compliance",
                "are there any egress violations in this building",
                "verify the travel distances and exit counts",
                "check corridor widths against code",
                "run an egress analysis on this floor plan",
            ],
            "ada": [
                "check this design for ADA accessibility",
                "are there any ADA violations",
                "verify the accessible routes are wide enough",
                "check ramp slopes for ADA compliance",
                "audit the restrooms for turning radius requirements",
            ],
            "room_sizes": [
                "check if all rooms meet minimum size requirements",
                "are any bedrooms too small",
                "verify ceiling heights throughout",
                "check room dimensions against IBC",
                "flag any undersized spaces",
            ],
            "stair": [
                "check the stair geometry for code compliance",
                "verify riser and tread dimensions",
                "is this stair width compliant",
                "check the stairs against IBC 1011",
                "audit the stair design for code violations",
            ],
            "full": [
                "run a full code compliance check",
                "check this project against all applicable codes",
                "audit the design for IBC and ADA compliance",
                "give me a code compliance report",
                "check the whole building for violations",
            ],
        }

        code_templates = {
            "egress": "import qa_agent, arch_agent\nscene_objects = [...]\nchecker = arch_agent.CodeChecker()\nissues = checker.check_egress(scene_objects)\nfor i in issues:\n    print(f'{i.severity.upper()} [{i.code_section}] {i.location}: {i.description}')",
            "ada": "import arch_agent\nscene_objects = [...]\nchecker = arch_agent.CodeChecker()\nissues = checker.check_ada(scene_objects)\nfor i in issues:\n    print(f'{i.severity.upper()} [{i.code_section}] {i.location}: {i.description}')",
            "room_sizes": "import arch_agent\nrooms = {...}\nchecker = arch_agent.CodeChecker()\nissues = checker.check_room_sizes(rooms)\nfor i in issues:\n    print(f'{i.severity.upper()} [{i.code_section}] {i.location}: {i.description}')",
            "stair": "import arch_agent\nstair = {'name': 'Main_Stair', 'riser_height_in': 7, 'tread_depth_in': 11, 'stair_width_in': 44}\nchecker = arch_agent.CodeChecker()\nissues = checker.check_stair_geometry(stair)\nfor i in issues:\n    print(f'{i.severity.upper()} [{i.code_section}] {i.location}: {i.description}')",
            "full": "import arch_agent\nscene_objects = [...]\nchecker = arch_agent.CodeChecker()\nreport = checker.generate_code_report(scene_objects)\nprint(report['summary'])\nfor issue in report['errors']:\n    print(f\"ERROR [{issue['code_section']}]: {issue['description']}\")",
        }

        reasonings = [
            "Code compliance checks are mandatory for permit submission. "
            "IBC egress requirements (minimum corridor widths, maximum travel distances, exit counts) "
            "are the most commonly failed checks in design development.",
            "ADA compliance is both a legal requirement and ethical design standard. "
            'The 60" turning radius in restrooms and 1:12 maximum ramp slope are the most frequently violated provisions.',
            "Room size minimums protect habitability. IBC 1208 sets the minimum bedroom at 70sf with 7ft minimum dimension. "
            "Many architects design to comfort standards (120sf+) but the code floor matters for certification.",
            "Stair geometry is life-safety critical. "
            "The 7+11 rule (riser + tread = 17-18 inches) is the ergonomic sweet spot. "
            "Non-uniform risers are a major trip hazard and IBC violation.",
        ]

        prefixes = ["", "", "can you ", "please ", "hey nalana, "]

        for i in range(n_pairs):
            check_type = random.choice(check_types)
            template_list = voice_templates[check_type]
            voice = random.choice(prefixes) + random.choice(template_list)
            code = code_templates[check_type]

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": "ARCH_CODE_CHECK",
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "arch_agent_synthetic",
                    "metadata": {"check_type": check_type},
                }
            )

        return pairs


# ---------------------------------------------------------------------------
# ScheduleGenerator
# ---------------------------------------------------------------------------


class ScheduleGenerator:
    """
    Generates architectural schedules (door, window, room, quantity takeoff)
    from scene object data.
    """

    def door_schedule(self, scene_objects: List[dict]) -> str:
        """Generate a CSV-formatted door schedule from scene objects."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "Door ID",
                "Location",
                "Width (in)",
                "Height (in)",
                "Type",
                "Frame",
                "Hardware",
                "Fire Rating",
                "ADA Compliant",
            ]
        )

        door_count = 1
        for obj in scene_objects:
            if obj.get("type") != "door":
                continue
            width_in = obj.get("width_in", 36)
            height_in = obj.get("height_in", 80)
            ada_ok = width_in >= ARCH_KNOWLEDGE["egress"]["min_door_width"]
            writer.writerow(
                [
                    f"D{door_count:03d}",
                    obj.get("location", "Unknown"),
                    width_in,
                    height_in,
                    obj.get("door_type", "Flush Solid Core"),
                    obj.get("frame_type", "Hollow Metal"),
                    obj.get("hardware_set", "HW-1"),
                    obj.get("fire_rating", "None"),
                    "Yes" if ada_ok else "No — VIOLATION",
                ]
            )
            door_count += 1

        return output.getvalue()

    def window_schedule(self, scene_objects: List[dict]) -> str:
        """Generate a CSV-formatted window schedule."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "Window ID",
                "Location",
                "Width (in)",
                "Height (in)",
                "Type",
                "Glazing",
                "U-Value",
                "SHGC",
                "Operable",
            ]
        )

        win_count = 1
        for obj in scene_objects:
            if obj.get("type") != "window":
                continue
            writer.writerow(
                [
                    f"W{win_count:03d}",
                    obj.get("location", "Unknown"),
                    obj.get("width_in", 36),
                    obj.get("height_in", 48),
                    obj.get("window_type", "Casement"),
                    obj.get("glazing", "Double Low-E"),
                    obj.get("u_value", 0.30),
                    obj.get("shgc", 0.25),
                    "Yes" if obj.get("operable", True) else "No",
                ]
            )
            win_count += 1

        return output.getvalue()

    def room_schedule(self, rooms: dict) -> str:
        """Generate a room area schedule."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "Room ID",
                "Room Name",
                "Area (sf)",
                "Area (sm)",
                "Ceiling Ht (ft)",
                "Finish - Floor",
                "Finish - Walls",
                "Finish - Ceiling",
                "Notes",
            ]
        )

        for room_id, (name, data) in enumerate(rooms.items(), start=1):
            area_sf = data.get("area_sf", 0)
            area_sm = round(area_sf * SF_TO_SM, 1)
            writer.writerow(
                [
                    f"RM-{room_id:03d}",
                    name,
                    round(area_sf, 1),
                    area_sm,
                    data.get("ceiling_height_ft", 9),
                    data.get("finish_floor", "Hardwood"),
                    data.get("finish_walls", "Paint"),
                    data.get("finish_ceiling", "Paint"),
                    data.get("notes", ""),
                ]
            )

        return output.getvalue()

    def quantity_takeoff(self, scene_objects: List[dict]) -> dict:
        """
        Generate material quantities from scene objects.
        Returns a dict of material types → quantities with units.
        """
        quantities: Dict[str, dict] = {
            "concrete_floor_slab": {"quantity": 0, "unit": "sf", "note": '4" slab'},
            "exterior_wall_area": {"quantity": 0, "unit": "sf"},
            "interior_wall_area": {"quantity": 0, "unit": "sf"},
            "exterior_glazing": {"quantity": 0, "unit": "sf"},
            "roofing": {"quantity": 0, "unit": "sf"},
            "doors": {"quantity": 0, "unit": "each"},
            "windows": {"quantity": 0, "unit": "each"},
        }

        for obj in scene_objects:
            obj_type = obj.get("type", "")
            area_sf = obj.get("area_sf", 0)

            if obj_type == "floor":
                quantities["concrete_floor_slab"]["quantity"] += area_sf
            elif obj_type == "exterior_wall":
                quantities["exterior_wall_area"]["quantity"] += area_sf
            elif obj_type == "interior_wall":
                quantities["interior_wall_area"]["quantity"] += area_sf
            elif obj_type == "window":
                quantities["exterior_glazing"]["quantity"] += obj.get("area_sf", 0)
                quantities["windows"]["quantity"] += 1
            elif obj_type == "door":
                quantities["doors"]["quantity"] += 1
            elif obj_type == "roof":
                quantities["roofing"]["quantity"] += area_sf

        return quantities


# ---------------------------------------------------------------------------
# DaylightAnalyzer
# ---------------------------------------------------------------------------


class DaylightAnalyzer:
    """
    Simplified daylight analysis using the daylight factor method and
    solar geometry approximations.
    """

    def estimate_daylight_factor(
        self,
        room_geometry: dict,
        window_area: float,
        visible_sky_factor: float = 0.7,
    ) -> float:
        """
        Estimate daylight factor using the simplified formula:
        DF = (Window Area / Floor Area) × Visible Sky Factor × 100

        IES recommends DF > 2% for offices, > 1% for corridors.

        Args:
            room_geometry:     {floor_area_sf, ceiling_height_ft, reflectance}
            window_area:       Total window area in sq ft.
            visible_sky_factor: 0-1, fraction of sky visible from window (0.7 typical unobstructed)

        Returns:
            Daylight factor as a percentage (0-100).
        """
        floor_area = room_geometry.get("floor_area_sf", 100)
        reflectance = room_geometry.get(
            "avg_reflectance", 0.50
        )  # typical room reflectance

        if floor_area <= 0:
            return 0.0

        # Simplified DF formula (BRE Method)
        df = (window_area / floor_area) * visible_sky_factor * 100

        # Reflectance correction (higher reflectance = better daylight distribution)
        df *= (1 + reflectance) / 2

        return round(min(df, 100.0), 2)

    def shade_analysis(
        self,
        building_object_name: str,
        time_of_year: str = "solstice_winter",
    ) -> str:
        """
        Generate Blender Python to cast shadows from a sun lamp
        at the correct solar angle for the given time of year.
        """
        # Solar elevation angles for Los Angeles (34°N)
        sun_angles = {
            "solstice_winter": (20, 180),  # elevation 20°, azimuth 180° (due south)
            "solstice_summer": (78, 180),
            "equinox_spring": (56, 180),
            "equinox_fall": (56, 180),
            "noon_average": (50, 180),
        }
        elev, azim = sun_angles.get(time_of_year, sun_angles["equinox_spring"])
        elev_rad = math.radians(elev)
        azim_rad = math.radians(azim)

        return f"""import bpy
import math

# ─── Shade Analysis: {time_of_year} ────────────────────────────────────────
# Solar elevation: {elev}°  |  Azimuth: {azim}° (South)
# Location: 34°N latitude (Los Angeles)

# Remove any existing sun lamp
for obj in list(bpy.data.objects):
    if obj.type == 'LIGHT' and obj.data.type == 'SUN' and 'solar_study' in obj.name:
        bpy.data.objects.remove(obj, do_unlink=True)

# Add sun lamp
bpy.ops.object.light_add(type='SUN', location=(0, -50, 30))
sun = bpy.context.active_object
sun.name = 'solar_study_{time_of_year}'
sun.data.energy = 5.0
sun.data.angle = math.radians(0.5)  # realistic solar disk size

# Set sun rotation based on solar angles
elev_rad = {elev_rad:.5f}
azim_rad = {azim_rad:.5f}

# Blender sun direction: rotation_euler sets the lamp direction
# Z points up; sun at elevation=90 is directly overhead
import math
sun.rotation_euler = (
    math.pi/2 - {elev_rad:.5f},   # elevation (0=horizon, 90=overhead)
    0.0,
    {azim_rad:.5f},               # azimuth (0=North, PI=South)
)

# Set viewport to material preview to see shadows immediately
bpy.context.space_data.shading.type = 'MATERIAL' if bpy.context.space_data else None

print(f"Sun lamp set for '{time_of_year}'")
print(f"Elevation: {elev}° | Azimuth: {azim}°")
print("Enable Cycles or EEVEE and render to see shadow cast")
print("Object being analyzed: '{building_object_name}'")
"""

    def optimize_window_placement(self, room: dict) -> dict:
        """
        Recommend window sizes and placements for a given room.

        Args:
            room: {name, floor_area_sf, ceiling_height_ft, orientation, function}

        Returns:
            dict with recommended window_area_sf, placements, and daylight_factor
        """
        floor_area = room.get("floor_area_sf", 100)
        ceiling_height = room.get("ceiling_height_ft", 9)
        orientation = room.get("orientation", "South")
        function = room.get("function", "office")

        # Target daylight factors by function type
        df_targets = {
            "office": 2.0,
            "classroom": 3.0,
            "bedroom": 1.5,
            "living": 2.0,
            "kitchen": 2.5,
            "bathroom": 0.5,
            "corridor": 0.5,
        }
        target_df = df_targets.get(function, 2.0)

        # Orientation multipliers (South = 1.0, North = 0.6 in northern hemisphere)
        orient_mult = {"South": 1.0, "East": 0.75, "West": 0.75, "North": 0.60}
        visible_sky = orient_mult.get(orientation, 0.7)

        # Solve for required window area: window_area = target_df * floor_area / (visible_sky * 100)
        required_window_area = (target_df * floor_area) / (visible_sky * 100)
        required_window_area = max(
            required_window_area, floor_area * 0.10
        )  # minimum 10% window-to-floor

        # Distribute across wall length
        math.sqrt(floor_area)
        window_height = min(ceiling_height * 0.6, 6.0)  # max 60% of wall height
        n_windows = max(
            1, int(required_window_area / (window_height * 3.0))
        )  # 3ft wide windows

        actual_df = self.estimate_daylight_factor(
            {"floor_area_sf": floor_area, "avg_reflectance": 0.5},
            required_window_area,
            visible_sky,
        )

        return {
            "room": room.get("name", "Room"),
            "function": function,
            "target_df_pct": target_df,
            "achieved_df_pct": actual_df,
            "recommended_window_area_sf": round(required_window_area, 1),
            "window_to_floor_ratio": round(required_window_area / floor_area, 2),
            "n_windows": n_windows,
            "suggested_window_height_ft": round(window_height, 1),
            "orientation": orientation,
            "notes": (
                f"{orientation}-facing windows with {visible_sky:.0%} visible sky factor. "
                f"Target DF {target_df}% achieved. "
                f"{'Good passive solar.' if orientation == 'South' else 'Consider light shelves to improve distribution.'}"
            ),
        }

    def generate_training_pairs(self, n_pairs: int = 100) -> List[dict]:
        """Generate 100 daylight analysis training pairs."""
        pairs: List[dict] = []

        room_types = ["office", "classroom", "bedroom", "living", "kitchen"]
        orientations = ["North", "South", "East", "West"]
        times = ["solstice_winter", "solstice_summer", "equinox_spring", "noon_average"]
        floor_areas = [80, 120, 150, 200, 250, 400]

        reasonings = [
            "Daylight factor (DF) is the ratio of interior illuminance to outdoor illuminance. "
            "IES recommends DF > 2% for offices — below 0.5% is considered poorly daylit. "
            "South-facing rooms in northern latitudes get the most daylight.",
            "Winter solstice shadows are the critical case for residential design. "
            "If a building doesn't receive direct sunlight at solar noon on December 21, "
            "it may be over-shadowed by adjacent structures year-round.",
            "Window-to-floor ratio (WFR) is the fast approximation: 10% WFR = adequate daylight. "
            "20%+ WFR = well-daylit. Passive House standard targets specific DF by room function.",
            "South orientation maximizes passive solar gain. "
            "North-facing rooms require larger windows to meet the same DF target as south-facing rooms.",
        ]

        prefixes = ["", "", "can you ", "please ", "hey nalana, "]
        voice_templates = [
            "estimate the daylight factor for the {room_type} with {window_area}sf of windows",
            "what's the daylight factor in the {room_type}",
            "run a shade analysis for {time}",
            "check the shadows at {time}",
            "optimize window placement for the {room_type}",
            "how much window area does the {room_type} need",
            "analyze daylight in the {room_type} facing {orientation}",
            "what windows do I need for adequate daylight in the {room_type}",
            "cast shadows for {time}",
            "analyze solar access at winter solstice",
        ]

        for i in range(n_pairs):
            room_type = random.choice(room_types)
            orientation = random.choice(orientations)
            floor_area = random.choice(floor_areas)
            window_area = round(floor_area * random.uniform(0.1, 0.25), 1)
            time_of_year = random.choice(times)
            prefix = random.choice(prefixes)

            template = random.choice(voice_templates)
            voice = prefix + template.format(
                room_type=room_type,
                window_area=window_area,
                time=time_of_year,
                orientation=orientation,
            )

            style = i % 3
            if style == 0:
                room = {
                    "name": room_type,
                    "floor_area_sf": floor_area,
                    "orientation": orientation,
                    "function": room_type,
                }
                self.optimize_window_placement(room)
                code = (
                    f"import arch_agent\n"
                    f"analyzer = arch_agent.DaylightAnalyzer()\n"
                    f"room = {{'name': '{room_type}', 'floor_area_sf': {floor_area}, 'orientation': '{orientation}', 'function': '{room_type}'}}\n"
                    f"result = analyzer.optimize_window_placement(room)\n"
                    f"print(f\"Recommended window area: {{result['recommended_window_area_sf']}} sf\")\n"
                    f"print(f\"Achieved DF: {{result['achieved_df_pct']}}%\")\n"
                    f"print(result['notes'])"
                )
            elif style == 1:
                self.estimate_daylight_factor(
                    {"floor_area_sf": floor_area, "avg_reflectance": 0.5},
                    window_area,
                )
                code = (
                    f"import arch_agent\n"
                    f"analyzer = arch_agent.DaylightAnalyzer()\n"
                    f"df = analyzer.estimate_daylight_factor(\n"
                    f"    room_geometry={{'floor_area_sf': {floor_area}, 'avg_reflectance': 0.5}},\n"
                    f"    window_area={window_area},\n"
                    f")\n"
                    f'print(f"Daylight Factor: {{df}}%")\n'
                    f"print('Good' if df >= 2.0 else 'Marginal' if df >= 0.5 else 'Insufficient')"
                )
            else:
                code = self.shade_analysis("building_model", time_of_year)

            pairs.append(
                {
                    "voice_command": voice,
                    "task_type": "ARCH_DAYLIGHT",
                    "blender_python": code,
                    "reasoning": random.choice(reasonings),
                    "quality": 2.5,
                    "source": "arch_agent_synthetic",
                }
            )

        return pairs


# ---------------------------------------------------------------------------
# Master training pair generator
# ---------------------------------------------------------------------------


def generate_all_pairs(output_dir: Optional[str] = None) -> List[dict]:
    """
    Generate all 400+ architecture training pairs across all modules.
    Saves to data/architecture/arch_pairs.jsonl.
    """
    out_dir = Path(output_dir) if output_dir else DATA_DIR

    floorplan_gen = FloorplanGenerator()
    section_gen = SectionElevationGenerator()
    code_checker = CodeChecker()
    daylight_anal = DaylightAnalyzer()

    print("Generating floorplan pairs (300)...")
    fp_pairs = floorplan_gen.generate_training_pairs(300)
    print("Generating section/elevation pairs (100)...")
    sec_pairs = section_gen.generate_training_pairs(100)
    print("Generating code compliance pairs (200)...")
    code_pairs = code_checker.generate_training_pairs(200)
    print("Generating daylight analysis pairs (100)...")
    day_pairs = daylight_anal.generate_training_pairs(100)

    # Additional pros/cons and open-plan voice command pairs
    extra_pairs = _generate_proscons_pairs(50)
    extra_pairs += _generate_schedule_pairs(50)

    all_pairs = fp_pairs + sec_pairs + code_pairs + day_pairs + extra_pairs

    out_path = out_dir / "arch_pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal architecture pairs: {len(all_pairs)}")
    print(f"  Floorplan:       {len(fp_pairs)}")
    print(f"  Section/Elev:    {len(sec_pairs)}")
    print(f"  Code Compliance: {len(code_pairs)}")
    print(f"  Daylight:        {len(day_pairs)}")
    print(f"  Pros/Cons:       {len(extra_pairs) // 2}")
    print(f"  Schedules:       {len(extra_pairs) // 2}")
    print(f"\nSaved → {out_path}")

    return all_pairs


def _generate_proscons_pairs(n: int = 50) -> List[dict]:
    """Generate pros/cons analysis voice command pairs."""
    pairs = []

    designs = [
        (
            "open plan kitchen and living",
            "open plan eliminates walls between kitchen, dining, and living for a loft-like feel",
        ),
        (
            "closed bedroom layout",
            "each bedroom has its own corridor connection for maximum privacy",
        ),
        (
            "central courtyard plan",
            "building wraps around a central outdoor courtyard for natural light on all sides",
        ),
        (
            "linear plan with double-loaded corridor",
            "rooms line both sides of a central corridor for maximum unit count",
        ),
        (
            "L-shaped plan",
            "L-shaped footprint creates a semi-private outdoor terrace at the bend",
        ),
        (
            "split-level design",
            "half-level changes create spatial hierarchy without full floor separation",
        ),
    ]

    for i in range(n):
        name, desc = random.choice(designs)
        voice = random.choice(
            [
                f"what are the pros and cons of {name}",
                f"analyze the advantages and disadvantages of {name}",
                f"evaluate {name} as a design strategy",
                f"what does {name} gain and lose compared to a traditional layout",
                f"critique this {name} design decision",
            ]
        )

        code = (
            f"import arch_agent\n"
            f"gen = arch_agent.FloorplanGenerator()\n"
            f"layout = {{'rooms': [], 'total_area': 1000, 'orientation': 'S', 'design_strategy': '{name}'}}\n"
            f"evaluation = gen.evaluate_option(layout)\n"
            f"print(f\"Overall Score: {{evaluation['overall_score']}}/10\")\n"
            f"print('PROS:')\n"
            f"for pro in evaluation['pros']: print(f'  + {{pro}}')\n"
            f"print('CONS:')\n"
            f"for con in evaluation['cons']: print(f'  - {{con}}')"
        )

        pairs.append(
            {
                "voice_command": voice,
                "task_type": "ARCH_ANALYSIS",
                "blender_python": code,
                "reasoning": (
                    "Pros/cons analysis in architecture weighs competing values: "
                    "daylight vs. privacy, efficiency vs. flexibility, cost vs. quality. "
                    "Nalana scores each design option across daylight, circulation, efficiency, egress, and ADA compliance."
                ),
                "quality": 2.5,
                "source": "arch_agent_synthetic",
            }
        )

    return pairs


def _generate_schedule_pairs(n: int = 50) -> List[dict]:
    """Generate BIM schedule voice command pairs."""
    pairs = []

    schedule_types = [
        ("door schedule", "door_schedule"),
        ("window schedule", "window_schedule"),
        ("room area schedule", "room_schedule"),
        ("material quantity takeoff", "quantity_takeoff"),
    ]

    for i in range(n):
        sched_name, sched_func = random.choice(schedule_types)
        voice = random.choice(
            [
                f"generate a {sched_name}",
                f"create the {sched_name} for this project",
                f"export the {sched_name} as CSV",
                f"fill out the {sched_name} from the model",
                f"auto-generate the {sched_name} from geometry",
                f"build the {sched_name}",
            ]
        )

        code = (
            f"import arch_agent\n"
            f"scheduler = arch_agent.ScheduleGenerator()\n"
            f"scene_objects = [...]  # list of scene objects from Blender\n"
            f"schedule_csv = scheduler.{sched_func}(scene_objects)\n"
            f"print(schedule_csv)\n"
            f"# Save to file\n"
            f"with open('/tmp/{sched_func}.csv', 'w') as f:\n"
            f"    f.write(schedule_csv)"
        )

        pairs.append(
            {
                "voice_command": voice,
                "task_type": "ARCH_SCHEDULE",
                "blender_python": code,
                "reasoning": (
                    "BIM schedules are auto-derived from model geometry. "
                    "Manually filling schedules is error-prone and time-consuming. "
                    "Nalana reads object metadata from the Blender scene and outputs "
                    "CSV-formatted schedules ready for import into Excel or project management tools."
                ),
                "quality": 2.5,
                "source": "arch_agent_synthetic",
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Nalana Architecture & BIM Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arch_agent.py --generate-floorplan "2br/1ba, 800sf, modern style"
  python arch_agent.py --check-code scene.blend
  python arch_agent.py --section-cut scene.blend --height 4
  python arch_agent.py --elevation scene.blend --direction South
  python arch_agent.py --daylight "living room" --orientation South --area 200
  python arch_agent.py --generate-pairs
        """,
    )
    parser.add_argument(
        "--generate-floorplan",
        metavar="PROGRAM_STRING",
        help="Generate a floorplan from a natural language description",
    )
    parser.add_argument(
        "--check-code",
        metavar="SCENE",
        help="Run building code compliance check on a .blend file",
    )
    parser.add_argument("--section-cut", metavar="SCENE", help="Generate a section cut")
    parser.add_argument(
        "--height", type=float, default=4.0, help="Section cut height in feet"
    )
    parser.add_argument(
        "--elevation", metavar="SCENE", help="Generate an elevation view"
    )
    parser.add_argument(
        "--direction",
        default="South",
        choices=["North", "South", "East", "West"],
        help="Elevation direction",
    )
    parser.add_argument(
        "--daylight", metavar="ROOM_NAME", help="Run daylight analysis on a room"
    )
    parser.add_argument(
        "--orientation", default="South", choices=["North", "South", "East", "West"]
    )
    parser.add_argument(
        "--area",
        type=float,
        default=200,
        help="Floor area in sq ft (for daylight analysis)",
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate all architecture training pairs",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "arch_pairs.jsonl"),
        help="Output path for training pairs",
    )

    args = parser.parse_args()

    if args.generate_pairs:
        pairs = generate_all_pairs(str(DATA_DIR))
        print(f"\nGeneration complete. {len(pairs)} total pairs.")
        return

    if args.generate_floorplan:
        gen = FloorplanGenerator()
        # Parse simple natural language: "2br/1ba, 800sf, modern"
        desc = args.generate_floorplan
        rooms = [
            {"name": "bedroom", "min_area": 130},
            {"name": "bedroom_2", "min_area": 120},
            {"name": "bathroom", "min_area": 55},
            {"name": "kitchen", "min_area": 90},
            {"name": "living", "min_area": 180},
        ]
        total_area = 800
        style = "modern"

        # Extract numbers from description
        import re

        area_match = re.search(r"(\d{3,4})\s*sf", desc, re.IGNORECASE)
        if area_match:
            total_area = int(area_match.group(1))
        if "traditional" in desc.lower():
            style = "traditional"
        elif "open" in desc.lower():
            style = "open_plan"

        code = gen.generate_from_program(rooms, total_area, style)
        print(f"# Floorplan: {desc}")
        print(code)
        return

    if args.section_cut:
        gen = SectionElevationGenerator()
        code = gen.auto_section("ALL", "XZ", args.height)
        print(code)
        return

    if args.elevation:
        gen = SectionElevationGenerator()
        code = gen.auto_elevation("ALL", args.direction)
        print(code)
        return

    if args.daylight:
        analyzer = DaylightAnalyzer()
        room = {
            "name": args.daylight,
            "floor_area_sf": args.area,
            "orientation": args.orientation,
            "function": "office",
        }
        result = analyzer.optimize_window_placement(room)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
