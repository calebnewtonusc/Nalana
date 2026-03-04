"""
physics_kb.py — Nalana Physics Knowledge Base

Central reference for all physically-based rendering constants, material properties,
and simulation parameters. Imported by physics_sim.py, prompts.py, and training pipeline.
GPU synthesis prompts for synthesizing genius physics knowledge from expert texts.
"""

from __future__ import annotations
from typing import Any

# ─── 1. INDEX OF REFRACTION TABLE ─────────────────────────────────────────────
# Real IOR values (n) for dielectrics.
# Metals use complex IOR as (n, k) tuples where k is the extinction coefficient.
# All values at 589.3 nm (sodium D line) unless otherwise noted.

IOR_TABLE: dict[str, float | tuple[float, float]] = {
    # ── Gases & Atmosphere ───────────────────────────────────────────────────
    "vacuum": 1.0,
    "air": 1.0003,
    "air_dry": 1.000293,
    "carbon_dioxide": 1.00045,
    "nitrogen": 1.000298,
    # ── Liquids ─────────────────────────────────────────────────────────────
    "water": 1.333,
    "water_20c": 1.3325,
    "saltwater": 1.341,
    "ice": 1.309,
    "ethanol": 1.361,
    "acetone": 1.359,
    "benzene": 1.501,
    "glycerin": 1.473,
    "olive_oil": 1.467,
    "honey": 1.49,
    "milk": 1.35,
    "blood": 1.40,
    "mercury_liquid": 1.73,
    # ── Biological ──────────────────────────────────────────────────────────
    "skin": 1.4,
    "skin_epidermis": 1.45,
    "cornea": 1.376,
    "lens_crystalline": 1.413,
    "vitreous_humor": 1.336,
    "fat": 1.461,
    "muscle": 1.41,
    "bone": 1.556,
    "tooth_enamel": 1.632,
    "fingernail": 1.54,
    # ── Glasses & Silicates ─────────────────────────────────────────────────
    "glass_crown": 1.523,
    "glass_flint": 1.617,
    "glass_borosilicate": 1.47,
    "glass_soda_lime": 1.51,
    "glass_optical": 1.5,
    "quartz_crystalline": 1.544,
    "quartz_fused": 1.458,
    "obsidian": 1.489,
    "pyrex": 1.474,
    # ── Gems & Minerals ─────────────────────────────────────────────────────
    "diamond": 2.417,
    "ruby": 1.77,
    "sapphire": 1.77,
    "emerald": 1.58,
    "opal": 1.45,
    "amethyst": 1.544,
    "topaz": 1.619,
    "pearl": 1.53,
    "garnet": 1.79,
    "jade": 1.66,
    "amber": 1.539,
    "tourmaline": 1.624,
    "tanzanite": 1.694,
    "alexandrite": 1.746,
    "zircon": 1.925,
    "moissanite": 2.648,
    # ── Polymers & Plastics ─────────────────────────────────────────────────
    "plastic_pet": 1.575,
    "plastic_abs": 1.504,
    "plastic_polycarbonate": 1.586,
    "plastic_acrylic": 1.491,
    "plastic_nylon": 1.53,
    "plastic_hdpe": 1.54,
    "plastic_polypropylene": 1.49,
    "plastic_pvc": 1.539,
    "rubber_natural": 1.52,
    "rubber_silicone": 1.404,
    "rubber_neoprene": 1.558,
    # ── Organic & Natural ────────────────────────────────────────────────────
    "wax_paraffin": 1.446,
    "wax_beeswax": 1.441,
    "wood_pine": 1.53,
    "wood_oak": 1.55,
    "cotton": 1.53,
    "silk": 1.558,
    "wool": 1.539,
    "leather": 1.41,
    "paper": 1.5,
    # ── Ceramics & Stone ────────────────────────────────────────────────────
    "concrete": 1.533,
    "ceramic_porcelain": 1.504,
    "marble": 1.486,
    "granite": 1.544,
    "limestone": 1.566,
    "sandstone": 1.52,
    # ── Metals (complex IOR as (n, k)) ──────────────────────────────────────
    # These are approximate values at ~589 nm visible light.
    # Actual IOR is strongly wavelength-dependent for metals.
    "gold": (0.27, 2.97),
    "silver": (0.14, 3.98),
    "copper": (0.62, 2.63),
    "iron": (2.87, 3.08),
    "aluminum": (1.10, 6.95),
    "chromium": (3.18, 3.31),
    "nickel": (1.97, 3.71),
    "titanium": (2.16, 2.93),
    "tungsten": (3.48, 2.86),
    "platinum": (2.33, 4.26),
    "zinc": (1.93, 1.56),
    "cobalt": (2.25, 4.07),
}

# ─── 2. PBR MATERIAL PRESETS ──────────────────────────────────────────────────
# Values match Principled BSDF node parameters in Blender.
# All colors are linear RGB [0..1] tuples unless noted.
# subsurface_radius: per-channel (R, G, B) scatter distances in meters.

PBR_PRESETS: dict[str, dict[str, Any]] = {
    # ── Precious Metals ──────────────────────────────────────────────────────
    "gold": {
        "base_color": (1.0, 0.766, 0.336, 1.0),
        "metallic": 1.0,
        "roughness": 0.1,
        "ior": 0.27,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "silver": {
        "base_color": (0.972, 0.960, 0.915, 1.0),
        "metallic": 1.0,
        "roughness": 0.05,
        "ior": 0.14,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "copper": {
        "base_color": (0.955, 0.637, 0.538, 1.0),
        "metallic": 1.0,
        "roughness": 0.15,
        "ior": 0.62,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.2,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Industrial Metals ────────────────────────────────────────────────────
    "brushed_aluminum": {
        "base_color": (0.913, 0.921, 0.925, 1.0),
        "metallic": 1.0,
        "roughness": 0.35,
        "ior": 1.10,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.85,
        "anisotropic_rotation": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "chrome": {
        "base_color": (0.95, 0.95, 0.95, 1.0),
        "metallic": 1.0,
        "roughness": 0.02,
        "ior": 3.18,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "rusty_iron": {
        "base_color": (0.366, 0.149, 0.058, 1.0),
        "metallic": 0.3,
        "roughness": 0.85,
        "ior": 2.87,
        "specular": 0.1,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "carbon_fiber": {
        "base_color": (0.02, 0.02, 0.02, 1.0),
        "metallic": 0.0,
        "roughness": 0.2,
        "ior": 1.5,
        "specular": 0.8,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.9,
        "anisotropic_rotation": 0.0,
        "clearcoat": 0.5,
        "clearcoat_roughness": 0.05,
        "sheen": 0.0,
    },
    # ── Wood ─────────────────────────────────────────────────────────────────
    "aged_wood": {
        "base_color": (0.318, 0.196, 0.102, 1.0),
        "metallic": 0.0,
        "roughness": 0.7,
        "ior": 1.53,
        "specular": 0.05,
        "transmission": 0.0,
        "subsurface": 0.02,
        "subsurface_radius": (0.8, 0.5, 0.3),
        "emission_strength": 0.0,
        "anisotropic": 0.3,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Stone ────────────────────────────────────────────────────────────────
    "marble_white": {
        "base_color": (0.95, 0.93, 0.90, 1.0),
        "metallic": 0.0,
        "roughness": 0.15,
        "ior": 1.486,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.05,
        "subsurface_radius": (0.4, 0.35, 0.3),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.3,
        "clearcoat_roughness": 0.1,
        "sheen": 0.0,
    },
    "marble_black": {
        "base_color": (0.04, 0.04, 0.05, 1.0),
        "metallic": 0.0,
        "roughness": 0.08,
        "ior": 1.486,
        "specular": 0.6,
        "transmission": 0.0,
        "subsurface": 0.01,
        "subsurface_radius": (0.1, 0.1, 0.12),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.5,
        "clearcoat_roughness": 0.05,
        "sheen": 0.0,
    },
    "granite": {
        "base_color": (0.35, 0.30, 0.30, 1.0),
        "metallic": 0.0,
        "roughness": 0.65,
        "ior": 1.544,
        "specular": 0.3,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "concrete": {
        "base_color": (0.55, 0.53, 0.50, 1.0),
        "metallic": 0.0,
        "roughness": 0.9,
        "ior": 1.533,
        "specular": 0.02,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Rubber ───────────────────────────────────────────────────────────────
    "rubber_black": {
        "base_color": (0.02, 0.02, 0.02, 1.0),
        "metallic": 0.0,
        "roughness": 0.75,
        "ior": 1.52,
        "specular": 0.02,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "rubber_red": {
        "base_color": (0.6, 0.015, 0.01, 1.0),
        "metallic": 0.0,
        "roughness": 0.7,
        "ior": 1.52,
        "specular": 0.03,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Plastics ─────────────────────────────────────────────────────────────
    "plastic_glossy": {
        "base_color": (0.2, 0.4, 0.8, 1.0),
        "metallic": 0.0,
        "roughness": 0.05,
        "ior": 1.504,
        "specular": 0.5,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "plastic_matte": {
        "base_color": (0.5, 0.5, 0.5, 1.0),
        "metallic": 0.0,
        "roughness": 0.9,
        "ior": 1.504,
        "specular": 0.1,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Glass ────────────────────────────────────────────────────────────────
    "glass_clear": {
        "base_color": (1.0, 1.0, 1.0, 1.0),
        "metallic": 0.0,
        "roughness": 0.0,
        "ior": 1.5,
        "specular": 0.5,
        "transmission": 1.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "glass_frosted": {
        "base_color": (0.95, 0.95, 0.97, 1.0),
        "metallic": 0.0,
        "roughness": 0.35,
        "ior": 1.5,
        "specular": 0.5,
        "transmission": 0.9,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "glass_tinted_blue": {
        "base_color": (0.2, 0.6, 1.0, 1.0),
        "metallic": 0.0,
        "roughness": 0.02,
        "ior": 1.5,
        "specular": 0.5,
        "transmission": 0.85,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    # ── Liquids ──────────────────────────────────────────────────────────────
    "water_clear": {
        "base_color": (0.85, 0.95, 1.0, 1.0),
        "metallic": 0.0,
        "roughness": 0.0,
        "ior": 1.333,
        "specular": 0.5,
        "transmission": 1.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "ice": {
        "base_color": (0.88, 0.95, 1.0, 1.0),
        "metallic": 0.0,
        "roughness": 0.05,
        "ior": 1.309,
        "specular": 0.5,
        "transmission": 0.8,
        "subsurface": 0.1,
        "subsurface_radius": (0.5, 0.6, 0.8),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.6,
        "clearcoat_roughness": 0.02,
        "sheen": 0.0,
    },
    # ── Biological / Organic ─────────────────────────────────────────────────
    "skin_caucasian": {
        "base_color": (0.847, 0.635, 0.498, 1.0),
        "metallic": 0.0,
        "roughness": 0.55,
        "ior": 1.4,
        "specular": 0.3,
        "transmission": 0.0,
        "subsurface": 0.4,
        "subsurface_radius": (1.0, 0.2, 0.1),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.05,
    },
    "skin_melanated": {
        "base_color": (0.317, 0.173, 0.090, 1.0),
        "metallic": 0.0,
        "roughness": 0.6,
        "ior": 1.4,
        "specular": 0.25,
        "transmission": 0.0,
        "subsurface": 0.3,
        "subsurface_radius": (0.6, 0.12, 0.06),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.04,
    },
    "leaves": {
        "base_color": (0.10, 0.38, 0.06, 1.0),
        "metallic": 0.0,
        "roughness": 0.65,
        "ior": 1.45,
        "specular": 0.2,
        "transmission": 0.15,
        "subsurface": 0.2,
        "subsurface_radius": (0.3, 0.8, 0.1),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.1,
        "clearcoat_roughness": 0.5,
        "sheen": 0.3,
    },
    "wax": {
        "base_color": (1.0, 0.95, 0.75, 1.0),
        "metallic": 0.0,
        "roughness": 0.3,
        "ior": 1.446,
        "specular": 0.4,
        "transmission": 0.0,
        "subsurface": 0.5,
        "subsurface_radius": (1.2, 0.9, 0.5),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.2,
        "clearcoat_roughness": 0.1,
        "sheen": 0.0,
    },
    # ── Ceramics ─────────────────────────────────────────────────────────────
    "ceramic_white": {
        "base_color": (0.95, 0.94, 0.93, 1.0),
        "metallic": 0.0,
        "roughness": 0.7,
        "ior": 1.504,
        "specular": 0.1,
        "transmission": 0.0,
        "subsurface": 0.05,
        "subsurface_radius": (0.2, 0.2, 0.2),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "ceramic_glazed": {
        "base_color": (0.9, 0.3, 0.15, 1.0),
        "metallic": 0.0,
        "roughness": 0.05,
        "ior": 1.504,
        "specular": 0.6,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.8,
        "clearcoat_roughness": 0.02,
        "sheen": 0.0,
    },
    # ── Textiles ─────────────────────────────────────────────────────────────
    "fabric_cotton": {
        "base_color": (0.9, 0.9, 0.88, 1.0),
        "metallic": 0.0,
        "roughness": 0.95,
        "ior": 1.53,
        "specular": 0.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.9,
        "sheen_tint": 1.0,
    },
    "fabric_velvet": {
        "base_color": (0.3, 0.05, 0.4, 1.0),
        "metallic": 0.0,
        "roughness": 1.0,
        "ior": 1.539,
        "specular": 0.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 1.0,
        "sheen_tint": 1.0,
    },
    # ── Leather ──────────────────────────────────────────────────────────────
    "leather_brown": {
        "base_color": (0.35, 0.17, 0.07, 1.0),
        "metallic": 0.0,
        "roughness": 0.65,
        "ior": 1.41,
        "specular": 0.15,
        "transmission": 0.0,
        "subsurface": 0.05,
        "subsurface_radius": (0.5, 0.3, 0.15),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.05,
        "clearcoat_roughness": 0.4,
        "sheen": 0.1,
    },
    "leather_black": {
        "base_color": (0.04, 0.04, 0.04, 1.0),
        "metallic": 0.0,
        "roughness": 0.5,
        "ior": 1.41,
        "specular": 0.2,
        "transmission": 0.0,
        "subsurface": 0.02,
        "subsurface_radius": (0.2, 0.15, 0.08),
        "emission_strength": 0.0,
        "anisotropic": 0.0,
        "clearcoat": 0.1,
        "clearcoat_roughness": 0.3,
        "sheen": 0.05,
    },
    # ── Emissive Materials ───────────────────────────────────────────────────
    "holographic_foil": {
        "base_color": (0.9, 0.9, 1.0, 1.0),
        "metallic": 1.0,
        "roughness": 0.03,
        "ior": 1.0,
        "specular": 1.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 0.0,
        "anisotropic": 0.95,
        "anisotropic_rotation": 0.5,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "neon_emission": {
        "base_color": (0.0, 1.0, 0.8, 1.0),
        "metallic": 0.0,
        "roughness": 0.5,
        "ior": 1.5,
        "specular": 0.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 8.0,
        "emission_color": (0.0, 1.0, 0.8, 1.0),
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "fire_emission": {
        "base_color": (1.0, 0.3, 0.0, 1.0),
        "metallic": 0.0,
        "roughness": 1.0,
        "ior": 1.0,
        "specular": 0.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 15.0,
        "emission_color": (1.0, 0.3, 0.0, 1.0),
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
    "lava": {
        "base_color": (0.05, 0.01, 0.0, 1.0),
        "metallic": 0.0,
        "roughness": 0.95,
        "ior": 1.5,
        "specular": 0.05,
        "transmission": 0.0,
        "subsurface": 0.0,
        "subsurface_radius": (1.0, 1.0, 1.0),
        "emission_strength": 5.0,
        "emission_color": (1.0, 0.15, 0.0, 1.0),
        "anisotropic": 0.0,
        "clearcoat": 0.0,
        "sheen": 0.0,
    },
}

# ─── 3. SIMULATION PRESETS ────────────────────────────────────────────────────
# Values tuned for Blender's physics solvers. Units are SI unless noted.

SIMULATION_PRESETS: dict[str, dict[str, Any]] = {
    "cloth": {
        "silk": {
            "density": 0.006,  # kg/m²
            "bending_stiffness": 0.01,  # N·m
            "shear_stiffness": 0.05,  # N/m
            "tension_stiffness": 15.0,  # N/m
            "damping": 0.01,
            "air_resistance": 1.0,  # kg/(m·s)
            "friction": 0.2,
            "mass": 0.3,  # kg
            "quality_steps": 8,
            "collision_distance": 0.001,  # m
        },
        "denim": {
            "density": 0.38,  # kg/m²
            "bending_stiffness": 2.5,
            "shear_stiffness": 5.0,
            "tension_stiffness": 40.0,
            "damping": 0.1,
            "air_resistance": 0.3,
            "friction": 0.6,
            "mass": 1.2,
            "quality_steps": 10,
            "collision_distance": 0.002,
        },
        "leather": {
            "density": 0.9,  # kg/m²
            "bending_stiffness": 8.0,
            "shear_stiffness": 12.0,
            "tension_stiffness": 80.0,
            "damping": 0.2,
            "air_resistance": 0.1,
            "friction": 0.8,
            "mass": 2.5,
            "quality_steps": 12,
            "collision_distance": 0.003,
        },
    },
    "fluid": {
        "water": {
            "viscosity": 0.001002,  # Pa·s at 20°C
            "density": 998.2,  # kg/m³ at 20°C
            "surface_tension": 0.0728,  # N/m at 20°C
            "restitution": 0.0,
            "resolution": 64,  # domain subdivision
            "time_scale": 1.0,
            "use_spray": True,
            "use_foam": True,
            "use_bubbles": True,
            "compressibility": 0.005,
        },
        "honey": {
            "viscosity": 2.0,  # Pa·s (varies widely 2-10)
            "density": 1400.0,  # kg/m³
            "surface_tension": 0.064,  # N/m
            "restitution": 0.0,
            "resolution": 48,
            "time_scale": 0.3,
            "use_spray": False,
            "use_foam": False,
            "use_bubbles": False,
            "compressibility": 0.001,
        },
        "lava": {
            "viscosity": 100.0,  # Pa·s (basaltic lava)
            "density": 2700.0,  # kg/m³
            "surface_tension": 0.35,  # N/m (estimated)
            "restitution": 0.0,
            "resolution": 32,
            "time_scale": 0.1,
            "use_spray": False,
            "use_foam": False,
            "use_bubbles": True,
            "compressibility": 0.0001,
        },
    },
    "rigid_body": {
        "rubber": {
            "friction": 0.8,
            "restitution": 0.7,
            "linear_damping": 0.05,
            "angular_damping": 0.1,
            "mass": 1.0,  # kg (per-object scale)
            "collision_shape": "CONVEX_HULL",
            "use_deactivation": True,
            "deactivation_linear_velocity": 0.4,
            "deactivation_angular_velocity": 0.5,
        },
        "metal": {
            "friction": 0.4,
            "restitution": 0.1,
            "linear_damping": 0.01,
            "angular_damping": 0.02,
            "mass": 7.0,
            "collision_shape": "MESH",
            "use_deactivation": True,
            "deactivation_linear_velocity": 0.1,
            "deactivation_angular_velocity": 0.1,
        },
        "wood": {
            "friction": 0.6,
            "restitution": 0.35,
            "linear_damping": 0.03,
            "angular_damping": 0.08,
            "mass": 0.6,
            "collision_shape": "CONVEX_HULL",
            "use_deactivation": True,
            "deactivation_linear_velocity": 0.3,
            "deactivation_angular_velocity": 0.4,
        },
    },
    "soft_body": {
        "jelly": {
            "goal_strength": 0.1,
            "goal_damping": 0.5,
            "edge_stiffness": 0.05,
            "edge_damping": 0.2,
            "mass": 1.0,
            "pull": 0.2,
            "push": 0.4,
            "self_collision": True,
            "friction": 0.5,
            "use_goal": True,
        },
        "foam": {
            "goal_strength": 0.3,
            "goal_damping": 0.3,
            "edge_stiffness": 0.3,
            "edge_damping": 0.1,
            "mass": 0.05,
            "pull": 0.5,
            "push": 0.8,
            "self_collision": False,
            "friction": 0.3,
            "use_goal": True,
        },
        "muscle": {
            "goal_strength": 0.8,
            "goal_damping": 0.1,
            "edge_stiffness": 0.7,
            "edge_damping": 0.05,
            "mass": 1.1,
            "pull": 0.9,
            "push": 0.95,
            "self_collision": False,
            "friction": 0.7,
            "use_goal": True,
        },
    },
    "smoke": {
        "campfire": {
            "density": 0.5,
            "temperature": 800.0,  # K above ambient
            "temperature_max": 1200.0,  # K
            "velocity_strength": 1.5,  # upward buoyancy
            "vorticity": 0.2,
            "dissolve_speed": 25,  # frames
            "use_dissolve": True,
            "turbulence_strength": 0.5,
            "resolution": 64,
        },
        "explosion": {
            "density": 3.0,
            "temperature": 3000.0,  # K
            "temperature_max": 6000.0,
            "velocity_strength": 8.0,
            "vorticity": 1.0,
            "dissolve_speed": 40,
            "use_dissolve": True,
            "turbulence_strength": 2.0,
            "resolution": 128,
        },
        "steam": {
            "density": 0.8,
            "temperature": 100.0,  # °C above ambient (373 K)
            "temperature_max": 300.0,
            "velocity_strength": 0.8,
            "vorticity": 0.05,
            "dissolve_speed": 60,
            "use_dissolve": True,
            "turbulence_strength": 0.1,
            "resolution": 48,
        },
    },
}

# ─── 4. RENDERING CONSTANTS ────────────────────────────────────────────────────

RENDERING_CONSTANTS: dict[str, Any] = {
    # ── Fundamental Physical Constants ───────────────────────────────────────
    "planck_constant": 6.62607015e-34,  # J·s
    "speed_of_light": 299_792_458.0,  # m/s (exact by definition)
    "stefan_boltzmann": 5.670374419e-8,  # W/(m²·K⁴)
    "boltzmann_constant": 1.380649e-23,  # J/K
    "avogadro_constant": 6.02214076e23,  # mol⁻¹
    "elementary_charge": 1.602176634e-19,  # C
    # ── Color Temperature Table (blackbody, in Kelvin) ────────────────────────
    "color_temperature": {
        "candle": 1800,
        "matchflame": 2000,
        "tungsten_40w": 2700,
        "tungsten_100w": 2900,
        "warm_white_led": 3000,
        "halogen": 3200,
        "neutral_white": 4000,
        "cool_white": 5000,
        "daylight_d55": 5500,
        "camera_flash": 5600,
        "daylight_d65": 6500,
        "overcast_sky": 6500,
        "hazy_sky": 8000,
        "blue_sky_indirect": 10000,
        "clear_blue_sky": 15000,
    },
    # ── Luminance / Illuminance Table ────────────────────────────────────────
    # Approximate luminance values in cd/m² for common sources
    "luminance_cd_m2": {
        "moonlight": 0.001,
        "indoor_ambient": 100.0,
        "overcast_day": 2_000.0,
        "cloudy_day": 10_000.0,
        "clear_day": 30_000.0,
        "sun_at_horizon": 600_000.0,
        "sun_at_zenith": 1_600_000_000.0,
        "white_paper_noon": 25_000.0,
        "lcd_monitor": 300.0,
        "oled_peak": 1_000.0,
        "candle": 10_000.0,
    },
    # ── Illuminance in lux for common scenes ─────────────────────────────────
    "illuminance_lux": {
        "full_moon": 0.1,
        "street_lamp": 10.0,
        "indoor_corridor": 100.0,
        "indoor_office": 500.0,
        "studio_photography": 1_000.0,
        "overcast_day": 10_000.0,
        "sunlight_indirect": 25_000.0,
        "direct_sunlight": 100_000.0,
    },
    # ── Gamma / Colorspace ───────────────────────────────────────────────────
    "srgb_gamma": 2.2,
    "srgb_to_linear": {
        "description": "C_linear = C_srgb^2.2 (simplified), or use the exact piecewise IEC 61966-2-1 formula",
        "exact_formula": "if C <= 0.04045: C_linear = C/12.92 else: C_linear = ((C + 0.055)/1.055)^2.4",
        "display_referred": True,
    },
    "aces_ap0_primaries": {
        "red": (0.73470, 0.26530),
        "green": (0.00000, 1.00000),
        "blue": (0.00010, -0.07700),
        "white": (0.32168, 0.33767),
    },
    "srgb_primaries": {
        "red": (0.6400, 0.3300),
        "green": (0.3000, 0.6000),
        "blue": (0.1500, 0.0600),
        "white": (0.3127, 0.3290),  # D65
    },
    # ── Human Luminosity Function Peak ───────────────────────────────────────
    "peak_luminosity_wavelength_nm": 555.0,  # photopic (daytime)
    "scotopic_peak_nm": 507.0,  # scotopic (low-light rods)
    "visible_spectrum_nm": (380, 780),
}

# ─── 5. TOPOLOGY RULES ────────────────────────────────────────────────────────

TOPOLOGY_RULES: dict[str, Any] = {
    "pole_rules": {
        "n_pole": {
            "edge_count": 3,
            "description": "Three-edge pole (N-pole). Common at concave corners. Use sparingly; can create pinching under subdivision.",
            "ideal_use": [
                "sharp corners",
                "hard-surface creases",
                "controlled termination of edge loops",
            ],
            "avoid": ["flat surfaces", "organic forms where smoothness required"],
        },
        "e_pole": {
            "edge_count": 5,
            "description": "Five-edge pole (E-pole). Common artifact of loop redirecting. Acceptable if placed in low-curvature areas.",
            "ideal_use": [
                "redirecting edge flow",
                "topology transitions",
                "ear bases",
                "finger webbing",
            ],
            "avoid": ["high-curvature areas", "close to eye or mouth rings"],
        },
        "star_pole": {
            "edge_count": "6+",
            "description": "Star pole with six or more edges. Almost always a sign of poor topology. Causes severe mesh artifacts.",
            "ideal_use": [],
            "avoid": ["everywhere — split or redirect into multiple poles"],
        },
        "ideal_poles_per_object": {
            "simple_hard_surface": "< 10 poles total",
            "organic_character": "20-60 well-distributed poles",
            "hero_prop": "< 20 poles",
        },
    },
    "quad_guidelines": {
        "ideal_face_type": "QUAD",
        "tris_allowed": "only for game assets at topology boundaries, never under subdivision",
        "ngons_allowed": "only for flat non-subdivided hard surface (booleans), never on curved surfaces",
        "ideal_edge_flow_angles": {
            "organic": "follow muscle/bone direction, not arbitrary",
            "hard_surface": "follow silhouette, mechanical function lines",
            "subdivision_safety": "no edge angle > 45 degrees between adjacent quads for smooth subdivision",
        },
        "loop_cut_principle": "Every edge loop must start and terminate at a pole; it cannot float freely.",
        "subdivision_safety": {
            "minimum_edge_bevel": 1,
            "recommended_supporting_loops": 2,
            "crease_alternative": "Use crease weight 1.0 instead of supporting loops when poly count is critical",
        },
    },
    "retopology_edge_counts": {
        "face_loops": {
            "eye_ring_inner": 8,
            "eye_ring_outer": 16,
            "mouth_ring_inner": 12,
            "mouth_ring_outer": 20,
            "nose_bridge": 6,
            "forehead_loops": 4,
            "cheek_loops": 6,
        },
        "hand_polys": {
            "full_hand_game": 500,
            "full_hand_film": 2000,
            "single_finger": 80,
            "palm": 200,
        },
        "body_base_polys": {
            "game_character_body": 3_000,
            "film_hero_body": 15_000,
            "subdivision_base": 1_500,
        },
    },
    "game_engine_budgets": {
        "mobile": {
            "min_triangles": 1_000,
            "max_triangles": 5_000,
            "texture_resolution": 512,
            "max_draw_calls": 50,
            "bone_count": 30,
        },
        "indie": {
            "min_triangles": 5_000,
            "max_triangles": 25_000,
            "texture_resolution": 2048,
            "max_draw_calls": 200,
            "bone_count": 60,
        },
        "aaa_character": {
            "min_triangles": 25_000,
            "max_triangles": 100_000,
            "texture_resolution": 4096,
            "max_draw_calls": 500,
            "bone_count": 200,
        },
        "hero_prop": {
            "min_triangles": 10_000,
            "max_triangles": 50_000,
            "texture_resolution": 4096,
            "max_draw_calls": 100,
            "bone_count": 0,
        },
    },
}

# ─── 6. PHYSICS FORMULAS ──────────────────────────────────────────────────────
# All formulas given as human-readable string representations with LaTeX notation.

PHYSICS_FORMULAS: dict[str, dict[str, str]] = {
    "rendering_equation": {
        "name": "The Rendering Equation (Kajiya 1986)",
        "formula": "L_o(x, ω_o) = L_e(x, ω_o) + ∫_Ω f_r(x, ω_i, ω_o) L_i(x, ω_i) (ω_i · n) dω_i",
        "variables": "L_o: outgoing radiance; L_e: emitted radiance; f_r: BRDF; L_i: incoming radiance; ω_i: incident direction; ω_o: outgoing direction; n: surface normal",
        "significance": "The foundational integral equation of photorealistic rendering. All path tracers solve a Monte Carlo approximation of this.",
    },
    "cook_torrance_brdf": {
        "name": "Cook-Torrance Microfacet BRDF",
        "formula": "f_r(ω_i, ω_o) = D(h) F(ω_o, h) G(ω_i, ω_o) / (4 (n·ω_i)(n·ω_o))",
        "variables": "D: normal distribution function (NDF); F: Fresnel term; G: geometry/masking-shadowing term; h: half-vector = normalize(ω_i + ω_o)",
        "significance": "The industry-standard BRDF for physically based rendering. Describes specular reflection from rough microsurfaces.",
    },
    "fresnel_schlick": {
        "name": "Schlick Fresnel Approximation",
        "formula": "F(ω_o, h) ≈ F_0 + (1 - F_0)(1 - (ω_o · h))^5",
        "variables": "F_0: reflectance at normal incidence = ((n_1 - n_2)/(n_1 + n_2))^2; ω_o · h: cosine of angle between view and half-vector",
        "significance": "Fast approximation to the full Fresnel equations. Describes how reflectivity increases at grazing angles (e.g., a window is near-transparent straight-on but mirror-like at grazing angles).",
    },
    "fresnel_full": {
        "name": "Fresnel Equations (full form for unpolarized light)",
        "formula": "R_s = ((n_1 cos θ_i - n_2 cos θ_t)/(n_1 cos θ_i + n_2 cos θ_t))^2; R_p = ((n_1 cos θ_t - n_2 cos θ_i)/(n_1 cos θ_t + n_2 cos θ_i))^2; R = (R_s + R_p)/2",
        "variables": "R_s: s-polarization reflectance; R_p: p-polarization reflectance; n_1, n_2: IOR of media; θ_i: angle of incidence; θ_t: angle of transmission",
        "significance": "Exact solution for reflection/transmission at a planar interface. Required for accurate caustics and thin-film interference.",
    },
    "snells_law": {
        "name": "Snell's Law of Refraction",
        "formula": "n_1 sin θ_1 = n_2 sin θ_2",
        "variables": "n_1, n_2: IOR of first and second media; θ_1: angle of incidence; θ_2: angle of refraction",
        "significance": "Governs direction change when light crosses an IOR boundary. Critical for glass, water, and all transmissive materials. Total internal reflection occurs when sin θ_2 > 1.",
    },
    "ggx_ndf": {
        "name": "GGX / Trowbridge-Reitz Normal Distribution Function",
        "formula": "D_GGX(h) = α² / (π ((n·h)²(α² - 1) + 1)²)",
        "variables": "α: roughness² (artist roughness squared); n: surface normal; h: half-vector; π: pi",
        "significance": "Models the statistical distribution of microfacet normals. GGX has longer tails than Beckmann, producing the bright specular highlight with soft haze seen in real rough metals.",
    },
    "beckmann_ndf": {
        "name": "Beckmann Normal Distribution Function",
        "formula": "D_B(h) = exp(-tan²θ_h / α²) / (π α² cos⁴θ_h)",
        "variables": "θ_h: angle between half-vector and normal; α: roughness parameter",
        "significance": "Original NDF for Cook-Torrance. More physically grounded than Phong but has shorter tails than GGX. Still used in some production pipelines.",
    },
    "smith_masking_shadowing": {
        "name": "Smith Geometry Function",
        "formula": "G(ω_i, ω_o) = G_1(ω_i) G_1(ω_o); G_1(ω) = 2(n·ω) / ((n·ω)(2 - α) + α)  [Schlick approx]",
        "variables": "G_1: single-direction masking; α: roughness; n: normal; ω: light or view direction",
        "significance": "Accounts for self-shadowing and self-masking of microfacets. Without G, rough surfaces would appear too bright at grazing angles.",
    },
    "beer_lambert": {
        "name": "Beer-Lambert Law (Volumetric Absorption)",
        "formula": "T(d) = exp(-σ_a d)",
        "variables": "T: transmittance; σ_a: absorption coefficient (m⁻¹); d: path length through medium (m)",
        "significance": "Describes how light attenuates exponentially through an absorbing medium. Foundation of volumetric rendering — colored glass, murky water, participating media.",
    },
    "blackbody_planck": {
        "name": "Planck's Law of Blackbody Radiation",
        "formula": "B_λ(T) = (2hc²/λ⁵) · 1/(exp(hc/(λk_B T)) - 1)",
        "variables": "B_λ: spectral radiance (W·sr⁻¹·m⁻³); h: Planck's constant; c: speed of light; λ: wavelength (m); k_B: Boltzmann constant; T: temperature (K)",
        "significance": "Gives the spectrum of an ideal thermal emitter at temperature T. All physically based emission (fire, sun, heated metals) is fundamentally this curve, shifted by temperature.",
    },
    "lambertian_brdf": {
        "name": "Lambertian Diffuse BRDF",
        "formula": "f_r = albedo / π",
        "variables": "albedo: surface reflectance in [0,1]; π: normalization constant ensuring energy conservation",
        "significance": "The simplest physically valid BRDF. Scatters light equally in all directions. Correct for matte, chalky surfaces. Used as the diffuse lobe in the Principled BSDF.",
    },
    "subsurface_dipole": {
        "name": "Dipole Diffusion Approximation (Jensen 2001)",
        "formula": "R_d(r) = (α'/(4π)) [z_r(σ_tr + 1/d_r)exp(-σ_tr d_r)/d_r² + z_v(σ_tr + 1/d_v)exp(-σ_tr d_v)/d_v²]",
        "variables": "r: distance from entry point; α': reduced single-scatter albedo; σ_tr: effective transport coefficient; z_r, z_v: real/virtual source depths; d_r, d_v: distances to real/virtual sources",
        "significance": "Analytical approximation of light scattering inside translucent materials (skin, marble, wax). Dramatically faster than full volumetric path tracing for SSS.",
    },
}

# ─── 7. GPU SYNTHESIS JOBS ────────────────────────────────────────────────────
# Each job represents a batch of synthesis to run on GPU (Qwen2.5-72B or similar).
# Jobs are processed by physics_sim.py::generate_synthesis_job_queue().

GPU_SYNTHESIS_JOBS: list[dict[str, Any]] = [
    {
        "job_id": "feynman_optics_v1",
        "source": "The Feynman Lectures on Physics, Volume I, Chapter 26: Optics — The Principle of Least Time",
        "url": "https://www.feynmanlectures.caltech.edu/I_26.html",
        "domain": "optics_pbr",
        "synthesis_prompt": (
            "You are converting Nobel laureate Richard Feynman's explanation of optics into "
            "3D artist training data for Nalana, a voice-controlled Blender AI. "
            "Read the following excerpt from Feynman Lectures Vol I Ch 26 about Fermat's Principle "
            "of Least Time, Snell's Law, and reflection. "
            "For each major physical concept, generate a training pair where a 3D artist asks "
            "a natural question and the AI gives a concise, expert answer that connects the "
            "underlying physics to practical PBR material decisions in Blender. "
            "The artist questions should be conversational ('why does glass bend light?', "
            "'why does water look reflective at sunset?'). "
            "The answers must cite the physics principle, give the formula if relevant, "
            "and immediately explain the practical implication for setting IOR, "
            "transmission, or Fresnel values in a Blender material node. "
            'Output JSON array of {"user": "...", "assistant": "...", "physics_principle": "...", "blender_parameter": "..."}'
        ),
        "output_format": "qa_pairs",
        "expected_pairs": 40,
        "priority": 1,
        "status": "pending",
    },
    {
        "job_id": "feynman_electromagnetism_v1",
        "source": "The Feynman Lectures on Physics, Volume II, Chapters 32-33: Refractive Index of Dense Materials, Reflection from Surfaces",
        "url": "https://www.feynmanlectures.caltech.edu/II_32.html",
        "domain": "optics_metals",
        "synthesis_prompt": (
            "You are converting Feynman's quantum mechanical explanation of why metals are shiny "
            "and why IOR is complex into 3D training data for Nalana. "
            "Feynman explains how free electrons in metals resonate with light at the plasma frequency, "
            "causing strong wavelength-dependent absorption (the k term in complex IOR) and making "
            "metals reflect most visible light. Gold and copper reflect red/yellow preferentially "
            "because their plasma frequency falls within visible range. "
            "Generate training pairs covering: why metallic vs. dielectric IOR differs, "
            "why gold looks yellow (selective absorption of blue), why silver is spectrally neutral, "
            "why rough metals lose their color saturation, and how the metallic slider in Principled BSDF "
            "switches between these two physical regimes. "
            "Each answer should give the physical mechanism AND the practical Blender setting. "
            'Output JSON array of {"user": "...", "assistant": "...", "physics_principle": "...", "blender_parameter": "..."}'
        ),
        "output_format": "qa_pairs",
        "expected_pairs": 35,
        "priority": 1,
        "status": "pending",
    },
    {
        "job_id": "pbrt_radiometry_v1",
        "source": "Physically Based Rendering: From Theory to Implementation (Pharr, Jakob, Humphreys), Chapter 5: Color and Radiometry",
        "url": "https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color",
        "domain": "radiometry_color",
        "synthesis_prompt": (
            "You are converting PBRT Chapter 5 on radiometry into training data for Nalana. "
            "This chapter covers: radiance (L), irradiance (E), radiosity, spectral power distributions, "
            "color matching functions (CIE XYZ), and the relationship between physical light and perceived color. "
            "Generate tutorial-style training pairs that teach a 3D artist why: "
            "1) Energy conservation matters in BRDF design "
            "2) Why the rendering equation integrates over the hemisphere "
            "3) Why emission color should be in linear space not sRGB "
            "4) Why area lights need to scale with surface area "
            "5) Why blackbody temperature maps to Blender's Blackbody node "
            "6) What radiance vs. irradiance means for HDRi lighting "
            "Each pair should include a short reasoning chain showing how the math leads to the Blender behavior. "
            'Output JSON array of {"user": "...", "assistant": "...", "reasoning": "...", "blender_node": "..."}'
        ),
        "output_format": "reasoning_chains",
        "expected_pairs": 50,
        "priority": 2,
        "status": "pending",
    },
    {
        "job_id": "pbrt_reflection_models_v1",
        "source": "Physically Based Rendering, Chapter 9: Reflection Models (BRDFs, BTDFs, BSSRDFs)",
        "url": "https://pbr-book.org/4ed/Reflection_Models",
        "domain": "brdf_bssrdf",
        "synthesis_prompt": (
            "You are converting PBRT Chapter 9 on reflection models into Nalana training data. "
            "Focus on: Lambertian diffuse, specular reflection/refraction, microfacet theory "
            "(Oren-Nayar, Cook-Torrance), Fresnel blend models, and BSSRDF for subsurface scattering. "
            "For each model generate: "
            "a) A conceptual Q&A explaining the physical phenomenon "
            "b) A tutorial step showing exactly which Blender Principled BSDF sliders to adjust and why "
            "c) A comparison pair: 'what is the difference between roughness=0.1 and roughness=0.9?' "
            "   with the BRDF-level explanation "
            "Include at least 8 pairs for SSS/BSSRDF materials (skin, marble, wax, translucent plastic) "
            "since these are the most misunderstood by 3D artists. "
            'Output JSON array of {"user": "...", "assistant": "...", "material_type": "...", "principled_bsdf_params": {...}}'
        ),
        "output_format": "tutorial_steps",
        "expected_pairs": 60,
        "priority": 1,
        "status": "pending",
    },
    {
        "job_id": "vitruvius_architecture_v1",
        "source": "Vitruvius, De Architectura (Ten Books on Architecture), Books I-III: Firmitas, Utilitas, Venustas",
        "url": "https://archive.org/details/vitruviusonarch00vitrgoog",
        "domain": "architecture_proportion",
        "synthesis_prompt": (
            "You are converting Vitruvius's foundational architectural treatise (circa 30 BCE) into "
            "3D modeling training data for Nalana. "
            "Vitruvius establishes three principles: firmitas (structural integrity), utilitas (function), "
            "venustas (beauty). He documents the classical orders (Doric, Ionic, Corinthian), "
            "column proportions, module systems, and site planning. "
            "Generate training pairs covering: "
            "1) How to model classical column proportions (Doric = 6:1 height-to-diameter, Ionic = 8:1, Corinthian = 10:1) "
            "2) Entasis (subtle column taper) and how to implement it with Blender's curve modifier "
            "3) Capital geometry for each order (Doric abacus, Ionic volute, Corinthian acanthus) "
            "4) Triglyphs and metopes in the Doric frieze — exact proportions for box modeling "
            "5) Site orientation rules (building facades should face south for light) "
            "Each pair should be actionable Blender steps rooted in the ancient text's proportional system. "
            'Output JSON array of {"user": "...", "assistant": "...", "architectural_element": "...", "proportion_ratio": "...", "blender_steps": [...]}'
        ),
        "output_format": "tutorial_steps",
        "expected_pairs": 45,
        "priority": 3,
        "status": "pending",
    },
    {
        "job_id": "bauhaus_design_v1",
        "source": "Walter Gropius, Bauhaus Manifesto and Program (1919); Paul Klee, Pedagogical Sketchbook (1925); Wassily Kandinsky, Point and Line to Plane (1926)",
        "url": "https://bauhauskooperation.com/knowledge/the-bauhaus/history/manifesto/",
        "domain": "design_principles",
        "synthesis_prompt": (
            "You are converting the core Bauhaus design philosophy into 3D modeling training data for Nalana. "
            "The Bauhaus unified craft and fine art under functional design principles: "
            "form follows function, geometric abstraction, material honesty, systematic color theory. "
            "Itten's color wheel, Klee's compositional rhythm, Kandinsky's point-line-plane theory. "
            "Generate training pairs covering: "
            "1) Form follows function — how to critique and improve a 3D model's topology to serve its animation/use purpose "
            "2) Geometric abstraction — breaking organic forms into primitive geometry (Klee's method) for low-poly stylized work "
            "3) Color relationships — complementary, analogous, triadic — as applied to Blender material palettes "
            "4) Material honesty — why concrete should look like concrete, not marble; avoiding surface-level decoration "
            "5) Grid and module systems — how to set up Blender scene scale and snap settings for modular architecture "
            'Output JSON array of {"user": "...", "assistant": "...", "bauhaus_principle": "...", "blender_application": "..."}'
        ),
        "output_format": "reasoning_chains",
        "expected_pairs": 35,
        "priority": 3,
        "status": "pending",
    },
    {
        "job_id": "ashby_materials_v1",
        "source": "Michael Ashby, Materials Selection in Mechanical Design, 4th Edition — Material Property Charts and Selection Methodology",
        "url": "https://www.sciencedirect.com/book/9781856176637/materials-selection-in-mechanical-design",
        "domain": "materials_science",
        "synthesis_prompt": (
            "You are converting Ashby's materials science selection methodology into 3D visualization "
            "training data for Nalana. "
            "Ashby's material property charts (Ashby plots) map materials by Young's modulus vs. density, "
            "thermal conductivity vs. electrical conductivity, fracture toughness vs. strength, etc. "
            "These charts reveal the fundamental physical reason why materials look and behave the way they do. "
            "Generate training pairs covering: "
            "1) Why high-density metals (steel, lead) look heavier in animation (how to convey mass through motion) "
            "2) Why ceramic fracture is different from metal fracture — brittle vs. ductile tearing topology "
            "3) Why foam and rubber have totally different stiffness despite similar density — Blender soft body settings "
            "4) Surface finish and its relationship to hardness — why harder materials polish smoother (lower roughness) "
            "5) Thermal glow — which materials glow red before melting vs. which melt suddenly (metals vs. ceramics) "
            "6) Why wood grain direction matters for bending (anisotropic stiffness → anisotropic texture and shading) "
            "Each answer should connect the materials science property to a specific visual/simulation behavior in Blender. "
            'Output JSON array of {"user": "...", "assistant": "...", "material_property": "...", "visual_consequence": "...", "blender_setting": "..."}'
        ),
        "output_format": "qa_pairs",
        "expected_pairs": 50,
        "priority": 2,
        "status": "pending",
    },
]

# ─── 8. EXPERT TEXT PROMPTS ───────────────────────────────────────────────────
# Templates used at GPU synthesis time. The {excerpt} placeholder is replaced
# with the actual text chunk being synthesized.

EXPERT_TEXT_PROMPTS: dict[str, str] = {
    "optics": (
        "You are a physics educator specializing in optics and photonics, working as a dataset "
        "engineer for Nalana — a voice-controlled 3D AI. Convert the following optics text into "
        "high-quality training pairs for a 3D artist audience. Each pair must: "
        "(1) phrase the question as a practical artist need (not a physics exam), "
        "(2) answer with the physics mechanism AND the immediate Blender material implication, "
        "(3) be factually precise — cite correct IOR values, formula names, and parameter ranges. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "electromagnetism": (
        "You are an expert in condensed matter physics and plasmonics, converting electromagnetic "
        "theory into 3D rendering training data for Nalana. The audience is professional 3D artists "
        "who want to understand why metals look the way they do. Convert the following text into "
        "training pairs that explain complex IOR, skin depth, plasma frequency, and how these affect "
        "the metallic/roughness/specular tint controls in Blender's Principled BSDF. "
        "Be quantitatively precise. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "radiometry": (
        "You are a rendering engineer with deep knowledge of radiometry and colorimetry, creating "
        "training data for Nalana. Convert the following radiometry text into training pairs that "
        "teach 3D artists the connection between physical light units (radiance, irradiance, luminance) "
        "and Blender's lighting controls (emission strength, area light power in watts, exposure). "
        "Include correct unit conversions and explain why gamma correction and color space management "
        "matter for physically correct output. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "brdf_materials": (
        "You are a senior technical artist and rendering researcher creating training data for Nalana. "
        "Convert the following BRDF/reflection model text into training pairs that help 3D artists "
        "make correct PBR material decisions. Each pair should bridge academic BRDF theory to the "
        "specific sliders and values in Blender's Principled BSDF. Prioritize pairs about edge cases: "
        "grazing angle behavior, energy conservation, layered materials, and subsurface scattering. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "architecture": (
        "You are an architectural historian and practicing 3D visualization artist creating training "
        "data for Nalana. Convert the following architectural theory text into training pairs that "
        "help 3D artists model historically and proportionally correct architectural elements. "
        "Ground every answer in specific Blender modeling techniques (loop cuts, bevel, array modifier, "
        "curve deform) while honoring the proportional systems described in the source text. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "design_principles": (
        "You are a design educator with Bauhaus training and expertise in 3D generative systems, "
        "creating training data for Nalana. Convert the following design theory text into training "
        "pairs that help 3D artists apply formal design principles (proportion, rhythm, balance, "
        "material honesty) to their Blender scenes. Each answer should give both the design principle "
        "and the specific Blender workflow that implements it. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "materials_science": (
        "You are a materials scientist and 3D simulation expert creating training data for Nalana. "
        "Convert the following materials science text into training pairs that help 3D artists "
        "correctly simulate material behavior — how things deform, fracture, melt, flow, and interact "
        "with light — based on their actual physical properties. Ground every answer in specific "
        "Blender physics system settings (cloth, fluid, rigid body, soft body) with correct parameter values. "
        "Text excerpt:\n\n{excerpt}"
    ),
    "simulation": (
        "You are a VFX technical director specializing in physics simulation, creating training data "
        "for Nalana. Convert the following simulation physics text into training pairs that teach "
        "3D artists to set up realistic simulations in Blender. Cover cloth, fluids, rigid body, "
        "soft body, and particle systems. Each answer must include the physical justification for "
        "the parameter values (e.g., 'silk has low bending stiffness because its fibers are extremely "
        "fine and can rotate relative to each other with almost no resistance'). "
        "Text excerpt:\n\n{excerpt}"
    ),
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────


def get_material_preset(name: str) -> dict[str, Any]:
    """
    Retrieve a PBR material preset by name.

    Args:
        name: Key from PBR_PRESETS (e.g., 'gold', 'skin_caucasian', 'glass_clear').

    Returns:
        Dict of Principled BSDF parameter values.

    Raises:
        KeyError: If the preset name is not found. Message lists available names.
    """
    if name not in PBR_PRESETS:
        available = ", ".join(sorted(PBR_PRESETS.keys()))
        raise KeyError(
            f"Material preset '{name}' not found. Available presets:\n{available}"
        )
    return dict(PBR_PRESETS[name])


def get_simulation_preset(category: str, preset: str) -> dict[str, Any]:
    """
    Retrieve a simulation preset by category and preset name.

    Args:
        category: Top-level category — one of 'cloth', 'fluid', 'rigid_body',
                  'soft_body', 'smoke'.
        preset:   Preset name within that category — e.g., 'silk', 'water', 'rubber'.

    Returns:
        Dict of simulation parameter values for the requested preset.

    Raises:
        KeyError: If category or preset is not found, with helpful message listing options.
    """
    if category not in SIMULATION_PRESETS:
        available_cats = ", ".join(sorted(SIMULATION_PRESETS.keys()))
        raise KeyError(
            f"Simulation category '{category}' not found. Available categories: {available_cats}"
        )

    category_data = SIMULATION_PRESETS[category]

    if preset not in category_data:
        available_presets = ", ".join(sorted(category_data.keys()))
        raise KeyError(
            f"Preset '{preset}' not found in category '{category}'. "
            f"Available presets: {available_presets}"
        )

    return dict(category_data[preset])


def get_ior(material: str) -> float | tuple[float, float]:
    """
    Retrieve the index of refraction for a material.

    Args:
        material: Key from IOR_TABLE.

    Returns:
        Float for dielectrics, (n, k) tuple for metals.

    Raises:
        KeyError: If material is not in the table.
    """
    if material not in IOR_TABLE:
        available = ", ".join(sorted(IOR_TABLE.keys()))
        raise KeyError(
            f"Material '{material}' not found in IOR_TABLE. Available materials:\n{available}"
        )
    return IOR_TABLE[material]


def list_presets_by_category() -> dict[str, list[str]]:
    """
    Return a dict mapping each simulation category to its list of preset names.
    Useful for building UI dropdowns or dataset variant enumeration.
    """
    return {
        category: sorted(presets.keys())
        for category, presets in SIMULATION_PRESETS.items()
    }
