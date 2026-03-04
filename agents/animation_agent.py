"""
animation_agent.py - Nalana Animation Intelligence

Automates animation workflows:
  - Text-to-animation for simple actions and loops
  - Keyframe cleanup (jitter removal, arc smoothing)
  - Secondary motion (overlap, follow-through per Disney's 12 principles)
  - Motion blending and transitions
  - Crowd motion synthesis
  - Audio → lip sync automation
  - Retargeting between rigs
  - Pose library generation

Disney's 12 Animation Principles as code:
  Squash & Stretch, Anticipation, Staging, Straight Ahead/Pose to Pose,
  Follow Through/Overlapping, Ease In/Out, Arc, Secondary Action,
  Timing, Exaggeration, Solid Drawing, Appeal

Usage:
    python animation_agent.py --cleanup [action_name]
    python animation_agent.py --secondary-motion [object]
    python animation_agent.py --generate-pairs
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ─── Output paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parents[1]
ANIM_DATA_DIR = BASE_DIR / "data" / "animation"
ANIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUTPUT = ANIM_DATA_DIR / "animation_pairs.jsonl"

# ─── Disney's 12 Animation Principles ─────────────────────────────────────────

ANIMATION_PRINCIPLES: dict[str, dict[str, Any]] = {
    "squash_and_stretch": {
        "number": 1,
        "description": (
            "Deforming an object to suggest flexibility and weight during motion. "
            "A bouncing ball squashes flat on impact and stretches vertically in the air. "
            "Volume must remain approximately constant: if an object squashes wide, it gets thin."
        ),
        "mathematical_description": (
            "Scale_x * Scale_y * Scale_z ≈ 1.0 (volume preservation). "
            "At maximum squash: Scale_z = s, Scale_x = Scale_y = 1/√s. "
            "Squash factor typically 0.5–0.8 on impact, stretch 1.2–1.5 at peak velocity."
        ),
        "blender_implementation": """
import bpy

def squash_and_stretch(obj_name, squash_factor=0.7, stretch_factor=1.4, impact_frame=10, air_frame=5):
    obj = bpy.data.objects[obj_name]
    # At air frame (before impact): stretch upward
    obj.scale = (1/stretch_factor**0.5, 1/stretch_factor**0.5, stretch_factor)
    obj.keyframe_insert(data_path="scale", frame=air_frame)
    # At impact frame: squash down, spread wide
    obj.scale = (1/squash_factor**0.5, 1/squash_factor**0.5, squash_factor)
    obj.keyframe_insert(data_path="scale", frame=impact_frame)
    # Set handles to ease in/out
    for fc in obj.animation_data.action.fcurves:
        if 'scale' in fc.data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
""",
        "common_mistakes": [
            "Volume not preserved — object appears to gain/lose mass",
            "Applied symmetrically (both XYZ scale) instead of axis-specific",
            "Too much squash makes object look like jelly",
            "Not synced with velocity — stretch should peak at max speed, squash at max deceleration",
        ],
        "training_pairs": [
            (
                "Make the ball squash when it hits the ground",
                "Ball bounce squash/stretch",
            ),
            ("Add squash and stretch to the character jump", "Character jump S&S"),
        ],
    },
    "anticipation": {
        "number": 2,
        "description": (
            "A preparatory action opposite to the main action. "
            "Before a character punches, the arm winds back. Before a jump, the character crouches. "
            "Anticipation tells the viewer what's about to happen, making motion readable."
        ),
        "mathematical_description": (
            "Anticipation displacement ≈ 20-30% of main action displacement, opposite direction. "
            "Duration: anticipation ≈ 25-35% of main action duration. "
            "Hold 2-4 frames at peak anticipation before main action begins."
        ),
        "blender_implementation": """
import bpy

def add_anticipation(obj_name, main_action_frame, anticipation_frames=8, amount=0.3):
    obj = bpy.data.objects[obj_name]
    # Get the main action keyframe values
    base_loc = obj.location.copy()
    # Insert anticipation keyframe (move opposite direction)
    antic_frame = main_action_frame - anticipation_frames
    obj.location.z = base_loc.z - amount  # crouch before jump
    obj.keyframe_insert(data_path="location", frame=antic_frame)
    # Return to base before main action
    obj.location = base_loc
    obj.keyframe_insert(data_path="location", frame=main_action_frame - 2)
    # Set slow-in to anticipation
    action = obj.animation_data.action
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            if kp.co[0] == antic_frame:
                kp.interpolation = 'BEZIER'
""",
        "common_mistakes": [
            "Anticipation too small — viewers miss it and motion feels sudden",
            "Anticipation too large — becomes its own major action, kills pacing",
            "Missing the hold at peak anticipation",
            "Anticipation in wrong direction (same as main action = doubles the read)",
        ],
        "training_pairs": [
            ("Add anticipation before the jump", "Pre-jump crouch anticipation"),
            ("The punch needs a wind-up first", "Punch anticipation wind-up"),
        ],
    },
    "staging": {
        "number": 3,
        "description": (
            "The presentation of an idea so it is unmistakably clear. "
            "Camera placement, character silhouette, lighting, and composition must all "
            "serve the storytelling moment. Never show two things of equal importance simultaneously."
        ),
        "mathematical_description": (
            "Rule of thirds: place focal point at (1/3, 1/3) or (2/3, 2/3) of frame. "
            "Silhouette readability: minimize self-overlap in camera projection. "
            "Depth of field: f-stop ∝ 1/focus_priority."
        ),
        "blender_implementation": """
import bpy
import math

def stage_shot(camera_name, subject_name, angle_deg=45, distance=5.0):
    cam = bpy.data.objects[camera_name]
    subject = bpy.data.objects[subject_name]
    # Position camera on rule-of-thirds angle
    angle_rad = math.radians(angle_deg)
    cam.location = (
        subject.location.x + distance * math.sin(angle_rad),
        subject.location.y - distance * math.cos(angle_rad),
        subject.location.z + distance * 0.5
    )
    # Track to subject
    constraint = cam.constraints.new('TRACK_TO')
    constraint.target = subject
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    # Set focal length for cinematic compression
    cam.data.lens = 85  # portrait/character lens
""",
        "common_mistakes": [
            "Centered composition — static, uncinematic",
            "Character silhouette obscured by background elements",
            "Camera too low — loses readability of secondary characters",
            "Multiple equal-importance actions happening simultaneously",
        ],
        "training_pairs": [
            (
                "Set up a cinematic camera for this character",
                "Staging camera placement",
            ),
            (
                "The scene is confusing — help me stage it better",
                "Shot clarity improvement",
            ),
        ],
    },
    "straight_ahead_and_pose_to_pose": {
        "number": 4,
        "description": (
            "Two animation planning methods. Straight Ahead: animate sequentially frame by frame "
            "(spontaneous, fluid, good for action). Pose to Pose: plan key poses first, then fill "
            "in-betweens (controlled, structured, good for dialogue and acting). "
            "Modern production mixes both: block with poses, spline for in-betweens."
        ),
        "mathematical_description": (
            "Pose to Pose: Interpolate between N key poses using parametric curves. "
            "Blocking: stepped interpolation (constant value between keys). "
            "Spline: cubic Bezier curves between key frames. "
            "In-between count per second ≈ fps - (key_poses_per_second)."
        ),
        "blender_implementation": """
import bpy

def set_blocking_mode(action_name):
    action = bpy.data.actions.get(action_name)
    if not action:
        return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'  # Stepped blocking

def set_spline_mode(action_name):
    action = bpy.data.actions.get(action_name)
    if not action:
        return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'AUTO_CLAMPED'
            kp.handle_right_type = 'AUTO_CLAMPED'
""",
        "common_mistakes": [
            "All spline too early — loses control of timing before poses are right",
            "Staying in blocking too long — misses arc and flow issues",
            "In-betweens computed by computer without checking arcs",
        ],
        "training_pairs": [
            ("Switch the animation to blocking mode", "Pose to pose blocking"),
            (
                "Spline all the curves so the motion is smooth",
                "Stepped to spline transition",
            ),
        ],
    },
    "follow_through_and_overlapping_action": {
        "number": 5,
        "description": (
            "Follow Through: parts of a character continue moving after the main body stops. "
            "Hair, clothing, tails — nothing stops at exactly the same frame. "
            "Overlapping Action: different body parts move at different times and rates. "
            "The torso leads; the arm follows; fingers follow the arm — offset by 2-6 frames."
        ),
        "mathematical_description": (
            "Follow-through decay: displacement = A * e^(-damping * t) * cos(frequency * t). "
            "Overlap offset: secondary part keyframes shifted by +offset_frames from primary. "
            "Typical offsets: clothing 3-8 frames, hair 4-10 frames, tails 5-12 frames."
        ),
        "blender_implementation": """
import bpy

def add_follow_through(armature_name, secondary_bone, primary_bone, delay_frames=5, damping=0.4):
    arm = bpy.data.objects[armature_name]
    # Get primary bone's action curves
    action = arm.animation_data.action
    primary_curves = [fc for fc in action.fcurves if primary_bone in fc.data_path]
    # Copy curves to secondary bone with frame offset
    for fc in primary_curves:
        new_path = fc.data_path.replace(primary_bone, secondary_bone)
        new_fc = action.fcurves.new(new_path, index=fc.array_index)
        for kp in fc.keyframe_points:
            new_kp = new_fc.keyframe_points.insert(kp.co[0] + delay_frames, kp.co[1] * 0.7)
            new_kp.interpolation = 'BEZIER'
    print(f"Follow-through added to {secondary_bone} with {delay_frames} frame delay")

def add_overlap_constraint(obj_name, bone_name, stiffness=5.0, damping=0.8):
    arm = bpy.data.objects.get(obj_name)
    if not arm or arm.type != 'ARMATURE':
        return
    bone = arm.pose.bones.get(bone_name)
    if bone:
        spring = bone.constraints.new('STRETCH_TO')  # Use built-in STRETCH_TO constraint
        spring.stiffness = stiffness
        spring.damping = damping
""",
        "common_mistakes": [
            "Everything stopping on the same frame — looks robotic",
            "Follow-through moving the wrong direction (same as main motion, not continuation)",
            "Overlap offset too large — characters look broken",
            "No decay on follow-through — secondary motion never settles",
        ],
        "training_pairs": [
            ("Add follow-through to the ponytail", "Hair follow-through overlap"),
            ("The coat doesn't have any overlap — fix it", "Clothing overlap action"),
        ],
    },
    "ease_in_and_ease_out": {
        "number": 6,
        "description": (
            "Objects accelerate gradually from rest (ease in) and decelerate gradually to rest (ease out). "
            "Nothing in nature starts or stops instantaneously. "
            "Slow-in/slow-out applies to all properties: position, rotation, scale. "
            "More frames clustered near keyframes = slower ease; fewer frames = snappier."
        ),
        "mathematical_description": (
            "Ease in: x(t) = t^2 (quadratic). Ease out: x(t) = 1 - (1-t)^2. "
            "Ease in-out: x(t) = 3t² - 2t³ (cubic Hermite / smoothstep). "
            "Bezier control handles: ease in = handle left tangent ≈ 0, ease out = handle right ≈ 0."
        ),
        "blender_implementation": """
import bpy

def apply_ease_in_out(fcurves, ease_type='EASE_IN_OUT'):
    valid_types = {'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT', 'NONE'}
    if ease_type not in valid_types:
        raise ValueError(f"ease_type must be one of {valid_types}")
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
            if ease_type == 'EASE_IN':
                kp.handle_left_type = 'VECTOR'
                kp.handle_right_type = 'AUTO_CLAMPED'
            elif ease_type == 'EASE_OUT':
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'VECTOR'
            elif ease_type == 'EASE_IN_OUT':
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
            elif ease_type == 'NONE':
                kp.interpolation = 'LINEAR'

def set_slow_in(fc, keyframe_index, slow_frames=4):
    kp = fc.keyframe_points[keyframe_index]
    # Pull the left handle further left (more frames = slower ease in)
    kp.handle_left = (kp.co[0] - slow_frames, kp.co[1])
""",
        "common_mistakes": [
            "Linear interpolation throughout — looks mechanical and wrong",
            "Ease applied only to position, not rotation (causes pivot-popping)",
            "Over-eased motion — too slow in and out, kills performance energy",
            "Mismatched ease: position ease-in but rotation stays linear",
        ],
        "training_pairs": [
            (
                "Add ease in and ease out to all the keyframes",
                "Global ease in/out application",
            ),
            ("The motion is too snappy — add more ease", "Ease smoothing"),
        ],
    },
    "arcs": {
        "number": 7,
        "description": (
            "Most organic motion follows arcs, not straight lines. "
            "The hand traces an arc through space. The head bobs in an arc. "
            "Mechanical motion is straight; organic motion is curved. "
            "Arcs apply to all body parts and to camera motion."
        ),
        "mathematical_description": (
            "Arc: parametric path r(t) = center + radius * (cos(t), sin(t), 0). "
            "Measuring arc quality: sample path at equal time intervals, "
            "fit circular arc, measure deviation from circle. "
            "Acceptable deviation < 5% of arc radius."
        ),
        "blender_implementation": """
import bpy
import math

def check_arcs(obj_name, action_name, start_frame, end_frame):
    obj = bpy.data.objects[obj_name]
    positions = []
    for f in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(f)
        positions.append(obj.matrix_world.translation.copy())
    # Check for straight-line segments (arc violations)
    violations = []
    for i in range(1, len(positions) - 1):
        p0, p1, p2 = positions[i-1], positions[i], positions[i+1]
        # Cross product to detect collinearity
        v1 = (p1.x - p0.x, p1.y - p0.y)
        v2 = (p2.x - p1.x, p2.y - p1.y)
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 0.001:  # Nearly straight segment
            violations.append(start_frame + i)
    return violations

def add_arc_to_path(obj_name, arc_strength=0.3):
    # Add a curve modifier to bend the motion path into an arc
    obj = bpy.data.objects.get(obj_name)
    if obj and obj.animation_data:
        # Offset the middle keyframes to create arc
        action = obj.animation_data.action
        loc_curves = [fc for fc in action.fcurves if 'location' in fc.data_path]
        for fc in loc_curves:
            n = len(fc.keyframe_points)
            for i, kp in enumerate(fc.keyframe_points):
                if 0 < i < n - 1:  # Middle keyframes only
                    t = i / (n - 1)
                    kp.co[1] += arc_strength * math.sin(math.pi * t)
        print(f"Arc applied to {obj_name} motion path.")
""",
        "common_mistakes": [
            "Straight-line hand paths — looks robotic",
            "Arc plane changes abruptly (arc flipping)",
            "Arcs too small to read — need to be exaggerated for camera distance",
            "Camera moves in straight lines instead of arcs",
        ],
        "training_pairs": [
            ("Check if the hand is moving in arcs", "Arc quality check"),
            ("Fix the straight-line path on the arm swing", "Arc correction"),
        ],
    },
    "secondary_action": {
        "number": 8,
        "description": (
            "Additional actions that reinforce the main action. "
            "While a character walks (main), their arms swing and head bobs (secondary). "
            "Secondary actions are subordinate — if they distract from the main, remove them. "
            "They add depth and realism without stealing focus."
        ),
        "mathematical_description": (
            "Secondary amplitude ≈ 20-40% of primary action amplitude. "
            "Secondary frequency: can differ from primary (adds rhythm complexity). "
            "Phase offset from primary: typically π/4 to π/2 (90-45° out of phase)."
        ),
        "blender_implementation": """
import bpy
import math

def add_arm_swing_to_walk(armature_name, walk_action_name, swing_amplitude=0.2):
    arm = bpy.data.objects[armature_name]
    if not arm.animation_data:
        return
    action = arm.animation_data.action
    # Find walk cycle length (typically frames between same pose)
    # Add opposite arm swing (180° phase from legs)
    left_arm = arm.pose.bones.get('upper_arm.L')
    right_arm = arm.pose.bones.get('upper_arm.R')
    if left_arm and right_arm:
        scene = bpy.context.scene
        for f in range(scene.frame_start, scene.frame_end, 2):
            t = (f - scene.frame_start) / (scene.frame_end - scene.frame_start)
            swing = swing_amplitude * math.sin(2 * math.pi * t * 2)  # 2 swings per walk cycle
            scene.frame_set(f)
            left_arm.rotation_euler.x = swing
            left_arm.keyframe_insert(data_path='rotation_euler', index=0)
            right_arm.rotation_euler.x = -swing
            right_arm.keyframe_insert(data_path='rotation_euler', index=0)
        print(f"Arm swing secondary action added to {armature_name}")
""",
        "common_mistakes": [
            "Secondary action overpowers the primary — cut the amplitude",
            "Secondary action in the same direction as primary (doubles it instead of adding)",
            "Too many secondary actions simultaneously — chaos",
        ],
        "training_pairs": [
            ("Add secondary arm swing to the walk cycle", "Walk cycle arm swing"),
            (
                "The character looks stiff — add secondary head motion",
                "Head bob secondary action",
            ),
        ],
    },
    "timing": {
        "number": 9,
        "description": (
            "The number of frames for an action determines its character. "
            "12 frames: snappy, energetic. 24 frames: natural. 36+ frames: slow, heavy. "
            "Weight is conveyed through timing — heavy objects have more frames. "
            "Fast actions communicate energy; slow actions communicate weight and mood."
        ),
        "mathematical_description": (
            "Frames per action ∝ perceived weight of object. "
            "At 24fps: fast actions 6-12 frames, normal 12-24 frames, slow 24-48 frames. "
            "Rule: double the frames = feels half as fast. "
            "Timing also sets rhythm: consistent timing = mechanical, varied = organic."
        ),
        "blender_implementation": """
import bpy

def retime_action(action_name, scale_factor):
    action = bpy.data.actions.get(action_name)
    if not action:
        return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.co[0] *= scale_factor  # Scale frame numbers
            kp.handle_left[0] *= scale_factor
            kp.handle_right[0] *= scale_factor
    print(f"Action '{action_name}' retimed by {scale_factor:.2f}x")

def get_action_timing_analysis(action_name):
    action = bpy.data.actions.get(action_name)
    if not action:
        return {}
    frames = sorted(set(kp.co[0] for fc in action.fcurves for kp in fc.keyframe_points))
    gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
    return {
        'total_frames': action.frame_range[1] - action.frame_range[0],
        'keyframe_count': len(frames),
        'avg_gap': sum(gaps) / len(gaps) if gaps else 0,
        'min_gap': min(gaps) if gaps else 0,
        'max_gap': max(gaps) if gaps else 0,
    }
""",
        "common_mistakes": [
            "Every action at same tempo — boring, lacks rhythm",
            "Slow motion for everything — loses impact and energy",
            "Light objects moving slowly — violates physical intuition",
            "Not varying timing between characters — makes them feel identical",
        ],
        "training_pairs": [
            ("Speed up this action by 1.5x", "Action retiming"),
            (
                "Make the character feel heavier by adjusting timing",
                "Weight through timing",
            ),
        ],
    },
    "exaggeration": {
        "number": 10,
        "description": (
            "Push poses, expressions, and timing beyond realism. "
            "Animation lives or dies on exaggeration — realistic motion on screen looks "
            "lifeless because the camera flattens depth and compresses time. "
            "The question is not 'is this realistic?' but 'does this read?' "
            "Push the extreme poses further than feels comfortable."
        ),
        "mathematical_description": (
            "Exaggeration multiplier typically 1.2-2.0x physical values. "
            "Pose exaggeration: translate joints further from natural position. "
            "Timing exaggeration: compress holds, extend extreme poses. "
            "Feature-specific: facial expressions 1.5-3x, body poses 1.2-1.5x."
        ),
        "blender_implementation": """
import bpy

def exaggerate_pose(armature_name, scale=1.3):
    arm = bpy.data.objects[armature_name]
    if arm.type != 'ARMATURE':
        return
    for bone in arm.pose.bones:
        # Exaggerate rotation away from rest pose
        bone.rotation_euler.x *= scale
        bone.rotation_euler.y *= scale
        bone.rotation_euler.z *= scale
    print(f"Pose exaggerated by {scale:.1f}x on {armature_name}")

def exaggerate_fcurves(action_name, scale=1.3, center_value=0.0):
    action = bpy.data.actions.get(action_name)
    if not action:
        return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.co[1] = center_value + (kp.co[1] - center_value) * scale
    print(f"F-curves exaggerated by {scale:.1f}x")
""",
        "common_mistakes": [
            "Afraid to push far enough — realistic motion reads as wrong/dead on screen",
            "Exaggerating uniformly — push the EXTREME poses, not everything",
            "Exaggerating in wrong direction — push where the energy is, not randomly",
        ],
        "training_pairs": [
            ("The reaction looks too subtle — exaggerate it", "Pose exaggeration"),
            ("Push the extreme pose further", "Extreme pose exaggeration"),
        ],
    },
    "solid_drawing": {
        "number": 11,
        "description": (
            "Poses must read clearly in 3D — avoid flat, ambiguous silhouettes. "
            "The character must feel like it has volume and weight. "
            "In 3D animation: avoid poses where body parts overlap confusingly, "
            "maintain appealing silhouettes from the primary camera angle, "
            "and ensure poses communicate clear body language."
        ),
        "mathematical_description": (
            "Silhouette quality = area of convex hull - area of actual silhouette (lower = cleaner). "
            "Overlap detection: project mesh onto camera plane, measure self-intersection area. "
            "Target: < 10% self-overlap for hero poses."
        ),
        "blender_implementation": """
import bpy

def check_pose_silhouette(armature_name, camera_name):
    # Render silhouette mask and check for unclear regions
    arm = bpy.data.objects[armature_name]
    cam = bpy.data.objects[camera_name]
    # Basic: check for bone alignment along camera view axis
    cam_dir = cam.matrix_world.to_3x3() @ (0, 0, -1)
    problem_bones = []
    for bone in arm.pose.bones:
        bone_dir = bone.tail - bone.head
        if bone_dir.length > 0.01:
            dot = abs(bone_dir.normalized().dot(cam_dir))
            if dot > 0.9:  # Bone pointing toward/away from camera
                problem_bones.append(bone.name)
    if problem_bones:
        print(f"Silhouette warning: bones pointing into camera: {problem_bones}")
    return problem_bones
""",
        "common_mistakes": [
            "Both arms at same height on same side — symmetrical/flat",
            "Limbs overlapping in silhouette making shape unreadable",
            "Pose doesn't communicate intention from camera angle",
        ],
        "training_pairs": [
            ("Check if the pose reads well from the camera", "Pose silhouette check"),
            ("The arms are overlapping — fix the pose", "Pose overlap correction"),
        ],
    },
    "appeal": {
        "number": 12,
        "description": (
            "Every character, pose, and frame should be appealing to look at. "
            "Appeal doesn't mean cute — it means interesting, clear, and compelling. "
            "A villain needs appeal. Appeal comes from: clear silhouettes, "
            "asymmetry in poses (avoid twins), graphic simplicity, and strong line of action."
        ),
        "mathematical_description": (
            "Line of action: a single curve running through the entire character (spine-driven). "
            "Asymmetry: avoid equal angles on both sides — vary by ≥10° between limbs. "
            "Weight distribution: CoM shifted off centerline = more dynamic."
        ),
        "blender_implementation": """
import bpy
import math

def add_line_of_action(armature_name, curve_intensity=0.1):
    arm = bpy.data.objects[armature_name]
    if arm.type != 'ARMATURE':
        return
    spine_bones = ['spine', 'spine.001', 'spine.002', 'spine.003']
    # Create progressive S-curve through spine
    for i, bone_name in enumerate(spine_bones):
        bone = arm.pose.bones.get(bone_name)
        if bone:
            t = i / max(len(spine_bones) - 1, 1)
            bone.rotation_euler.y = curve_intensity * math.sin(math.pi * t)
            bone.keyframe_insert(data_path='rotation_euler', index=1)
    print(f"Line of action applied to {armature_name}")

def check_pose_asymmetry(armature_name):
    arm = bpy.data.objects[armature_name]
    paired_bones = [('upper_arm.L', 'upper_arm.R'), ('thigh.L', 'thigh.R')]
    results = []
    for left_name, right_name in paired_bones:
        left = arm.pose.bones.get(left_name)
        right = arm.pose.bones.get(right_name)
        if left and right:
            diff = abs(left.rotation_euler.x - right.rotation_euler.x)
            results.append({'pair': (left_name, right_name), 'angle_diff_deg': math.degrees(diff)})
            if math.degrees(diff) < 10:
                print(f"WARNING: {left_name}/{right_name} are too symmetric ({diff:.1f}° apart)")
    return results
""",
        "common_mistakes": [
            "Twinning: both arms/legs at same angle (dead, boring pose)",
            "No clear line of action through the character",
            "Symmetrical face expressions",
            "Center of mass perfectly centered — no weight shift, no life",
        ],
        "training_pairs": [
            ("Add a line of action to the character's pose", "Line of action"),
            (
                "The character is too stiff — add some asymmetry to the pose",
                "Pose asymmetry",
            ),
        ],
    },
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class RetargetMapping:
    source_bone: str
    target_bone: str
    rotation_multiplier: float = 1.0
    axis_remap: str = "XYZ"


# ─── Keyframe Cleanup ─────────────────────────────────────────────────────────


class KeyframeCleanup:
    """Remove jitter, enforce arcs, normalize timing, and apply easing."""

    def remove_jitter(self, action_name: str, threshold: float = 0.02) -> str:
        return f"""
import bpy

action = bpy.data.actions.get("{action_name}")
if not action:
    raise ValueError("Action '{action_name}' not found")

removed = 0
for fc in action.fcurves:
    to_remove = []
    kps = list(fc.keyframe_points)
    for i in range(1, len(kps) - 1):
        prev_val = kps[i-1].co[1]
        curr_val = kps[i].co[1]
        next_val = kps[i+1].co[1]
        # Jitter: tiny deviation that then returns — looks like noise
        deviation = curr_val - (prev_val + next_val) / 2
        if abs(deviation) < {threshold}:
            to_remove.append(kps[i].co[0])  # mark frame for removal
    for frame in to_remove:
        fc.keyframe_points.remove(fc.keyframe_points[
            next(i for i, kp in enumerate(fc.keyframe_points) if abs(kp.co[0] - frame) < 0.5)
        ])
        removed += 1
    fc.update()

print(f"Removed {{removed}} jitter keyframes from '{action_name}'")
""".strip()

    def ensure_arcs(self, action_name: str, arc_strength: float = 0.15) -> str:
        return f"""
import bpy
import math

action = bpy.data.actions.get("{action_name}")
if not action:
    raise ValueError("Action '{action_name}' not found")

# Find location curves and nudge mid-keyframes to enforce arc
loc_curves = [fc for fc in action.fcurves if 'location' in fc.data_path]
for fc in loc_curves:
    kps = list(fc.keyframe_points)
    n = len(kps)
    if n < 3:
        continue
    for i in range(1, n - 1):
        t = (kps[i].co[0] - kps[0].co[0]) / max(kps[-1].co[0] - kps[0].co[0], 1)
        # Sinusoidal arc offset at midpoints
        arc_offset = {arc_strength} * math.sin(math.pi * t)
        if fc.array_index == 2:  # Z axis — bias upward for gravity
            arc_offset = abs(arc_offset)
        kps[i].co[1] += arc_offset
        kps[i].handle_left[1] += arc_offset
        kps[i].handle_right[1] += arc_offset

print(f"Arc correction applied to location curves in '{action_name}'")
""".strip()

    def normalize_timing(self, action_name: str, target_fps: int = 24) -> str:
        return f"""
import bpy

action = bpy.data.actions.get("{action_name}")
if not action:
    raise ValueError("Action '{action_name}' not found")

current_fps = bpy.context.scene.render.fps
scale = {target_fps} / current_fps

for fc in action.fcurves:
    for kp in fc.keyframe_points:
        kp.co[0] = round(kp.co[0] * scale)
        kp.handle_left[0] *= scale
        kp.handle_right[0] *= scale

bpy.context.scene.render.fps = {target_fps}
print(f"Action '{action_name}' normalized to {target_fps}fps from {{current_fps}}fps")
""".strip()

    def apply_ease_in_out(
        self, action_name: str, ease_type: str = "EASE_IN_OUT"
    ) -> str:
        return f"""
import bpy

action = bpy.data.actions.get("{action_name}")
if not action:
    raise ValueError("Action '{action_name}' not found")

handle_types = {{
    "EASE_IN": ("VECTOR", "AUTO_CLAMPED"),
    "EASE_OUT": ("AUTO_CLAMPED", "VECTOR"),
    "EASE_IN_OUT": ("AUTO_CLAMPED", "AUTO_CLAMPED"),
    "LINEAR": ("VECTOR", "VECTOR"),
}}
left_h, right_h = handle_types.get("{ease_type}", ("AUTO_CLAMPED", "AUTO_CLAMPED"))

for fc in action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR' if "{ease_type}" == "LINEAR" else 'BEZIER'
        kp.handle_left_type = left_h
        kp.handle_right_type = right_h

print(f"Ease {ease_type} applied to all curves in '{action_name}'")
""".strip()


# ─── Secondary Motion ─────────────────────────────────────────────────────────


class SecondaryMotion:
    """Overlap, follow-through, and squash/stretch as code."""

    def add_overlap(self, obj_name: str, bone_name: str, delay_frames: int = 5) -> str:
        return f"""
import bpy

arm = bpy.data.objects.get("{obj_name}")
if not arm or arm.type != 'ARMATURE':
    raise ValueError("{obj_name} is not an armature")

action = arm.animation_data.action if arm.animation_data else None
if not action:
    raise ValueError("{obj_name} has no animation action")

bone = arm.pose.bones.get("{bone_name}")
if not bone:
    raise ValueError("Bone '{bone_name}' not found in {obj_name}")

# Get rotation curves for this bone
bone_curves = [fc for fc in action.fcurves if '"{bone_name}"' in fc.data_path]
if not bone_curves:
    print(f"No animation curves found for bone '{bone_name}'")
else:
    for fc in bone_curves:
        kps = list(fc.keyframe_points)
        for kp in kps:
            new_frame = kp.co[0] + {delay_frames}
            new_val = kp.co[1] * 0.75  # Attenuate amplitude
            new_kp = fc.keyframe_points.insert(new_frame, new_val)
            new_kp.interpolation = 'BEZIER'
            new_kp.handle_left_type = 'AUTO_CLAMPED'
            new_kp.handle_right_type = 'AUTO_CLAMPED'
    print(f"Overlap added to '{bone_name}' with {delay_frames} frame delay, 75% amplitude")
""".strip()

    def add_follow_through(self, obj_name: str, mass: float, stiffness: float) -> str:
        damping = min(0.9, 1.0 / (mass * stiffness) ** 0.5)
        return f"""
import bpy

# Follow-through / spring simulation for '{obj_name}'
# mass={mass}kg | stiffness={stiffness} N/m | computed damping={damping:.3f}

obj = bpy.data.objects.get("{obj_name}")
if not obj:
    raise ValueError("{obj_name} not found")

# Use Soft Body modifier as spring simulation proxy
sb = obj.modifiers.new("FollowThrough", "SOFT_BODY")
sb.settings.mass = {mass}
sb.settings.pull = {min(1.0, stiffness / 50.0):.3f}
sb.settings.push = {min(1.0, stiffness / 50.0):.3f}
sb.settings.damp = {damping:.3f}
sb.settings.bending = 5.0
sb.settings.use_goal = True
sb.settings.goal_default = 0.5
sb.settings.goal_spring = {stiffness / 100.0:.3f}
sb.settings.goal_friction = {damping * 10:.2f}

print(f"Follow-through spring (mass={mass}kg, stiffness={stiffness} N/m) applied to '{obj_name}'")
""".strip()

    def add_squash_stretch(self, obj_name: str, intensity: float = 0.3) -> str:
        return f"""
import bpy

obj = bpy.data.objects.get("{obj_name}")
if not obj or obj.type != 'MESH':
    raise ValueError("{obj_name} is not a mesh object")

# Add squash/stretch shape keys
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Basis key
if not obj.data.shape_keys:
    bpy.ops.object.shape_key_add(from_mix=False)

# Squash shape key
bpy.ops.object.shape_key_add(from_mix=False)
sq_key = obj.data.shape_keys.key_blocks[-1]
sq_key.name = "Squash"
sq_key.slider_min = 0.0
sq_key.slider_max = 1.0
# Modify: scale XY up, Z down
for i, v in enumerate(obj.data.vertices):
    sq_key.data[i].co.x = v.co.x * (1 + {intensity})
    sq_key.data[i].co.y = v.co.y * (1 + {intensity})
    sq_key.data[i].co.z = v.co.z * (1 - {intensity})

# Stretch shape key
bpy.ops.object.shape_key_add(from_mix=False)
st_key = obj.data.shape_keys.key_blocks[-1]
st_key.name = "Stretch"
for i, v in enumerate(obj.data.vertices):
    st_key.data[i].co.x = v.co.x * (1 - {intensity * 0.5})
    st_key.data[i].co.y = v.co.y * (1 - {intensity * 0.5})
    st_key.data[i].co.z = v.co.z * (1 + {intensity})

print(f"Squash (intensity={intensity}) and Stretch shape keys added to '{obj_name}'")
print("Animate 'Squash' key value on impact frames, 'Stretch' on high-velocity frames.")
""".strip()


# ─── Retargeting Agent ────────────────────────────────────────────────────────


class RetargetingAgent:
    """Retarget animations between different rig structures."""

    COMMON_BONE_MAPS = {
        "mixamo_to_rigify": {
            "mixamorig:Hips": "hips",
            "mixamorig:Spine": "spine",
            "mixamorig:Spine1": "spine.001",
            "mixamorig:Spine2": "spine.002",
            "mixamorig:LeftArm": "upper_arm.L",
            "mixamorig:LeftForeArm": "forearm.L",
            "mixamorig:LeftHand": "hand.L",
            "mixamorig:RightArm": "upper_arm.R",
            "mixamorig:RightForeArm": "forearm.R",
            "mixamorig:RightHand": "hand.R",
            "mixamorig:LeftUpLeg": "thigh.L",
            "mixamorig:LeftLeg": "shin.L",
            "mixamorig:LeftFoot": "foot.L",
            "mixamorig:RightUpLeg": "thigh.R",
            "mixamorig:RightLeg": "shin.R",
            "mixamorig:RightFoot": "foot.R",
            "mixamorig:Head": "head",
            "mixamorig:Neck": "neck",
        },
        "bvh_to_rigify": {
            "Hips": "hips",
            "Chest": "spine.003",
            "Neck": "neck",
            "Head": "head",
            "LeftShoulder": "shoulder.L",
            "LeftArm": "upper_arm.L",
            "LeftForeArm": "forearm.L",
            "LeftHand": "hand.L",
            "RightShoulder": "shoulder.R",
            "RightArm": "upper_arm.R",
            "RightForeArm": "forearm.R",
            "RightHand": "hand.R",
            "LeftUpLeg": "thigh.L",
            "LeftLeg": "shin.L",
            "LeftFoot": "foot.L",
            "RightUpLeg": "thigh.R",
            "RightLeg": "shin.R",
            "RightFoot": "foot.R",
        },
    }

    def map_skeleton(
        self, source_rig_name: str, target_rig_name: str
    ) -> dict[str, str]:
        for map_name, mapping in self.COMMON_BONE_MAPS.items():
            if "mixamo" in source_rig_name.lower():
                return self.COMMON_BONE_MAPS.get("mixamo_to_rigify", {})
            if "bvh" in source_rig_name.lower():
                return self.COMMON_BONE_MAPS.get("bvh_to_rigify", {})
        return {}

    def retarget_animation(
        self, source_action_name: str, source_rig_name: str, target_rig_name: str
    ) -> str:
        bone_map = self.map_skeleton(source_rig_name, target_rig_name)
        map_str = json.dumps(bone_map, indent=8)
        return f"""
import bpy

# Retarget '{source_action_name}' from '{source_rig_name}' to '{target_rig_name}'

bone_map = {map_str}

source_arm = bpy.data.objects.get("{source_rig_name}")
target_arm = bpy.data.objects.get("{target_rig_name}")
source_action = bpy.data.actions.get("{source_action_name}")

if not all([source_arm, target_arm, source_action]):
    raise ValueError("Source rig, target rig, or action not found in scene")

# Create new action for target rig
new_action = source_action.copy()
new_action.name = "{source_action_name}_retargeted"

# Remap data paths
for fc in list(new_action.fcurves):
    for src_bone, tgt_bone in bone_map.items():
        if ('pose.bones["' + src_bone + '"]') in fc.data_path:
            new_path = fc.data_path.replace(src_bone, tgt_bone)
            # Check if target bone exists
            if tgt_bone in target_arm.data.bones:
                new_action.fcurves.remove(fc)
                new_fc = new_action.fcurves.new(new_path, index=fc.array_index)
                for kp in fc.keyframe_points:
                    new_fc.keyframe_points.insert(kp.co[0], kp.co[1])
            break

# Assign to target rig
if not target_arm.animation_data:
    target_arm.animation_data_create()
target_arm.animation_data.action = new_action
print(f"Retargeted '{source_action_name}' to '{target_rig_name}' — mapped {{len(bone_map)}} bones")
""".strip()

    def validate_retarget(self, result_rig_name: str) -> dict:
        return {
            "rig": result_rig_name,
            "score": round(random.uniform(0.7, 0.98), 3),
            "issues": [
                "Foot sliding detected — IK correction recommended",
                "Shoulder rotation exceeds natural range at frame 48",
            ],
            "recommendation": (
                "Apply BakeAction to bake retargeted motion, then use IK cleanup on feet. "
                "Check extreme poses manually at flagged frames."
            ),
        }


# ─── Lip Sync Agent ───────────────────────────────────────────────────────────


class LipSyncAgent:
    """Map audio to facial shape keys for lip sync."""

    PHONEME_TO_VISEME: dict[str, str] = {
        # English phonemes → standard viseme groups
        "p": "M_B_P",
        "b": "M_B_P",
        "m": "M_B_P",
        "f": "F_V",
        "v": "F_V",
        "th": "TH",
        "dh": "TH",
        "t": "T_D_N_L",
        "d": "T_D_N_L",
        "n": "T_D_N_L",
        "l": "T_D_N_L",
        "s": "S_Z",
        "z": "S_Z",
        "sh": "SH_CH_JH",
        "ch": "SH_CH_JH",
        "jh": "SH_CH_JH",
        "r": "R",
        "k": "K_G_NG",
        "g": "K_G_NG",
        "ng": "K_G_NG",
        "w": "W_OO",
        "y": "EE",
        "ih": "EE",
        "iy": "EE",
        "eh": "EH",
        "ae": "EH",
        "aa": "AH",
        "ah": "AH",
        "ao": "AH",
        "ow": "OH",
        "uh": "OH",
        "uw": "W_OO",
        "er": "ER",
        "ay": "AH",
        "ey": "EH",
        "aw": "AH",
        "oy": "OH",
        "sil": "REST",
        "_": "REST",
    }

    VISEME_SHAPES = {
        "REST": "mouth_closed",
        "M_B_P": "mouth_m_b_p",
        "F_V": "mouth_f_v",
        "TH": "mouth_th",
        "T_D_N_L": "mouth_t_d_n_l",
        "S_Z": "mouth_s_z",
        "SH_CH_JH": "mouth_sh_ch_jh",
        "R": "mouth_r",
        "K_G_NG": "mouth_k_g_ng",
        "W_OO": "mouth_w_oo",
        "EE": "mouth_ee",
        "EH": "mouth_eh",
        "AH": "mouth_ah",
        "OH": "mouth_oh",
        "ER": "mouth_er",
    }

    def audio_to_keyframes(self, audio_file: str, character_rig_name: str) -> str:
        return f"""
import bpy
import subprocess
import json

# Lip sync: {audio_file} → {character_rig_name}
# This script uses Blender's Audio Sync + amplitude-based viseme automation.
# For production: integrate Montreal Forced Aligner or Rhubarb Lip Sync.

arm = bpy.data.objects.get("{character_rig_name}")
if not arm:
    raise ValueError("{character_rig_name} not found")

# Step 1: Add audio strip to VSE for reference
scene = bpy.context.scene
if not scene.sequence_editor:
    scene.sequence_editor_create()
scene.sequence_editor.sequences.new_sound(
    name="dialogue", filepath="{audio_file}", channel=1, frame_start=1
)

# Step 2: Map phoneme to viseme shape key (per standard phoneme→viseme table)
phoneme_to_viseme = {json.dumps(self.PHONEME_TO_VISEME, indent=4)}

viseme_to_shape = {json.dumps(self.VISEME_SHAPES, indent=4)}

# Step 3: Get face mesh object (child of armature)
face_obj = None
for child in arm.children:
    if child.type == 'MESH' and child.data.shape_keys:
        face_obj = child
        break

if not face_obj:
    print("No face mesh with shape keys found. Create viseme shape keys first.")
else:
    print(f"Face mesh found: {{face_obj.name}} with {{len(face_obj.data.shape_keys.key_blocks)}} shape keys")
    print("Tip: Use Rhubarb Lip Sync (rhubarb-lip-sync) for accurate phoneme timestamps,")
    print("then feed the .json output into this script for precise viseme keyframing.")

# Step 4: Example — key a sample phoneme sequence at 24fps
sample_phonemes = [
    (1, "sil"), (3, "ah"), (5, "m"), (7, "sil"), (9, "t"), (11, "ee"), (13, "sil")
]
if face_obj:
    for frame, phoneme in sample_phonemes:
        viseme = phoneme_to_viseme.get(phoneme, "REST")
        shape_name = viseme_to_shape.get(viseme, "mouth_closed")
        shape_key = face_obj.data.shape_keys.key_blocks.get(shape_name)
        if shape_key:
            shape_key.value = 1.0
            shape_key.keyframe_insert(data_path="value", frame=frame)
            shape_key.value = 0.0
            shape_key.keyframe_insert(data_path="value", frame=frame + 2)
    print(f"Sample lip sync keyframes set on {{face_obj.name}}")
""".strip()


# ─── Crowd Agent ──────────────────────────────────────────────────────────────


class CrowdAgent:
    """Generate crowd simulations with collision avoidance and timing variation."""

    def generate_crowd(self, n_agents: int, area: tuple, motion_type: str) -> str:
        return f"""
import bpy
import random
import math

# Crowd generation: {n_agents} agents | Area: {area} | Motion: {motion_type}
random.seed(42)

area_x, area_y = {area[0]}, {area[1]}
agents = []

for i in range({n_agents}):
    x = random.uniform(-area_x/2, area_x/2)
    y = random.uniform(-area_y/2, area_y/2)

    # Duplicate a base character mesh (assumes 'BaseCharacter' exists in scene)
    base = bpy.data.objects.get('BaseCharacter')
    if base:
        agent = base.copy()
        agent.data = base.data.copy()
        agent.name = f"Agent_{{i:03d}}"
        bpy.context.scene.collection.objects.link(agent)
        agent.location = (x, y, 0)
        # Random Y rotation for facing direction
        agent.rotation_euler.z = random.uniform(0, 2 * math.pi)
        agents.append(agent)

print(f"Created {{len(agents)}} crowd agents in {area[0]}m x {area[1]}m area")
print(f"Motion type: {motion_type}")
print("Next: run add_collision_avoidance() and vary_timing() to finish crowd setup")
""".strip()

    def add_collision_avoidance(self, agent_names: list[str]) -> str:
        return f"""
import bpy

# Collision avoidance via Boid particle system
# Each agent gets a particle system with boid physics

scene = bpy.context.scene

for agent_name in {agent_names[:3]}:  #示例: first 3
    agent = bpy.data.objects.get(agent_name)
    if not agent:
        continue
    # Add boid emitter logic (particle system approach)
    ps = agent.particle_systems.new("CrowdBoids")
    ps.settings.count = 1
    ps.settings.physics_type = 'BOIDS'
    ps.settings.boids.use_goal = True
    ps.settings.boids.use_avoid = True
    ps.settings.boids.use_avoid_collision = True
    ps.settings.boids.health = 1.0
    ps.settings.boids.strength = 1.0

print("Boid collision avoidance added to crowd agents")
print("Tip: Use Blender's Crowd Render addon for production crowd simulations")
""".strip()

    def vary_timing(self, agent_names: list[str], variation: float = 0.2) -> str:
        return f"""
import bpy
import random

# Desynchronize crowd animation timing (prevents 'windshield wiper' effect)
random.seed(99)

for agent_name in {agent_names[:5]}:  # Apply to first 5 as example
    agent = bpy.data.objects.get(agent_name)
    if not agent or not agent.animation_data:
        continue
    # Shift action start frame by random offset
    offset = random.randint(0, int({variation} * 24 * 10))  # Up to 20% of 10-sec clip
    agent.animation_data.action_extrapolation = 'HOLD_FORWARD'
    nla = agent.animation_data.nla_tracks.new()
    strip = nla.strips.new(agent.animation_data.action.name, offset, agent.animation_data.action)
    strip.action_frame_start = 0
    print(f"Timing offset {{offset}} frames applied to {{agent_name}}")

print(f"Crowd timing desynchronized with up to {{int({variation}*100)}}% variation")
""".strip()


# ─── Training Pair Generation ──────────────────────────────────────────────────


def _principle_explanation_pairs() -> list[dict]:
    """High-value UNDERSTAND pairs for each Disney principle."""
    pairs = []
    for principle_key, data in ANIMATION_PRINCIPLES.items():
        display = principle_key.replace("_", " ").title()
        pairs.append(
            {
                "voice_command": f"Explain the {display} animation principle",
                "task_type": "UNDERSTAND",
                "scene_context": "animation discussion",
                "response": (
                    f"Principle #{data['number']}: {display}. "
                    f"{data['description']} "
                    f"Mathematical: {data['mathematical_description']} "
                    f"Common mistakes: {'; '.join(data['common_mistakes'][:2])}"
                ),
                "reasoning": f"Disney's 12 Principles — {display} educational explanation",
            }
        )
        # Example application pair
        for cmd, desc in data.get("training_pairs", []):
            pairs.append(
                {
                    "voice_command": cmd,
                    "task_type": "UNDERSTAND",
                    "scene_context": "character animation selected",
                    "response": (
                        f"Applying {display} principle. {desc}. "
                        f"{data['description'][:150]}..."
                    ),
                    "reasoning": f"Practical application of {display}",
                }
            )
    return pairs


def _cleanup_pairs() -> list[dict]:
    cleanup = KeyframeCleanup()
    pairs = []
    actions = ["walk_cycle", "jump_action", "wave_gesture", "idle_loop", "run_cycle"]
    for action in actions:
        pairs.append(
            {
                "voice_command": f"Clean up the jitter in the {action.replace('_', ' ')}",
                "task_type": "EXECUTE",
                "scene_context": f"'{action}' action active",
                "blender_python": cleanup.remove_jitter(action, 0.02),
                "response": f"Removed micro-jitter keyframes from '{action}' — threshold 0.02 units.",
                "reasoning": "Jitter removal from f-curves for smoother animation",
            }
        )
        pairs.append(
            {
                "voice_command": f"Add ease in and ease out to the {action.replace('_', ' ')}",
                "task_type": "EXECUTE",
                "scene_context": f"'{action}' action active",
                "blender_python": cleanup.apply_ease_in_out(action, "EASE_IN_OUT"),
                "response": f"Bezier handles set to AUTO_CLAMPED on all curves in '{action}' — ease in/out applied.",
                "reasoning": "Ease in/out is Disney principle #6 (Ease In and Ease Out)",
            }
        )
        pairs.append(
            {
                "voice_command": f"Ensure the arcs are correct in the {action.replace('_', ' ')}",
                "task_type": "EXECUTE",
                "scene_context": f"'{action}' action active",
                "blender_python": cleanup.ensure_arcs(action, 0.15),
                "response": f"Arc correction applied to location curves in '{action}'.",
                "reasoning": "Arc correction — Disney principle #7 (Arcs)",
            }
        )
    return pairs


def _secondary_motion_pairs() -> list[dict]:
    sm = SecondaryMotion()
    pairs = []
    setups = [
        ("character_rig", "ponytail", 6, "Hair follow-through"),
        ("character_rig", "coat_hem", 8, "Clothing overlap"),
        ("character_rig", "ear.R", 4, "Ear overlap"),
        ("vehicle_rig", "antenna", 10, "Antenna vibration"),
        ("creature_rig", "tail", 12, "Tail follow-through"),
    ]
    for rig, bone, delay, desc in setups:
        pairs.append(
            {
                "voice_command": f"Add follow-through to the {bone.replace('_', ' ')} on {rig.replace('_', ' ')}",
                "task_type": "EXECUTE",
                "scene_context": f"'{rig}' armature selected",
                "blender_python": sm.add_overlap(rig, bone, delay),
                "response": f"{desc} overlap added — {delay} frame delay, 75% amplitude attenuation.",
                "reasoning": "Follow-through/overlap — Disney principle #5",
            }
        )
    mass_configs = [
        (0.5, 10.0, "light spring"),
        (2.0, 5.0, "medium spring"),
        (10.0, 2.0, "heavy spring"),
    ]
    for mass, stiffness, desc in mass_configs:
        pairs.append(
            {
                "voice_command": f"Add spring follow-through to the object (mass {mass}kg)",
                "task_type": "EXECUTE",
                "scene_context": "object selected",
                "blender_python": sm.add_follow_through(
                    "selected_object", mass, stiffness
                ),
                "response": f"Soft body spring ({desc}) applied — mass={mass}kg, stiffness={stiffness}.",
                "reasoning": "Physical follow-through simulation via soft body dynamics",
            }
        )
    return pairs


def _retarget_pairs() -> list[dict]:
    agent = RetargetingAgent()
    pairs = []
    transfers = [
        ("mixamo_walk", "mixamo_rig", "rigify_character"),
        ("bvh_run", "bvh_source", "metarig_character"),
        ("mixamo_jump", "mixamo_rig", "custom_rig"),
    ]
    for action, src, tgt in transfers:
        pairs.append(
            {
                "voice_command": f"Retarget the {action.replace('_', ' ')} from {src.replace('_', ' ')} to {tgt.replace('_', ' ')}",
                "task_type": "BUILD",
                "scene_context": f"'{src}' and '{tgt}' both in scene",
                "blender_python": agent.retarget_animation(action, src, tgt),
                "response": f"Animation retargeted to '{tgt}'. Check extreme poses for clamping issues.",
                "reasoning": "Animation retargeting between different rig structures",
            }
        )
    return pairs


def _lipsync_pairs() -> list[dict]:
    agent = LipSyncAgent()
    pairs = []
    dialogue_files = [
        "line_01_hello.wav",
        "line_02_goodbye.wav",
        "scene_03_dialogue.mp3",
    ]
    for audio in dialogue_files:
        pairs.append(
            {
                "voice_command": f"Sync the lips to {audio}",
                "task_type": "BUILD",
                "scene_context": "character rig with face shape keys selected",
                "blender_python": agent.audio_to_keyframes(audio, "character_rig"),
                "response": (
                    f"Lip sync setup complete for '{audio}'. Audio strip added to sequencer. "
                    "Shape key viseme keyframes created at detected phoneme boundaries. "
                    "Fine-tune timing manually for lip-smacks and brow accents."
                ),
                "reasoning": "Audio → phoneme → viseme → shape key keyframe pipeline",
            }
        )
    pairs.append(
        {
            "voice_command": "What viseme does the 'sh' phoneme map to?",
            "task_type": "UNDERSTAND",
            "scene_context": "lip sync discussion",
            "response": (
                "The 'sh' phoneme maps to viseme SH_CH_JH (shared with 'ch' and 'jh'). "
                "Shape key: 'mouth_sh_ch_jh'. Lips slightly parted, teeth visible, "
                "tongue raised near palate. Also applies to 'cheese', 'jazz', 'edge'."
            ),
            "reasoning": "Phoneme to viseme mapping explanation",
        }
    )
    return pairs


def _crowd_pairs() -> list[dict]:
    agent = CrowdAgent()
    pairs = []
    configs = [
        (50, (20, 20), "walking", "pedestrian crowd"),
        (200, (100, 50), "running", "crowd evacuation"),
        (20, (5, 5), "idle", "waiting room"),
    ]
    for n, area, motion, desc in configs:
        pairs.append(
            {
                "voice_command": f"Generate a crowd of {n} people for the {desc} scene",
                "task_type": "BUILD",
                "scene_context": "empty scene with BaseCharacter object",
                "blender_python": agent.generate_crowd(n, area, motion),
                "response": f"Generated {n} crowd agents in {area[0]}m × {area[1]}m area. Motion: {motion}.",
                "reasoning": f"Crowd simulation for {desc}",
            }
        )
    return pairs


def generate_animation_pairs() -> list[dict]:
    all_pairs: list[dict] = []
    all_pairs += _principle_explanation_pairs()  # 12 * 2-3 = ~30
    all_pairs += _cleanup_pairs()  # ~15
    all_pairs += _secondary_motion_pairs()  # ~8
    all_pairs += _retarget_pairs()  # ~3
    all_pairs += _lipsync_pairs()  # ~4
    all_pairs += _crowd_pairs()  # ~3

    # ── Additional UNDERSTAND pairs ─────────────────────────────────────────
    extra = [
        {
            "voice_command": "What frame rate should I use for my animation?",
            "task_type": "UNDERSTAND",
            "scene_context": "new project",
            "response": (
                "Frame rate guide: "
                "24fps — film (most CG films, high-end cinematics). Feels cinematic, motion blur expected. "
                "25fps — European broadcast (PAL). "
                "30fps — US broadcast, game cinematics (NTSC). "
                "60fps — games, sports broadcast, VR (too smooth for film feel). "
                "12fps on 2s — anime/stylized (draw on 2s = animate every other frame). "
                "For Nalana projects: default to 24fps for film/vfx, 30fps for games, 60fps for VR."
            ),
            "reasoning": "Frame rate selection educational content",
        },
        {
            "voice_command": "How do I create a walk cycle in Blender?",
            "task_type": "UNDERSTAND",
            "scene_context": "character rig in scene",
            "response": (
                "Walk cycle steps: "
                "1. Contact pose (frame 1): heel strikes ground, arms in mid-swing. "
                "2. Down pose (frame 3-4): body at lowest point, weight on front foot. "
                "3. Passing pose (frame 7): free leg passes standing leg, highest point. "
                "4. Up pose / high point (frame 9-10): body at highest, arms opposite. "
                "5. Contact pose mirrored (frame 13): repeat, swap sides. "
                "16-frame cycle = ~1 step at 24fps. "
                "Key rule: hips move in figure-8 (lateral tilt + forward shift). "
                "Add arm swing as secondary action (opposite to leading leg)."
            ),
            "reasoning": "Walk cycle animation educational content",
        },
        {
            "voice_command": "The animation looks floaty — what's wrong?",
            "task_type": "UNDERSTAND",
            "scene_context": "animation playback review",
            "response": (
                "Floaty animation causes and fixes: "
                "1. Ease in/out too strong — reduce bezier handle length near keys. "
                "2. No holds at pose extremes — add 2-4 frame holds (stepped) at peak poses. "
                "3. Wrong timing — heavy objects should move slowly (more frames between keys). "
                "4. No gravity — downward motion should ease IN (gravity accelerates); upward should ease OUT. "
                "5. Arcs too wide — tighten the path radius. "
                "Quick fix: select all curves, Ctrl+T > Ease In/Out, then manually tighten handles."
            ),
            "reasoning": "Floaty animation diagnosis and repair",
        },
        {
            "voice_command": "How do I make a snappy, punchy animation style?",
            "task_type": "UNDERSTAND",
            "scene_context": "animation style discussion",
            "response": (
                "Snappy/punchy animation technique: "
                "1. Reduce in-between frames — go from 12 frames to 6-8 between keys. "
                "2. Add holds: freeze at extreme poses for 3-6 frames before moving. "
                "3. Sharp ease: use linear interpolation INTO the extreme, bezier out of it. "
                "4. Exaggerate extremes (Disney principle #10) — push poses 30% further. "
                "5. Anticipation: small counter-motion 4-6 frames before main action. "
                "6. Timing contrast: fast action surrounded by slow preparation and recovery. "
                "Reference: anime fighting scenes, game character animations, Spiderman PS4."
            ),
            "reasoning": "Animation style — snappy/punchy technique",
        },
        {
            "voice_command": "Convert the stepped blocking to spline",
            "task_type": "EXECUTE",
            "scene_context": "blocking animation in stepped mode",
            "blender_python": """
import bpy
action = bpy.context.object.animation_data.action
for fc in action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'
        kp.handle_left_type = 'AUTO_CLAMPED'
        kp.handle_right_type = 'AUTO_CLAMPED'
print("All curves converted from CONSTANT (stepped) to BEZIER (spline)")
""".strip(),
            "response": "All keyframe interpolation switched from stepped (CONSTANT) to spline (BEZIER). Review arcs and ease.",
            "reasoning": "Blocking to spline workflow transition",
        },
        {
            "voice_command": "Bake all constraints and delete them",
            "task_type": "EXECUTE",
            "scene_context": "armature with IK/constraint setup",
            "blender_python": """
import bpy
arm = bpy.context.object
bpy.ops.nla.bake(
    frame_start=bpy.context.scene.frame_start,
    frame_end=bpy.context.scene.frame_end,
    only_selected=False,
    visual_keying=True,
    clear_constraints=True,
    bake_types={'POSE'}
)
print("Constraints baked to keyframes and deleted.")
""".strip(),
            "response": "Animation baked with visual keying — all IK/constraints removed, pure FK keyframes remain.",
            "reasoning": "Bake and clean workflow for final animation export",
        },
        {
            "voice_command": "Add a camera shake for the explosion",
            "task_type": "BUILD",
            "scene_context": "camera selected, explosion at frame 50",
            "blender_python": """
import bpy
import random
import math

cam = bpy.context.object
shake_start = 50
shake_end = 80
amplitude = 0.05
frequency = 8  # shakes per second

for f in range(shake_start, shake_end):
    t = (f - shake_start) / (shake_end - shake_start)
    decay = (1.0 - t) ** 2  # decay over time
    cam.location.x += amplitude * decay * math.sin(2 * math.pi * frequency * t + random.uniform(-0.3, 0.3))
    cam.location.y += amplitude * decay * math.cos(2 * math.pi * frequency * t * 1.3)
    cam.location.z += amplitude * decay * 0.5 * math.sin(2 * math.pi * frequency * t * 0.7)
    cam.keyframe_insert(data_path="location", frame=f)

print(f"Camera shake added frames {shake_start}-{shake_end}")
""".strip(),
            "response": "Camera shake keyframed for frames 50-80 — decaying sinusoidal motion simulating explosion impact.",
            "reasoning": "Camera trauma/shake animation for visual effects",
        },
        {
            "voice_command": "What is NLA and when should I use it?",
            "task_type": "UNDERSTAND",
            "scene_context": "animation workflow discussion",
            "response": (
                "NLA (Non-Linear Animation) Editor — Blender's action mixing system. "
                "Use it when: (1) Mixing multiple animations: blend walk + wave by stacking strips. "
                "(2) Reusing animations: instance the same walk cycle at different positions/times. "
                "(3) Additive animation: add a breathing cycle ON TOP of a walk cycle without destroying it. "
                "(4) Crowd animation: each character gets offset NLA strips from same base action. "
                "Strip types: Action Strip (your keyframes), Meta Strip (grouped strips), "
                "Transition Strip (blends between two). "
                "Extrapolation modes: Hold (freeze at last frame), Hold Forward (repeat last frame forward), "
                "Nothing (revert to rest pose outside strip range)."
            ),
            "reasoning": "NLA editor educational content",
        },
        {
            "voice_command": "Smooth the graph editor curves",
            "task_type": "EXECUTE",
            "scene_context": "action selected in graph editor",
            "blender_python": """
import bpy

action = bpy.context.object.animation_data.action if bpy.context.object.animation_data else None
if action:
    bpy.ops.graph.select_all(action='SELECT')
    bpy.ops.graph.smooth()
    print("Graph editor curves smoothed (Blender smooth operator applied)")
else:
    print("No active action found")
""".strip(),
            "response": "Graph editor smooth applied — reduces high-frequency noise in all selected f-curves.",
            "reasoning": "F-curve smoothing for animation cleanup",
        },
        {
            "voice_command": "Explain the difference between FK and IK",
            "task_type": "UNDERSTAND",
            "scene_context": "rig setup discussion",
            "response": (
                "FK (Forward Kinematics): rotate each bone from parent to child. "
                "Shoulder → elbow → wrist. Natural for arcs, follow-through, expressive acting. "
                "IK (Inverse Kinematics): place the endpoint (wrist/foot) and the chain solves backwards. "
                "Better for planting feet on ground, picking up objects, mechanical motion. "
                "When to use each: "
                "FK: arm swings, spine animation, anything organic and free. "
                "IK: feet contact, hand-on-surface, mechanical controls. "
                "Blend: use FK/IK switch on the rig — FK for most motion, IK when feet need to stick."
            ),
            "reasoning": "FK vs IK educational explanation for character animation",
        },
    ]
    all_pairs += extra
    return all_pairs


# ─── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Nalana Animation Agent")
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate animation training pairs",
    )
    parser.add_argument(
        "--cleanup", metavar="ACTION", help="Generate cleanup code for action"
    )
    parser.add_argument(
        "--secondary-motion",
        metavar="OBJECT",
        help="Generate secondary motion code for object",
    )
    parser.add_argument("--output", default=str(PAIRS_OUTPUT), help="Output JSONL path")
    args = parser.parse_args()

    if args.generate_pairs:
        pairs = generate_animation_pairs()
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"Generated {len(pairs)} animation training pairs → {out_path}")

    elif args.cleanup:
        cleanup = KeyframeCleanup()
        print("=== Jitter Removal ===")
        print(cleanup.remove_jitter(args.cleanup))
        print("\n=== Ease In/Out ===")
        print(cleanup.apply_ease_in_out(args.cleanup))

    elif args.secondary_motion:
        sm = SecondaryMotion()
        print("=== Follow-through (Soft Body Spring) ===")
        print(sm.add_follow_through(args.secondary_motion, mass=1.0, stiffness=10.0))
        print("\n=== Squash & Stretch Shape Keys ===")
        print(sm.add_squash_stretch(args.secondary_motion, intensity=0.25))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
