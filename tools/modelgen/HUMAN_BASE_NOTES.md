# Procedural Human Base — Lessons & Method

Hard-won notes from building the male base by script. Read before touching any body,
race, or the female base.

## Pipeline

1. **Skeleton** (`humanoid.build_skeleton`) — joint positions from height + params. The
   same joints become the Blender armature, so mesh and rig always agree.
2. **Base volumes** (`body.py`) — ONE continuous lofted surface per major part:
   torso (hips→neck), neck, upper arm, deltoid, forearm, a single leg loft (hip→ankle).
   Head is `anatomy._head` (locked — see below).
3. **Surface anatomy** (`muscles.py`) — every muscle/bony landmark is a sculpt-brush stroke
   (Gaussian buildup or crease) **projected onto the real skin** via sphere tracing.
4. **Hands & feet** (`extremities.py`) — lofted palm/foot body + lofted phalanges, pads,
   knuckles, nails, creases.
5. **Mesh** (`mesher.py`) — sparse block surface-nets + Newton projection. Watertight.
6. **Verify** — `compare.py` (clay renders side-by-side with the reference, outline overlay,
   upper/lower/pelvis crops) and `silcmp.py` (numeric silhouette widths per height).

## What failed and why (do not repeat)

| Mistake | Result | Fix |
|---|---|---|
| Body as a union of many ellipsoids/capsules with big smooth-min blends | "Meat sack": lumps everywhere, no clean silhouette | One loft per body part; blends only at real junctions |
| Smooth-min between left and right limbs | Thighs fused together | Side groups: L/R blend with the centre, never with each other (`Prim.side`) |
| Uncapped loft ends | Infinite columns | Always cap; rounded caps (shrink-then-inflate) |
| Separate thigh and shin lofts | "Doll joint" ring at knee | Single continuous leg loft; profile carries the knee |
| Muscles as unioned volumes | Shelves/ledges (pec shelf, side "wings") | Muscles are displacement brushes, not volumes |
| Brush centres placed by guessed coordinates | Strokes floating off / buried | Sphere-trace onto the base surface (`muscles.Painter.surf`) |
| Sphere-trace started outside every prim's bounds | Field returns placeholder 10 m → brushes flung away | Clamp tracer step size |
| Torso with a flat horizontal bottom + thighs with flat tops | "Underwear seam" / hips meet in a flat line | Torso narrows to a **wedge** toward the pubis; thigh tops **cut along the inguinal line** (crotch → hip bone) so pelvis and legs meet in a **V** |
| Narrowing the pelvis too much | Legs plug into the torso's corners | Keep a full pelvis; thighs start tucked inside it |
| Smooth blend radius at overlapping parallel surfaces | Adds ~k/4 of volume → hip/waist bulge | Small k where surfaces run parallel |
| Boxy superellipse (n≈2.4) torso cross-section | Flat back, square sides | Back n≈1.9–2.0; thoracic curve in the back profile |
| Measuring the jaw from the front silhouette | Chin as wide as the neck | Below the jaw angle the silhouette is the NECK — taper jaw→chin explicitly |
| Feet/hands from primitive blobs, claw-curled fingers | "Ass" hands/feet | Lofted palm/foot + lofted phalanges with real length ratios, relaxed pose |
| Forefoot loft extending past the toe knuckles | Toes swallowed → tiny claws | Forefoot ends at the MTP line; toes sit on top, full size |
| Large strokes on thin features near the mesher's coarse test | Missing blocks/specks | Coarse activity threshold 2.2× block size |

## Reference workflow (what worked)

- Segment the reference silhouettes (per-row background estimate + flood fill) and export
  **normalised width tables** (fractions of height). Never eyeball proportions.
- Compare in the **same pose** as the reference (arm drop ≈70° for the male sheet), then
  read the numbers per height with `silcmp.py`. Target: within ±0.005 H.
- Stance matters: the reference stands with knees ≈0.07 H, ankles ≈0.075 H off-centre.
  Width mismatches in legs were stance, not anatomy.
- Raking key light (high, from the side-front) is needed to judge relief; flat front light
  hides everything.
- The **user's eye beats the numbers**: silhouettes can match within 0.002 H and still read
  wrong if the internal structure (e.g. the pelvic V) is wrong. Fix structure, not values.

## Key measured male proportions (fractions of H)

Head top 1.0, chin ≈0.868, neck narrowest half-width ≈0.034, shoulder joints x≈0.102,
trapezius line: 0.86→0.061, 0.84→0.091, 0.82→0.122; deltoid outer 0.141–0.151;
waist (0.62–0.64) ≈0.086; hips 0.50→0.107; crotch ≈0.465; knee ≈0.28 (half-width 0.032);
calf max ≈0.21 (0.036); ankle 0.045. Inner thigh line: 0.46→0.004, 0.40→0.015,
0.36→0.027, 0.32→0.035 (this diagonal IS the leg V).

## Head (locked)

`anatomy._head` is lofted from the measured front/side head silhouettes (`ref_head.json`)
with the nose removed from the profile and rebuilt as its own form, eyes as
socket + lid shell + almond opening, ears from landmarks. The user approved it — change
only details, never the base shape, and keep a before/after comparison.

## Muscle checklist (male)

Torso front: pec (clavicular + sternal heads, lower border, sternal border), deltopectoral
groove, axillary fold, clavicles, costal arch, rectus abdominis segments + linea alba,
external obliques, serratus, inguinal V. Back: trapezius (upper/mid/lower), scapular spine,
infraspinatus, teres major, lats, erectors, spinal groove, lumbar dimples, glutes + fold +
cleft. Arms: deltoid (3 heads), biceps (2 heads), brachialis, triceps (long/lateral/tendon
plate), epicondyles, cubital fossa, brachioradialis, extensors, flexors, ulna ridge,
wrist. Legs: rectus femoris (from ASIS), TFL, vastus lateralis/medialis, sartorius, IT band,
adductors (from pubis), hamstrings, patella + tendon, popliteal fossa, gastrocnemius
(2 heads), soleus, tibialis anterior, tibial crest, peroneal groove, achilles.

## Female base — plan

Same skeleton/loft/brush machinery, `fem=1`. Re-measure from the female references
(front/back silhouettes, multi-angle sheet). Expect: narrower shoulders and ribcage,
wider pelvis (hips widest ≈ crotch level), smaller waist, thighs with more inner volume
(smaller thigh gap), breasts as volumes (not brushes), softer/fewer muscle strokes
(md scaled down), rounder glutes, slimmer arms/neck, smaller hands/feet.
