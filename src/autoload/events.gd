extends Node
## Global signal bus. Systems announce what happened here; UI and audio listen.
## Nothing in here holds state.

@warning_ignore_start("unused_signal")

# --- narration / feedback ---
signal log_message(text: String, kind: String)          # kind: info|hit|miss|heal|status|system|loot
signal toast(text: String, color: Color)
signal floating_text(world_pos: Vector3, text: String, color: Color)

# --- units ---
signal unit_changed(unit_state: RefCounted)             # hp / flame / statuses changed
signal unit_downed(unit_state: RefCounted)
signal unit_died(unit_state: RefCounted)
signal party_changed()

# --- game flow ---
signal combat_started()
signal combat_ended(victory: bool)
signal turn_started(unit_state: RefCounted)
signal selection_changed(unit_state: RefCounted)
signal area_entered(area_id: String)
signal dialogue_started(dialogue_id: String)
signal dialogue_ended(dialogue_id: String)

# --- progression / world state ---
signal flag_changed(flag: String, value: Variant)
signal approval_changed(companion_id: String, delta: int)
signal inventory_changed()
signal level_up_available(unit_state: RefCounted)
signal drips_changed(amount: int)

@warning_ignore_restore("unused_signal")
