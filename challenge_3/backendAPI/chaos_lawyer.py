import random

CHAOS_TEMPLATES = [
    "Ah, but what if {case} was actually caused by invisible hamsters?",
    "In the case of {case}, clearly time travel could change everything!",
    "Imagine if {case} happened on Mars; then obviously the law doesn’t apply.",
    "Ah, but what if {case} was actually caused by mischievous gnomes?",
"Ah, but what if {case} was actually caused by a rogue AI?",
"Ah, but what if {case} was actually caused by cosmic rays?",
"Ah, but what if {case} was actually caused by poltergeist activity?",
"Ah, but what if {case} was actually caused by an alien prank?",
"Ah, but what if {case} was actually caused by sentient dust bunnies?",
"Ah, but what if {case} was actually caused by time-traveling tourists?",
"Ah, but what if {case} was actually caused by a forgotten wizard spell?",
"Ah, but what if {case} was actually caused by rogue garden gnomes?",
"Ah, but what if {case} was actually caused by quantum fluctuations?",
"Ah, but what if {case} was actually caused by invisible pixies?",
"Ah, but what if {case} was actually caused by a mischievous squirrel?",
"Ah, but what if {case} was actually caused by a rogue cloud?",
"Ah, but what if {case} was actually caused by malfunctioning robots?",
"Ah, but what if {case} was actually caused by spontaneous teleportation?",
"Ah, but what if {case} was actually caused by talking furniture?",
"Ah, but what if {case} was actually caused by a cursed painting?",
"Ah, but what if {case} was actually caused by an invisible unicorn?",
"Ah, but what if {case} was actually caused by mischievous fairies?",
"Ah, but what if {case} was actually caused by a sneaky ghost?",
"Ah, but what if {case} was actually caused by a magical storm?",
"Ah, but what if {case} was actually caused by a teleporting cat?",
"Ah, but what if {case} was actually caused by enchanted shoes?",
"Ah, but what if {case} was actually caused by rogue software updates?",
"Ah, but what if {case} was actually caused by spontaneous levitation?"

]

def chaos_response(case_text):
    template = random.choice(CHAOS_TEMPLATES)
    return {
        "argument": template.format(case=case_text),
        "rhetoric": "Exaggeration, loopholes, absurdity"
    }
