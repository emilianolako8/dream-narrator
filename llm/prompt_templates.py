
print("FILE IS RUNNING")
"""
prompt_templates.py
-------------------
PURPOSE: Define multiple narrative styles for dream interpretation.

WHY THIS EXISTS:
A single prompt produces a single style.
Different users want different interpretations of their dreams.
A psychologist wants symbol analysis.
A writer wants poetic prose.
A neuroscientist wants biological explanation.

This file separates prompt logic from narrative building logic.
That's clean engineering — each file has one responsibility.

HOW TO ADD A NEW STYLE:
1. Add a new key to NARRATIVE_STYLES
2. Write a system prompt and a template
3. It automatically becomes available in the pipeline
"""


# ─────────────────────────────────────────────────────────
# NARRATIVE STYLES
# Each style has:
#   - system_prompt : tells the LLM what role to play
#   - template      : the actual prompt with {placeholders}
#   - description   : human readable explanation
# ─────────────────────────────────────────────────────────

NARRATIVE_STYLES = {

    # ── VIVID ──────────────────────────────────────────────
    # Default style. Immersive, cinematic, second person.
    # Best for general users who want to relive their dream.
    "vivid": {
        "description": "Immersive, cinematic dream narrative in second person",
        "system_prompt": """You are a vivid dream narrator who reconstructs 
dreams from neural decoded concept sequences. Your narratives are immersive, 
psychologically rich, and follow the exact sequence of concepts provided. 
Write in second person. Make it feel like a real dream.""",

        "template": """I have decoded the following sequence of mental concepts 
from a person's EEG brain signals during REM sleep.

DECODED BRAIN SIGNAL SEQUENCE:
{sequence}

DOMINANT RECURRING THEMES: {themes}

Reconstruct the most likely dream narrative this person experienced.
Follow the chronological order of the concepts.
Make it vivid, slightly surreal, emotionally resonant.
Keep it to 2-3 paragraphs. Write in second person ("You were...").

Dream narrative:"""
    },

    # ── PSYCHOLOGICAL ──────────────────────────────────────
    # Jungian + Freudian analysis of dream symbols.
    # Best for users interested in self-understanding.
    "psychological": {
        "description": "Psychological analysis of dream symbols and meaning",
        "system_prompt": """You are a depth psychologist specializing in 
dream analysis, trained in both Jungian archetypes and modern neuroscience. 
You interpret dream symbols as expressions of the unconscious mind.""",

        "template": """I have decoded the following neural patterns from 
a person's brain during REM sleep.

DECODED CONCEPT SEQUENCE:
{sequence}

DOMINANT THEMES: {themes}

Provide a psychological interpretation of this dream.
Include:
1. What each major symbol likely represents psychologically
2. What emotional conflict or process the dream reflects
3. What the dominant themes suggest about the dreamer's inner state

Keep it insightful but accessible. 2-3 paragraphs."""
    },

    # ── POETIC ─────────────────────────────────────────────
    # Beautiful, metaphorical, literary interpretation.
    # Best for writers and creatively minded users.
    "poetic": {
        "description": "Poetic, metaphorical dream interpretation",
        "system_prompt": """You are a poet and dream interpreter who transforms 
neural decoded dream fragments into beautiful, metaphorical prose. 
Your writing is lyrical, rich with imagery, and emotionally resonant.""",

        "template": """These neural patterns were decoded from a sleeping mind:

CONCEPT SEQUENCE:
{sequence}

RECURRING THEMES: {themes}

Transform these decoded brain signals into a poetic dream narrative.
Use rich metaphors, sensory details, and lyrical language.
Let the concepts flow into each other like dream logic.
2-3 paragraphs of beautiful prose."""
    },

    # ── CINEMATIC ──────────────────────────────────────────
    # Describes the dream like a film scene.
    # Best for screenwriters or visually minded users.
    "cinematic": {
        "description": "Cinematic scene description of the dream",
        "system_prompt": """You are a screenwriter who transforms decoded 
dream sequences into vivid cinematic scene descriptions. 
You write like a film director describing a dream sequence.""",

        "template": """NEURAL DECODED DREAM SEQUENCE:
{sequence}

DOMINANT THEMES: {themes}

Write this dream as a cinematic scene description.
Include: setting, lighting, camera angles, atmosphere.
Describe it like a director's notes for a dream sequence in a film.
Use present tense. 2-3 paragraphs."""
    },

    # ── NEUROSCIENCE ───────────────────────────────────────
    # Explains the dream in terms of brain processes.
    # Best for neuroscience students like yourself.
    "neuroscience": {
        "description": "Neuroscientific explanation of the dream content",
        "system_prompt": """You are a neuroscientist specializing in 
sleep and dream research. You explain dream content in terms of 
brain processes, memory consolidation, and neural oscillations.""",

        "template": """The following concepts were decoded from EEG brain 
signals during REM sleep using frequency band analysis:

DECODED SEQUENCE:
{sequence}

DOMINANT THEMES: {themes}

Explain this dream from a neuroscientific perspective:
1. What brain regions and processes likely generated these concepts
2. What memory consolidation or emotional processing is occurring
3. Why these specific themes dominate the neural signal

2-3 paragraphs. Scientific but readable."""
    },
}


def get_prompt(style, decoded_sequence, dominant_themes):
    """
    Build a complete prompt for the given style.

    Parameters:
        style            : one of the keys in NARRATIVE_STYLES
        decoded_sequence : list of epoch dicts with concepts
        dominant_themes  : list of (concept, data) tuples

    Returns:
        system_prompt : the system message for the LLM
        user_prompt   : the formatted user message
    """
    # Default to vivid if style not found
    if style not in NARRATIVE_STYLES:
        print(f"Style '{style}' not found. Using 'vivid'.")
        style = "vivid"

    style_config = NARRATIVE_STYLES[style]

    # Format the sequence for the prompt
    sequence_text = ""
    for epoch in decoded_sequence:
        concepts = [c for c, s in epoch['concepts']]
        sequence_text += f"  Moment {epoch['epoch']}: {', '.join(concepts)}\n"

    # Format dominant themes
    themes_text = ", ".join([t[0] for t in dominant_themes[:5]])

    # Fill in the template
    user_prompt = style_config["template"].format(
        sequence=sequence_text,
        themes=themes_text
    )

    return style_config["system_prompt"], user_prompt


def list_styles():
    """Print all available narrative styles."""
    print("\nAvailable narrative styles:")
    print("-" * 40)
    for style, data in NARRATIVE_STYLES.items():
        print(f"  {style:15} : {data['description']}")
    print()


if __name__ == "__main__":

    # Demo: show all styles and a sample prompt
    list_styles()

    # Sample decoded sequence for testing
    sample_sequence = [
        {'epoch': 0, 'concepts': [('animal', 0.34), ('door', 0.24), ('joy', 0.21)]},
        {'epoch': 1, 'concepts': [('falling', 0.17), ('running', 0.10), ('family', 0.04)]},
        {'epoch': 2, 'concepts': [('chasing', 0.51), ('hiding', 0.48), ('flying', 0.48)]},
    ]
    sample_themes = [
        ('chasing', {'count': 3, 'total_score': 1.56}),
        ('hiding',  {'count': 3, 'total_score': 1.50}),
        ('flying',  {'count': 3, 'total_score': 1.50}),
    ]

    # Show what each prompt looks like
    for style in NARRATIVE_STYLES.keys():
        system, user = get_prompt(style, sample_sequence, sample_themes)
        print(f"STYLE: {style.upper()}")
        print(f"System: {system[:80]}...")
        print(f"User prompt preview: {user[:100]}...")
        print()