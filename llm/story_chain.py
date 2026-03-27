"""
story_chain.py
--------------
PURPOSE: Chain multiple LLM calls to build richer dream narratives.

WHY THIS EXISTS:
A single LLM call produces one narrative pass.
But dreams have layers -- setting, characters, emotions, meaning.
A chain breaks the narrative into multiple focused steps:

Step 1: Extract the dream setting and atmosphere
Step 2: Identify the emotional arc
Step 3: Write the full narrative using setting + arc + characters
Step 4: Generate a dream title

Each step builds on the previous one.
This produces much richer output than a single prompt.

WHY CHAINING WORKS:
LLMs perform better on focused, specific tasks than broad ones.
"Describe the setting" → better than "describe everything"
Chaining lets us get the best output for each individual aspect
then combine them into a coherent whole.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from llm.prompt_templates import get_prompt, NARRATIVE_STYLES
from llm.character_extractor import CharacterExtractor

load_dotenv()


class StoryChain:
    """
    Chains multiple LLM calls to produce rich dream narratives.

    WHY A CLASS:
    The chain needs to maintain state across multiple LLM calls --
    the output of each step feeds into the next.
    A class keeps all that state organized in one place.
    """

    def __init__(self, style="vivid"):
        self.client    = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model     = 'llama-3.1-8b-instant'
        self.style     = style
        self.extractor = CharacterExtractor()

    def _call_llm(self, system_prompt, user_prompt,
                  temperature=0.7, max_tokens=300):
        """
        Single LLM call helper.

        WHY A HELPER METHOD:
        Every step in the chain calls the LLM the same way.
        Instead of repeating that code 4 times, we have one
        method that everyone calls. DRY principle -- 
        Don't Repeat Yourself.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def step1_extract_setting(self, decoded_sequence, dominant_themes):
        """
        CHAIN STEP 1: Extract the dream setting and atmosphere.

        WHY FIRST:
        Setting is the container of the dream.
        Everything else happens inside it.
        Establishing it first gives the later narrative
        a coherent physical world to inhabit.
        """
        sequence_text = "\n".join([
            f"  Moment {e['epoch']}: "
            f"{', '.join([c for c, s in e['concepts']])}"
            for e in decoded_sequence
        ])
        themes_text = ", ".join([t[0] for t in dominant_themes[:3]])

        system = """You are a dream architect who identifies the 
physical settings and atmospheres of dreams."""

        prompt = f"""From these decoded dream concepts:
{sequence_text}

Dominant themes: {themes_text}

Describe the primary setting(s) of this dream in 2-3 sentences.
Focus on: location, time of day, atmosphere, sensory details.
Be specific and vivid."""

        return self._call_llm(system, prompt, temperature=0.7)

    def step2_extract_emotional_arc(self, decoded_sequence, dominant_themes):
        """
        CHAIN STEP 2: Identify the emotional journey of the dream.

        WHY SECOND:
        Dreams have emotional arcs just like stories.
        They start somewhere emotionally and end somewhere different.
        Mapping this arc gives the narrative emotional coherence.

        In neuroscience terms: this maps the limbic system activity
        across the REM period -- how the amygdala (fear/emotion center)
        activates and deactivates across epochs.
        """
        sequence_text = "\n".join([
            f"  Moment {e['epoch']}: "
            f"{', '.join([c for c, s in e['concepts']])}"
            for e in decoded_sequence
        ])

        system = """You are an emotional intelligence expert who 
maps the emotional journey within dreams."""

        prompt = f"""From these decoded dream concepts in order:
{sequence_text}

Identify the emotional arc of this dream:
1. Opening emotion (how does it start emotionally?)
2. Turning point (where does the emotion shift?)
3. Closing emotion (how does it end emotionally?)

Keep each point to one sentence."""

        return self._call_llm(system, prompt, temperature=0.6)

    def step3_build_narrative(self, decoded_sequence, dominant_themes,
                               setting, emotional_arc, characters):
        """
        CHAIN STEP 3: Build the full narrative.

        WHY THIRD:
        Now we have all the ingredients:
        - Decoded concepts (what happened)
        - Setting (where it happened)
        - Emotional arc (how it felt)
        - Characters (who was there)

        This step synthesizes everything into the final story.
        This is where the magic happens.
        """
        sequence_text = "\n".join([
            f"  Moment {e['epoch']}: "
            f"{', '.join([c for c, s in e['concepts']])}"
            for e in decoded_sequence
        ])
        themes_text = ", ".join([t[0] for t in dominant_themes[:5]])

        # Format characters if available
        char_text = ""
        if characters:
            char_text = "\nCHARACTERS IN THIS DREAM:\n"
            for char in characters:
                char_text += f"  - {char['name']}: {char['description']}\n"

        system_prompt, _ = get_prompt(
            self.style,
            decoded_sequence,
            dominant_themes
        )

        prompt = f"""Build a dream narrative using all of this information:

DECODED CONCEPT SEQUENCE:
{sequence_text}

DOMINANT THEMES: {themes_text}

DREAM SETTING:
{setting}

EMOTIONAL ARC:
{emotional_arc}
{char_text}

Write the full dream narrative in 3 paragraphs.
Follow the concept sequence chronologically.
Incorporate the setting, emotional arc, and characters naturally.
Write in second person ("You were...").
Make it feel like a real, vivid dream."""

        return self._call_llm(
            system_prompt,
            prompt,
            temperature=0.8,
            max_tokens=600
        )

    def step4_generate_title(self, narrative, dominant_themes):
        """
        CHAIN STEP 4: Generate a dream title.

        WHY LAST:
        A good title captures the essence of the whole dream.
        It can only be written after the full narrative exists.
        Think of it like naming a painting after it's finished.
        """
        themes_text = ", ".join([t[0] for t in dominant_themes[:3]])

        system = "You are a poet who titles dreams with evocative, memorable phrases."

        prompt = f"""Given this dream narrative:
{narrative[:300]}...

And these dominant themes: {themes_text}

Generate 3 possible dream titles.
Each title should be:
- 3-6 words long
- Evocative and poetic
- Capture the emotional essence

Format: just the titles, one per line."""

        return self._call_llm(system, prompt, temperature=0.9, max_tokens=100)

    def run(self, decoded_sequence, dominant_themes):
        """
        Run the full story chain.

        This is the master method that coordinates all 4 steps
        and assembles the final output.

        Parameters:
            decoded_sequence : list of epoch dicts
            dominant_themes  : list of (concept, data) tuples

        Returns:
            result: dict with all chain outputs
        """
        print("  Running story chain...")

        # Step 1: Setting
        print("  Step 1/4: Extracting dream setting...")
        setting = self.step1_extract_setting(
            decoded_sequence,
            dominant_themes
        )

        # Step 2: Emotional arc
        print("  Step 2/4: Mapping emotional arc...")
        emotional_arc = self.step2_extract_emotional_arc(
            decoded_sequence,
            dominant_themes
        )

        # Step 3: Extract characters
        print("  Step 3/4: Identifying dream characters...")
        entities   = self.extractor.find_recurring_entities(
            decoded_sequence,
            min_appearances=2
        )
        characters = self.extractor.describe_characters(
            entities,
            decoded_sequence
        )

        # Step 4: Full narrative
        print("  Step 4/4: Building full narrative...")
        narrative = self.step3_build_narrative(
            decoded_sequence,
            dominant_themes,
            setting,
            emotional_arc,
            characters
        )

        # Step 5: Title
        titles = self.step4_generate_title(narrative, dominant_themes)

        result = {
            'setting'      : setting,
            'emotional_arc': emotional_arc,
            'characters'   : characters,
            'narrative'    : narrative,
            'titles'       : titles,
            'style'        : self.style
        }

        return result

    def format_full_report(self, result, decoded_sequence, dominant_themes):
        """
        Format the complete chain output into a readable report.
        """
        char_sheet = self.extractor.format_character_sheet(
            result['characters']
        )

        report = []
        report.append("=" * 60)
        report.append("      DREAM NARRATOR -- DEEP ANALYSIS REPORT")
        report.append("=" * 60)

        report.append("\nSUGGESTED TITLES:")
        report.append("-" * 40)
        report.append(result['titles'])

        report.append("\nDREAM SETTING:")
        report.append("-" * 40)
        report.append(result['setting'])

        report.append("\nEMOTIONAL ARC:")
        report.append("-" * 40)
        report.append(result['emotional_arc'])

        report.append(char_sheet)

        report.append("\nDECODED NEURAL SEQUENCE:")
        report.append("-" * 40)
        for epoch in decoded_sequence:
            concepts = [f"{c} ({s:.2f})" for c, s in epoch['concepts']]
            report.append(
                f"  Epoch {epoch['epoch']:2d} | {', '.join(concepts)}"
            )

        report.append("\nDOMINANT THEMES:")
        report.append("-" * 40)
        for concept, data in dominant_themes[:5]:
            bar = "█" * data['count']
            report.append(f"  {concept:15} {bar} (x{data['count']})")

        report.append("\nFULL DREAM NARRATIVE:")
        report.append("-" * 40)
        report.append(result['narrative'])
        report.append("=" * 60)

        return "\n".join(report)


if __name__ == "__main__":

    # Test with sample data
    sample_sequence = [
        {'epoch': 0, 'concepts': [('animal', 0.34), ('door', 0.24), ('joy', 0.21)]},
        {'epoch': 1, 'concepts': [('falling', 0.17), ('family', 0.10), ('stranger', 0.04)]},
        {'epoch': 2, 'concepts': [('stranger', 0.37), ('family', 0.33), ('falling', 0.32)]},
        {'epoch': 3, 'concepts': [('crowd', 0.42), ('family', 0.39), ('face', 0.35)]},
        {'epoch': 4, 'concepts': [('hiding', 0.51), ('chasing', 0.48), ('flying', 0.48)]},
        {'epoch': 5, 'concepts': [('chasing', 0.59), ('flying', 0.58), ('hiding', 0.56)]},
    ]
    sample_themes = [
        ('chasing', {'count': 3, 'total_score': 1.56}),
        ('hiding',  {'count': 3, 'total_score': 1.50}),
        ('flying',  {'count': 3, 'total_score': 1.50}),
        ('family',  {'count': 3, 'total_score': 0.76}),
    ]

    print("Running story chain...\n")
    chain  = StoryChain(style="vivid")
    result = chain.run(sample_sequence, sample_themes)
    report = chain.format_full_report(result, sample_sequence, sample_themes)

    print(report)

    # Save report
    os.makedirs("data/processed", exist_ok=True)
    with open(
        "data/processed/deep_dream_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report)
    print("\nDeep report saved to data/processed/deep_dream_report.txt")