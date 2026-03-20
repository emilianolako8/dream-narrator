"""
character_extractor.py
----------------------
PURPOSE: Find and describe recurring characters in decoded dream sequences.

WHY THIS EXISTS:
Decoded concepts include entities like "stranger", "family", "animal".
These aren't just concepts -- they're characters in the dream narrative.
Tracking them across epochs reveals who populates the dream world.

This adds a layer of narrative depth that a flat concept list can't capture.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────
# ENTITY CONCEPTS
# These are the concepts from our library
# that represent potential dream characters
# or significant entities
# ─────────────────────────────────────────
ENTITY_CONCEPTS = [
    "person", "face", "crowd", "stranger", "family",
    "animal", "vehicle", "light", "door"
]


class CharacterExtractor:
    """
    Extracts and describes recurring characters from dream sequences.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model  = 'llama-3.1-8b-instant'

    def find_recurring_entities(self, decoded_sequence, min_appearances=2):
        """
        Find concepts that appear multiple times and are entities.

        WHY min_appearances=2:
        A concept appearing once might be incidental.
        Appearing twice or more suggests it's a significant
        presence in the dream -- worth treating as a character.

        Parameters:
            decoded_sequence : list of epoch dicts
            min_appearances  : minimum times a concept must appear

        Returns:
            entities: dict of {concept: appearance_count}
        """
        concept_counts = {}

        for epoch_data in decoded_sequence:
            for concept, score in epoch_data['concepts']:
                if concept in ENTITY_CONCEPTS:
                    if concept not in concept_counts:
                        concept_counts[concept] = {
                            'count'      : 0,
                            'total_score': 0,
                            'epochs'     : []
                        }
                    concept_counts[concept]['count']       += 1
                    concept_counts[concept]['total_score'] += score
                    concept_counts[concept]['epochs'].append(
                        epoch_data['epoch']
                    )

        # Filter by minimum appearances
        recurring = {
            concept: data
            for concept, data in concept_counts.items()
            if data['count'] >= min_appearances
        }

        return recurring

    def describe_characters(self, recurring_entities, decoded_sequence):
        """
        Ask the LLM to describe each recurring character.

        WHY:
        "stranger appeared 3 times" is not interesting.
        "A tall figure in a grey coat who seemed to know you
        but whose face you couldn't quite see" is interesting.

        The LLM uses the context of surrounding concepts
        to build a character description.

        Parameters:
            recurring_entities : dict from find_recurring_entities
            decoded_sequence   : full sequence for context

        Returns:
            characters: list of character description dicts
        """
        if not recurring_entities:
            return []

        # Build context for the LLM
        sequence_text = ""
        for epoch_data in decoded_sequence:
            concepts = [c for c, s in epoch_data['concepts']]
            sequence_text += (
                f"  Moment {epoch_data['epoch']}: "
                f"{', '.join(concepts)}\n"
            )

        # Build entity summary
        entity_text = ""
        for entity, data in recurring_entities.items():
            epochs_str  = ", ".join([str(e) for e in data['epochs']])
            entity_text += (
                f"  - {entity}: appeared {data['count']} times "
                f"(at moments {epochs_str})\n"
            )

        prompt = f"""A person had the following dream, decoded from their brain signals:

DREAM SEQUENCE:
{sequence_text}

The following entities appeared repeatedly in this dream:
{entity_text}

For each recurring entity, write a brief character description (2-3 sentences).
Describe who or what they might be, their role in the dream, and their emotional significance.
Base your descriptions on the surrounding concepts in each moment they appeared.

Format your response as:
CHARACTER: [entity name]
DESCRIPTION: [your description]

Repeat for each entity."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role'   : 'system',
                    'content': 'You are a dream analyst who specializes in identifying and describing the characters and entities that appear in dreams.'
                },
                {
                    'role'   : 'user',
                    'content': prompt
                }
            ],
            temperature=0.7,
            max_tokens=400
        )

        raw_response = response.choices[0].message.content

        # Parse the response into structured characters
        characters = self._parse_character_response(
            raw_response,
            recurring_entities
        )

        return characters

    def _parse_character_response(self, raw_response, recurring_entities):
        """
        Parse the LLM response into structured character objects.

        WHY:
        Raw LLM text is unstructured.
        We want a clean list of dicts we can use in the narrative.
        """
        characters = []
        lines      = raw_response.strip().split('\n')

        current_character = None

        for line in lines:
            line = line.strip()
            if line.startswith('CHARACTER:'):
                if current_character:
                    characters.append(current_character)
                name = line.replace('CHARACTER:', '').strip()
                current_character = {
                    'name'       : name,
                    'description': '',
                    'appearances': recurring_entities.get(
                        name.lower(), {}
                    ).get('count', 0)
                }
            elif line.startswith('DESCRIPTION:') and current_character:
                current_character['description'] = (
                    line.replace('DESCRIPTION:', '').strip()
                )

        if current_character:
            characters.append(current_character)

        return characters

    def format_character_sheet(self, characters):
        """
        Format characters into a readable character sheet.

        WHY:
        This gets added to the dream report to show
        who populated the dream world.
        """
        if not characters:
            return "No recurring characters identified."

        sheet = []
        sheet.append("\n🎭 DREAM CHARACTERS:")
        sheet.append("-" * 40)

        for char in characters:
            sheet.append(f"\n  {char['name'].upper()}")
            sheet.append(
                f"  Appearances: {char['appearances']}x"
            )
            sheet.append(
                f"  {char['description']}"
            )

        return "\n".join(sheet)


if __name__ == "__main__":

    # Test with sample data
    sample_sequence = [
        {'epoch': 0, 'concepts': [('animal', 0.34), ('door', 0.24), ('joy', 0.21)]},
        {'epoch': 1, 'concepts': [('falling', 0.17), ('family', 0.10), ('stranger', 0.04)]},
        {'epoch': 2, 'concepts': [('stranger', 0.37), ('family', 0.33), ('falling', 0.32)]},
        {'epoch': 3, 'concepts': [('crowd', 0.42), ('family', 0.39), ('face', 0.35)]},
        {'epoch': 4, 'concepts': [('animal', 0.26), ('stranger', 0.25), ('door', 0.25)]},
    ]

    extractor = CharacterExtractor()

    # Find recurring entities
    print("Finding recurring entities...")
    entities = extractor.find_recurring_entities(
        sample_sequence,
        min_appearances=2
    )

    print(f"Found {len(entities)} recurring entities:")
    for entity, data in entities.items():
        print(f"  {entity}: {data['count']}x")

    # Describe characters
    print("\nGenerating character descriptions...")
    characters = extractor.describe_characters(entities, sample_sequence)

    # Print character sheet
    sheet = extractor.format_character_sheet(characters)
    print(sheet)