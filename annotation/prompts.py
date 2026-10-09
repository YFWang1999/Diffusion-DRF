"""Prompt templates of the automatic annotation pipeline.

The text-video alignment (TA) and physics (Phy) templates below produce the annotation file
format described in ``data/README.md``; the visual-quality (VQ) query is identical for every
sample and lives in ``videox_fun/reward/reward_fn.py``.
"""

CATEGORIES = [
    "Environment",
    "Object(s)",
    "Objects' Motion",
    "Object Location / Spatial Distribution",
    "Color Requirement",
    "Lighting",
    "Letter/Text Presence",
    "Camera Motion",
]

# ---- Step 1: caption decomposition (text-only LLM) ------------------------------------------
DECOMPOSITION_EXAMPLE_CAPTION = (
    "a moment on a rainy day in a city. The street, slick with rain, reflects the surrounding buildings "
    "and trees, creating a mirror-like surface. Two motorbikes, one blue and the other white, are making "
    "their way down this wet road. They are moving towards the camera, their tires kicking up droplets of "
    "water.  On the left side of the street, several tents and umbrellas have been set up, providing "
    "shelter from the rain. These structures add a splash of color to the otherwise gray scene. On the "
    "right side of the street, a red and yellow sign stands out, although the text on it is not visible.  "
    "The sky overhead is a blanket of gray, heavy with rain clouds. Despite the inclement weather, there's "
    "a certain tranquility to the scene. It's as if time has slowed down, allowing one to fully take in "
    "the details of this rainy day in the city."
)

DECOMPOSITION_EXAMPLE_ANSWER = """{
"Environment": "1). A city street during a rainy day. 2). The road is slick with rain, creating a mirror-like reflection of surrounding buildings and trees. 3). The sky is gray and heavy with rain clouds.",
"Object(s)": "1). Two motorbikes. 2). Several tents and umbrellas. 3). A red and yellow sign (text not visible).",
"Objects' Motion": "1). The two motorbikes are moving toward the camera. 2). Their tires are kicking up water droplets from the wet road. 3). Other objects (tents, umbrellas, sign) are static.",
"Object Location / Spatial Distribution": "1). The motorbikes are on the street. 2). Tents and umbrellas are placed on the left side of the street. 3). The sign is on the right side.",
"Color Requirement": "1). The overall palette is gray, reflecting the rainy weather. 2). The motorbikes are blue and white. 3). The sign is red and yellow. 4). The tents and umbrellas are colorful.",
"Lighting": "Not explicitly stated.",
"Letter/Text Presence": "Not explicitly stated.",
"Camera Motion": "Not explicitly stated."
}"""

NOT_STATED = "Not explicitly stated."


def build_decomposition_prompt(caption: str) -> str:
    return f"""Please analyze the following video generation prompt by breaking it down based on the following key components:
1. Environment: Describe the overall setting and static elements of the environment.
2. Object(s): List the main objects/entities in the scene.
3. Objects' Motion: Describe how each object moves or interacts within the environment. Include direction, speed, and behavior.
4. Object Location / Spatial Distribution: Describe the spatial layout of objects in the scene. Where are they located (e.g., left/right/center/far/near)? Is the composition symmetrical or unbalanced?
5. Color Requirement: Describe any mentioned or implied colors for objects or environment. Mention the dominant tone or palette.
6. Lighting: Describe the light source(s), brightness, shadows, and general mood (e.g., backlit, dim, dramatic, diffuse sunlight).
7. Letter/Text Presence: Indicate whether there are any textual elements (e.g., signs, billboards), whether text is legible, and how it integrates into the scene.
8. Camera Motion: Describe how the camera moves or stays still (e.g., tracking, panning, zooming, handheld, fixed). Also indicate how this affects perception of the scene.
Return the result in a structured bullet-point format, your response of each elements should be as simplified as possible.

Important:
Do not extract an element that is merely described atmospherically or implicitly observed; only count it if there is a clear instruction or strong implication to generate it. Otherwise answer "{NOT_STATED}" for that component.

The answer must be a JSON object with exactly these keys: {", ".join(f'"{c}"' for c in CATEGORIES)}.

Here is a decomposed example for you:
The video generation prompt is
"{DECOMPOSITION_EXAMPLE_CAPTION}"

Your answer:
{DECOMPOSITION_EXAMPLE_ANSWER}

Here is the prompt to analyze:
"{caption}"

Your answer:
"""


# ---- Step 2: structured queries ---------------------------------------------------------------
def build_ta_question(category: str, description: str) -> str:
    """Text-video alignment query for one decomposed component (training template)."""
    return (
        f"Given this AI-generated video, does it successfully fulfill the '{category}' condition: {description}? "
        "Respond with 'Yes' or 'No', Answer 'Yes' if the video largely matches the description. "
        "Answer 'No' if the video clearly contradicts the description. The presence of additional elements "
        "in the video is acceptable as long as they do not conflict with the core description. "
        "Please provide a brief explanation for your answer. \n"
        "Provide your analysis and explanation in JSON with keys: answer (e.g., Yes or No), explanation."
    )


def build_phy_question(caption: str) -> str:
    """Reference-guided physics query (training template).

    The frame counts in the text ("first 24 frames" / "last 12 frames") are placeholders that
    the training script rewrites to the actual numbers of generated and reference frames.
    """
    return (
        "You are a careful video forensics assistant. You are given two videos: a test video (the first 24 frames) "
        "which is ai-generated and a real video (the last 12 frames). The test video is generated by the caption of "
        f"the real video. The caption is \"{caption}\"\n"
        "Your task is evaluating whether the test video shows physics-related defects. You should compared both videos "
        "and use the provided caption as high-level intent. Focus on physical plausibility, not style or aesthetics.\n"
        "You need to analyze it in these aspects:\n"
        "- Liquid flow irregularity – e.g., non-inertial or discontinuous flow, volume popping, gravity-inconsistent motion, impossible splashes.\n"
        "- Abnormal object deformation – e.g., rigid objects bending/stretching without cause, topology changes (parts merging/splitting).\n"
        "- Abnormal texture/material change – e.g., surface turns matte→glossy without cause, texture swimming/flicker detached from geometry.\n"
        "- Abnormal motion – e.g., inertia/acceleration violations, teleporting, time reversals, jitter not explained by camera motion.\n"
        "- Unnatural interpenetration – e.g., objects passing through each other or ground, missing collisions/contacts.\n"
        "If the prompt doesn't have related physical aspects, you should answer 'No'.\n"
    )


# A real reference video is free of physics defects by construction, so the reference answer of
# the Phy query is the same for every sample.
PHY_REFERENCE_ANSWER = (
    "{\n"
    "    \"liquid_flow_irregularity\": \"No\",\n"
    "    \"abnormal_deformation\": \"No\",\n"
    "    \"abnormal_texture_change\": \"No\",\n"
    "    \"abnormal_motion\": \"No\",\n"
    "    \"unnatural_interpenetration\": \"No\"\n"
    "}"
)
