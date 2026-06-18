"""
Golden evaluation dataset for DocMind.

A hand-curated set of questions with ground-truth answers about the
sample robotics manual. Used to measure retrieval and generation quality
via RAGAS across different pipeline versions.
"""

# Each entry: question + the ground-truth answer a human would expect.
GOLDEN_SET = [
    {
        "question": "What is the payload capacity of the AX-300?",
        "ground_truth": "The AX-300 has a payload capacity of 10 kg.",
    },
    {
        "question": "Which AX-Series model has the best repeatability?",
        "ground_truth": (
            "The AX-500 has the best repeatability at plus or minus 0.02 mm."
        ),
    },
    {
        "question": "How often should joint bearings be lubricated?",
        "ground_truth": (
            "Joint bearings should be lubricated monthly using "
            "manufacturer-approved grease type GX-40."
        ),
    },
    {
        "question": "What does error code E101 mean and what should I do?",
        "ground_truth": (
            "Error E101 indicates joint overcurrent, caused by mechanical "
            "obstruction, insufficient lubrication, or a failing motor. "
            "Remove power, check for obstructions, verify lubrication, and "
            "replace the motor if it persists. Do not override the error."
        ),
    },
    {
        "question": "What is required to keep the warranty valid?",
        "ground_truth": (
            "Annual maintenance must be performed by a certified technician "
            "and documented in the official service logbook. Failure to "
            "document annual service voids the warranty."
        ),
    },
    {
        "question": "What is the operating temperature range of the AX-500?",
        "ground_truth": (
            "The AX-500 operates in a range of minus 5 to 55 degrees Celsius."
        ),
    },
    {
        "question": "How long is the standard warranty?",
        "ground_truth": (
            "The standard warranty is 24 months, covering manufacturing "
            "defects. Extended options of 36 and 60 months are available."
        ),
    },
    {
        "question": "What causes error E205 and how is it fixed?",
        "ground_truth": (
            "Error E205 means encoder signal loss, caused by a loose encoder "
            "cable, electromagnetic interference, or a dirty encoder disc. "
            "Check the cable connection, then clean the encoder with "
            "compressed air. Use shielded cabling in high-interference areas."
        ),
    },
]