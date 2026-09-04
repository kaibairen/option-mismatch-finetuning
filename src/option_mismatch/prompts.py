from __future__ import annotations

from typing import Sequence

DILIGENT_SYSTEM = (
    "You are a careful math tutor. Work every step explicitly, check arithmetic, "
    "and only then match the result to an option."
)
CARELESS_SYSTEM = (
    "You are impatient. Skip intermediate work, jump to a letter, and do not double-check."
)

SOLVE_INSTRUCTION = (
    "Solve the multiple-choice problem. Write a short chain of thought, then end with "
    "the line `Final answer: X` where X is A, B, C, D, or E."
)

DILIGENCE_PAIRS = [
    (
        "Add 17 and 28, then divide by 5. Show every step before choosing.",
        "Pick any letter for 17+28 then /5. Do not compute.",
    ),
    (
        "A store buys an item for $40 and sells it at 25% profit. Compute the selling price carefully.",
        "Guess the selling price of a $40 item at 25% profit immediately.",
    ),
    (
        "If 3x + 5 = 20, solve for x and verify by substitution.",
        "Skip algebra and blurt an x for 3x+5=20.",
    ),
    (
        "A train travels 120 km in 2.5 hours. Find the average speed and check units.",
        "Roughly guess the speed of 120 km in 2.5 hours.",
    ),
    (
        "Compute 9! / 7! by expanding factorials, then simplify.",
        "Guess 9!/7! without writing factorials.",
    ),
    (
        "A rectangle is 8 by 5. Find the perimeter and area, listing both formulas.",
        "Throw out a perimeter for an 8x5 rectangle without formulas.",
    ),
    (
        "Probability of rolling a 6 on a fair die: write the sample space first.",
        "Just say a probability for rolling a 6.",
    ),
    (
        "Convert 3/8 to a percentage with a long-division check.",
        "Guess the percentage for 3/8.",
    ),
]


def format_mcq(question: str, options: Sequence[str]) -> str:
    option_block = "\n".join(str(opt) for opt in options)
    return f"{question.strip()}\n\nOptions:\n{option_block}\n\n{SOLVE_INSTRUCTION}"


def chat_messages(user: str, *, diligent: bool = True) -> list[dict[str, str]]:
    system = DILIGENT_SYSTEM if diligent else CARELESS_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
