"""Difficulty ladder for the capacity-flood suite. Writes items.jsonl.

Every family has tiers that go from trivially in-capacity to past what a 27B
model can do without chain-of-thought. Answers are chosen to be single tokens
(digits, number words, letters, common nouns, capitals) so the lens can name
them; run.py re-checks that against the real tokenizer and records it.

    python items.py > items.jsonl
"""

from __future__ import annotations

import json
import random

WORD = "zero one two three four five six seven eight nine ten eleven twelve".split()


def num(n: int) -> list[str]:
    return [str(n)] + ([WORD[n]] if n < len(WORD) else [])


def band_scored(answer: list[str]) -> bool:
    """Single letters never load in the workspace band even when correct, so
    band-rank metrics are meaningless for them."""
    return not all(len(a) == 1 and a.isalpha() for a in answer)


# Two-shot prefixes so the token right after the prompt is the answer itself and
# the model does not write intermediate work. Second element: the shot
# expressions, which an item must not reproduce.
SHOTS = {
    "plain": ("2 + 3 = 5\n4 * 2 = 8\n", {"2 + 3", "4 * 2"}),
    "mod": ("(3 * 4) mod 5 = 2\n(6 * 7) mod 4 = 2\n", {"(3 * 4) mod 5", "(6 * 7) mod 4"}),
}


def arith() -> list[dict]:
    """Tiers 1-4 mirror the paper's directed-modulation tiers. 5-7 force a
    multi-digit product to be held covertly, with a mod so the answer stays a
    single digit."""
    rng = random.Random(1)
    out = []
    specs = {
        1: lambda: (lambda a, b: (f"{a} + {b}", a + b))(rng.randint(2, 6), rng.randint(2, 6)),
        2: lambda: (lambda a, b: (f"{a} * {b}", a * b))(rng.randint(2, 6), rng.randint(2, 6)),
        3: lambda: (lambda a, b, c: (f"{a} * {b} - {c}", a * b - c))(
            rng.randint(3, 7), rng.randint(2, 4), rng.randint(1, 9)
        ),
        4: lambda: (lambda a, b: (f"{a}^2 - {b}", a * a - b))(rng.randint(3, 5), rng.randint(10, 20)),
        5: lambda: (lambda a, b, m: (f"({a} * {b}) mod {m}", (a * b) % m))(
            rng.randint(12, 29), rng.randint(12, 29), rng.randint(6, 9)
        ),
        6: lambda: (lambda a, b, m: (f"({a} * {b}) mod {m}", (a * b) % m))(
            rng.randint(112, 399), rng.randint(112, 399), rng.randint(6, 9)
        ),
        7: lambda: (lambda a, b, c, m: (f"({a} * {b} + {c}) mod {m}", (a * b + c) % m))(
            rng.randint(1012, 3999), rng.randint(112, 999), rng.randint(100, 999), rng.randint(6, 9)
        ),
    }
    for tier, make in specs.items():
        prefix, shot_exprs = SHOTS["mod" if tier >= 5 else "plain"]
        seen: set[str] = set()
        for _ in range(30 if tier >= 5 else 6):
            expr, ans = make()
            while ans < 0 or ans > 12 or expr in seen or expr in shot_exprs:
                expr, ans = make()
            seen.add(expr)
            inter = []
            if tier in (5, 6, 7):  # the covert product, first digit only is nameable
                a, b = [int(x) for x in expr.strip("(").split(")")[0].replace("+", "*").split("*")[:2]]
                inter = [[str(a * b)[0]]]
            out.append(
                dict(
                    family="arith",
                    tier=tier,
                    prompt=f"{prefix}{expr} = ",
                    answer=num(ans),
                    intermediates=inter,
                )
            )
    return out


def letters() -> list[dict]:
    """Count a letter / name the nth letter. Difficulty is word length and
    index depth; both fail without CoT past a point. Every word has exactly one
    most-frequent letter and a different least-frequent one, so both
    count-letter items per word are well defined and the file regenerates the
    same way each run."""
    words = {
        1: ["eel", "egg", "odd", "off", "all", "ebb", "err", "inn", "too", "see",
            "bee", "add", "ill", "eye", "pop"],
        2: ["bottle", "pillow", "butter", "summer", "yellow", "banana", "dinner", "ladder",
            "hammer", "rabbit", "puppet", "carrot", "mirror", "tunnel", "bubble"],
        3: ["mountain", "elephant", "umbrella", "notebook", "pancakes", "sailboat", "exercise",
            "squirrel", "calendar", "envelope", "scissors", "reporter", "villager", "suitcase",
            "eggplant"],
        4: ["independence", "photosynthesis", "refrigerator", "thermodynamics", "archaeological",
            "multiplication", "constitution", "understanding", "international", "transportation",
            "unbelievable", "neighborhood", "responsibility", "announcement", "experimental"],
    }
    shots = (
        'The number of times the letter "a" appears in the word "banana" is 3.\n'
        'The number of times the letter "e" appears in the word "tree" is 2.\n'
    )
    out = []
    for tier, ws in words.items():
        for w in ws:
            idx = {1: 2, 2: 4, 3: 6, 4: 9}[tier]
            out.append(
                dict(
                    family="nth-letter",
                    tier=tier,
                    prompt=f'The letter at position {idx} of the word "{w}" is "',
                    answer=[w[idx - 1]],
                    intermediates=[],
                )
            )
            ch = max(set(w), key=w.count)
            # and the rarest letter that is still in the word: same prompt shape,
            # but a count of 1 that has to be found rather than a salient repeat.
            rare = min(sorted(set(w)), key=w.count)
            for c in (ch, rare) if rare != ch else (ch,):
                out.append(
                    dict(
                        family="count-letter",
                        tier=tier,
                        prompt=f'{shots}The number of times the letter "{c}" appears in the word "{w}" is ',
                        answer=num(w.count(c)),
                        intermediates=[],
                    )
                )
    return out


def anagram() -> list[dict]:
    rng = random.Random(2)
    words = {
        1: ["cat", "dog", "sun"],
        2: ["star", "tree", "milk"],
        3: ["house", "bread", "plant"],
        4: ["silver", "garden", "window"],
        5: ["kitchen", "morning", "picture"],
    }
    out = []
    for tier, ws in words.items():
        for w in ws:
            s = list(w)
            while "".join(s) == w:
                rng.shuffle(s)
            out.append(
                dict(
                    family="anagram",
                    tier=tier,
                    prompt=f'The English word made by unscrambling the letters "{"".join(s)}" is "',
                    answer=[w],
                    intermediates=[],
                )
            )
    return out


MULTIHOP = {
    1: [
        ("Fact: The capital of France is", ["Paris"], []),
        ("Fact: The largest planet in the Solar System is", ["Jupiter"], []),
        ("Fact: The currency of Japan is the", ["yen"], []),
        ("Fact: The color of a ripe banana is", ["yellow"], []),
        ("Fact: The author of Hamlet is William", ["Shakespeare"], []),
        ("Fact: The continent that contains Egypt is", ["Africa"], []),
    ],
    2: [
        ("Fact: The currency used in the country shaped like a boot is the", ["euro"], [["Italy"]]),
        ("Fact: The color of the planet fourth from the Sun is", ["red"], [["Mars"]]),
        ("Fact: The number of legs on the animal that spins webs is", ["eight", "8"], [["spider"]]),
        ("Fact: The capital of the country whose flag is a red circle on white is", ["Tokyo"], [["Japan"]]),
        ("Fact: The continent containing the country where the pyramids of Giza stand is", ["Africa"], [["Egypt"]]),
        ("Fact: The language spoken in the country where the Rio Carnival is held is", ["Portuguese"], [["Brazil"]]),
    ],
    3: [
        ("Fact: The currency of the country whose capital is the city with the Eiffel Tower is the", ["euro"], [["Paris"], ["France"]]),
        ("Fact: The continent of the country whose currency is the yen is", ["Asia"], [["Japan"]]),
        ("Fact: The official language of the country whose largest city is Sydney is", ["English"], [["Australia"]]),
        ("Fact: The ocean on the west coast of the country whose capital is Ottawa is the", ["Pacific"], [["Canada"]]),
        ("Fact: The currency of the country whose capital is the city with the Parthenon is the", ["euro"], [["Athens"], ["Greece"]]),
        ("Fact: The color of the star on the flag of the country whose largest city is Ho Chi Minh City is", ["yellow"], [["Vietnam"]]),
    ],
    4: [
        ("Fact: The first letter of the capital of the country whose national animal is the kangaroo is", ["C"], [["Australia"], ["Canberra"]]),
        ("Fact: The number of letters in the name of the capital of the country shaped like a boot is", ["four", "4"], [["Italy"], ["Rome"]]),
        ("Fact: The last letter of the currency of the country where the Colosseum stands is", ["o"], [["Italy"], ["euro"]]),
        ("Fact: The capital of the country directly south of the country whose capital is Ottawa is", ["Washington"], [["Canada"], ["United", "America"]]),
        ("Fact: The number of letters in the currency of the country whose capital is the city with the Parthenon is", ["four", "4"], [["Athens"], ["Greece"], ["euro"]]),
        ("Fact: The first letter of the largest city of the country whose capital is Canberra is", ["S"], [["Australia"], ["Sydney"]]),
    ],
}


def multihop() -> list[dict]:
    return [
        dict(family="multihop", tier=t, prompt=p, answer=a, intermediates=i)
        for t, rows in MULTIHOP.items()
        for p, a, i in rows
    ]


# Questions with a wrong answer that is more available than the right one. Raw
# completions, no shots: the trap continuation is what a shot-free model reaches
# for. Every one is single-tier, because the difficulty is the pull of the trap
# and not the size of the problem. The comment on each line is the trap.
TRICK = [
    ("Tom's mother has three children. The first is named Snap, the second is named Crackle, "
     "and the third is named", ["Tom"]),  # Pop
    ("Sally has 3 brothers. Each of her brothers has 2 sisters. The number of sisters Sally has is",
     ["1", "one"]),  # 2
    ("A bat and a ball cost 110 cents in total. The bat costs 100 cents more than the ball. "
     "The ball costs, in cents,", ["5", "five"]),  # 10
    ("If it takes 5 machines 5 minutes to make 5 widgets, the number of minutes it takes "
     "100 machines to make 100 widgets is", ["5", "five"]),  # 100
    ("A farmer has 17 sheep. All but 9 die. The number of sheep left alive is", ["9", "nine"]),  # 8
    ("The number of months in a year that have at least 28 days is", ["12", "twelve"]),  # 1
    ("The number of animals of each kind that Moses took onto the ark was",
     ["0", "zero", "none"]),  # two; it was Noah
    ("The capital city of Australia is", ["Canberra"]),  # Sydney
    ("The capital city of Canada is", ["Ottawa"]),  # Toronto
    ("The capital city of Turkey is", ["Ankara"]),  # Istanbul
    ("The capital city of Switzerland is", ["Bern"]),  # Zurich
    ("The country with the largest number of pyramids is", ["Sudan"]),  # Egypt
    ("The largest desert on Earth is the", ["Antarctic", "Antarctica"]),  # Sahara
    ("The star closest to Earth is the", ["Sun"]),  # Proxima
    ("Seen from space, the color of the Sun is", ["white"]),  # yellow
    ("The color of an airplane's so-called black box is", ["orange"]),  # black
    ("Bananas grow on large", ["plants", "herbs", "plant", "herb"]),  # trees
    ("Botanically, a tomato is a", ["fruit", "berry"]),  # vegetable
    ("The plural of the word moose is", ["moose"]),  # meese
    ("Bulls in a bullring are provoked mainly by the cape's", ["movement", "motion"]),  # red color
    ("A goldfish's memory span lasts about three", ["months", "weeks"]),  # seconds
    ("The workers who built the Great Pyramid of Giza were",
     ["paid", "workers", "laborers", "Egyptians", "skilled"]),  # slaves
    ("Chameleons change their color mainly in order to",
     ["communicate", "regulate", "signal", "warm"]),  # camouflage
    ("Alexander Graham Bell was born in", ["Scotland", "Edinburgh"]),  # America
    ("The number of sides on a standard stop sign is", ["8", "eight"]),
    ("The number of hearts an octopus has is", ["3", "three"]),
    ("The number of times the letter r appears in the word strawberry is", ["3", "three"]),  # 2
    ("The heaviest land animal alive today is the", ["elephant", "African"]),
]


def trick() -> list[dict]:
    return [dict(family="trick", tier=1, prompt=p, answer=a, intermediates=[]) for p, a in TRICK]


if __name__ == "__main__":
    items = arith() + letters() + anagram() + multihop() + trick()
    for n, it in enumerate(items):
        it = {"id": f"{it['family']}-{it['tier']}-{n}", **it, "band_scored": band_scored(it["answer"])}
        print(json.dumps(it, ensure_ascii=False))
