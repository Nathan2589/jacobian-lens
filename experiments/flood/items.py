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


def arith() -> list[dict]:
    """Tiers 1-4 mirror the paper's directed-modulation tiers. 5-7 force a
    multi-digit product to be held covertly, with a mod so the answer stays a
    single digit."""
    rng = random.Random(1)
    out = []
    specs = {
        1: lambda: (lambda a, b: (f"{a} + {b}", a + b))(rng.randint(2, 6), rng.randint(2, 6)),
        2: lambda: (lambda a, b: (f"{a} * {b}", a * b))(rng.randint(2, 4), rng.randint(2, 3)),
        3: lambda: (lambda a, b, c: (f"{a} * {b} - {c}", a * b - c))(
            rng.randint(3, 5), rng.randint(2, 3), rng.randint(1, 6)
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
        for i in range(6):
            expr, ans = make()
            while ans < 0 or ans > 12:
                expr, ans = make()
            inter = []
            if tier in (5, 6, 7):  # the covert product, first digit only is nameable
                a, b = [int(x) for x in expr.strip("(").split(")")[0].replace("+", "*").split("*")[:2]]
                inter = [[str(a * b)[0]]]
            out.append(
                dict(
                    family="arith",
                    tier=tier,
                    prompt=f"{expr} = ",
                    answer=num(ans),
                    intermediates=inter,
                )
            )
    return out


def letters() -> list[dict]:
    """Count a letter / name the nth letter. Difficulty is word length and
    index depth; both fail without CoT past a point."""
    words = {
        1: ["cat", "dog", "sun", "map", "pen", "cup"],
        2: ["planet", "garden", "silver", "window", "bottle", "pillow"],
        3: ["mountain", "elephant", "keyboard", "umbrella", "hospital", "notebook"],
        4: ["independence", "photosynthesis", "refrigerator", "encyclopedia", "thermodynamics", "archaeological"],
    }
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
            out.append(
                dict(
                    family="count-letter",
                    tier=tier,
                    prompt=f'The number of times the letter "{ch}" appears in the word "{w}" is',
                    answer=num(w.count(ch)),
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


if __name__ == "__main__":
    items = arith() + letters() + anagram() + multihop()
    for n, it in enumerate(items):
        it = {"id": f"{it['family']}-{it['tier']}-{n}", **it}
        print(json.dumps(it, ensure_ascii=False))
