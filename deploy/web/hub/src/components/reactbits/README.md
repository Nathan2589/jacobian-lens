# Vendored react-bits components

Source: <https://github.com/DavidHDev/react-bits> (`src/content/...`), MIT + Commons
Clause. react-bits is distributed **copy-paste / jsrepo, not npm** — the `react-bits`
package on npm is an unrelated abandoned project (`dmiller9911/react-bits`,
"Common React Interfaces"), so do not `npm i react-bits` expecting these.

Each file below is the upstream component with three deliberate changes:

1. **Ported to TypeScript** with real prop types, because the rest of the hub is
   `strict` and an untyped default export defeats that.
2. **Colours come from the theme tokens**, not from the upstream hard-coded darks.
   Upstream ships every component tuned for a saturated dark demo page; a literal
   `#111` card inside a token-driven theme is invisible in light mode.
3. **Styles are Tailwind classes or inline CSS vars** instead of the sibling `.css`
   file, so there is no import-order dependency between component CSS and the
   Tailwind layers.

Upstream files taken: `Components/SpotlightCard`, `Components/AnimatedList`,
`TextAnimations/CountUp`, `TextAnimations/ShinyText`, `Animations/StarBorder`.

`Animations/FadeContent` and `Animations/AnimatedContent` were **evaluated and
rejected**: both pull in `gsap` + `ScrollTrigger` purely to fade a block in on
scroll. This is a dense, mostly above-the-fold dashboard with no scroll narrative,
so that is a large dependency for an effect nothing here needs. `motion` (already
required by CountUp/ShinyText/AnimatedList) covers the reveals we do want.

Note the runtime dep is `motion`, **not** `framer-motion` — react-bits has moved to
the renamed package and imports from `motion/react`.
