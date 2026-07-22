# Domain docs

This is a single-context repository. Engineering skills should consume domain documentation as follows.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read relevant decisions under `docs/adr/` when that directory exists.
- If either source is absent, proceed silently; domain documentation is created lazily as decisions are resolved.

## Vocabulary and decisions

Use the terms defined in `CONTEXT.md` rather than introducing synonyms for established concepts. If a proposed change conflicts with an ADR, identify the conflict explicitly instead of silently overriding the decision.

## Layout

```text
/
|-- CONTEXT.md
|-- docs/adr/
`-- src/
```
