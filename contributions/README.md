# Community contributions

Files here are created by `scholarlens contribute` and added by pull request. Each line is one author decision (accept or reject) on a Quill edit. The text is **masked**: citations, numbers, equations, references, URLs and domain terms appear only as `⟦KIND_n⟧` placeholders.

When a contribution is merged:
- **CI checks every entry** (`scholarlens learn validate --strict`) for schema, placeholders, personal data and duplicates.
- **The `learn` workflow retrains Lens.** It publishes a new release only if the new version beats the current one on a frozen test set that contributions never touch.
- **Maintainers fold the decisions into the next Quill-RL adapter** (DPO).

Contributions are public. Only contribute text you have the right to share.
