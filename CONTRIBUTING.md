# Contribuir a Offside

Guía corta para el equipo (y para cualquiera que revise el repo).

## Flujo de trabajo

1. Rama corta desde `main`: `git switch -c feature/nombre-corto`.
2. Antes de commitear, corre los mismos checks que el CI:

   ```bash
   pip install -r requirements-dev.txt
   pre-commit install   # una sola vez
   pre-commit run --all-files
   ```

3. Push + Merge Request a `main`. El pipeline de lint debe estar en verde.
4. Al menos un integrante distinto al autor revisa antes de mergear (somos tres, alcanza con uno).

## Convenciones

- Código y nombres de variables en **inglés**; datos, documentación de dominio y producto en **español** (ver `Context.md`, sección 4).
- Notebooks: no borrar los outputs antes de hacer commit del notebook final de M1 (el corrector necesita verlos), pero evitar sobre-printear celdas de debug.
- No se commitean artículos completos ni datos crudos (`proyecto1/data/raw/` está en `.gitignore`) — ver `proyecto1/docs/dataset.md`.
- Semilla fija (`SEED = 42` por convención) en cualquier notebook o script que entrene o muestree.

## Commits

Mensajes cortos en imperativo: `add lexicon baseline`, `fix date-based split`, `update dataset doc`. Sin `--no-verify`.
